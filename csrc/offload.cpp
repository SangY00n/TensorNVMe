#include <stdio.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <error.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <sys/uio.h>
#include "offload.h"
#include "space_mgr.h"

// #define _LARGEFILE_SOURCE
// #define _LARGEFILE64_SOURCE
// #define _FILE_OFFSET_BITS 64

iovec *tensors_to_iovec(const std::vector<ull> &data_ptr_list, const std::vector<ull> &nbytes_list)
{
    iovec *iovs = static_cast<iovec *>(calloc(data_ptr_list.size(), sizeof(iovec)));
    for (size_t i = 0; i < data_ptr_list.size(); i++)
    {
        iovs[i].iov_base = reinterpret_cast<void *>(data_ptr_list[i]);
        iovs[i].iov_len = nbytes_list[i];
    }
    return iovs;
}

std::unordered_set<std::string> get_backends()
{
    std::unordered_set<std::string> backends;
#ifndef DISABLE_URING
    backends.insert("uring");
#endif
#ifndef DISABLE_AIO
    backends.insert("aio");
#endif
    return backends;
}

void probe_asyncio(const std::string &backend)
{
    FILE *fp = tmpfile();
    if (!fp)
    {
        printf("Create tmpfile error: %s\n", strerror(errno));
        throw std::runtime_error("uring probe failed\n");
    }
    try
    {
        std::unique_ptr<AsyncIO> aio;
        if (backend == "uring")
#ifndef DISABLE_URING
            aio.reset(new UringAsyncIO(2));
#else
            throw std::runtime_error("backend is not installed\n");
#endif
        else
#ifndef DISABLE_AIO
            aio.reset(new AIOAsyncIO(2));
#else
            throw std::runtime_error("backend is not installed\n");
#endif

        int fd = fileno(fp);
        const int n_loop = 5, n_len = 18;

        char text[n_loop][n_len];

        int offset = 0;
        size_t len;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->write(fd, text[i], len, offset, nullptr);
            offset += len;
        }
        aio->sync_write_events();

        char new_text[n_loop][n_len];
        offset = 0;
        for (int i = 0; i < n_loop; i++)
        {
            len = n_len;
            aio->read(fd, new_text[i], len, offset, nullptr);
            offset += len;
        }
        aio->sync_read_events();
        for (int i = 0; i < n_loop; i++)
        {
            for (int j = 0; j < n_len; j++)
            {
                assert(text[i][j] == new_text[i][j]);
            }
        }
        fclose(fp);
    }
    catch (...)
    {
        fclose(fp);
        throw std::runtime_error("uring probe failed\n");
    }
}

bool probe_backend(const std::string &backend)
{
    std::unordered_set<std::string> backends = get_backends();
    if (backends.find(backend) == backends.end())
        return false;
    try
    {
        probe_asyncio(backend);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

AsyncIO *create_asyncio(unsigned int n_entries, const std::string &backend)
{
    std::unordered_set<std::string> backends = get_backends();
    if (backends.empty())
        throw std::runtime_error("No asyncio backend is installed");
    if (backends.find(backend) == backends.end())
        throw std::runtime_error("Unsupported backend: " + backend);
    if (!probe_backend(backend))
        throw std::runtime_error("Backend \"" + backend + "\" is not install correctly");
#ifndef DISABLE_URING
    if (backend == "uring")
        return new UringAsyncIO(n_entries);
#endif
#ifndef DISABLE_AIO
    if (backend == "aio")
        return new AIOAsyncIO(n_entries);
#endif
    throw std::runtime_error("Unsupported backend: " + backend);
}

Offloader::Offloader(const std::string &filename, unsigned int n_entries, const std::string &backend) : filename(filename), space_mgr(SpaceManager(0))
{
    this->aio = create_asyncio(n_entries, backend);
    this->fd = open(filename.c_str(), O_RDWR | O_CREAT , S_IRUSR | S_IWUSR);
    // this->fd = open(filename.c_str(), O_RDWR | O_CREAT | O_LARGEFILE, S_IRUSR | S_IWUSR); // O_LARGEFILE 넣어야 하나 했는데 안넣어도 되는 듯
    this->aio->register_file(fd); // liburing만 기반 aio인 경우만 register하고, libaio는 안함.
}

SpaceInfo Offloader::prepare_write(const ull nbytes, const std::string &key)
{
    // if (!tensor.is_contiguous() || !tensor.is_cpu())
    //     throw std::runtime_error("Tensor must be contiguous and on cpu");
    ull bytes = nbytes;
    ull offset = this->space_mgr.alloc(bytes);
    SpaceInfo space_info(offset, bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo Offloader::prepare_read(const ull nbytes, const std::string &key)
{
    // if (!tensor.is_contiguous() || !tensor.is_cpu())
    //     throw std::runtime_error("Tensor must be contiguous and on cpu");
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    ull bytes = nbytes;
    SpaceInfo space_info = this->tensors_info[key];
    if (bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

void Offloader::async_write(ull data_ptr, ull nbytes, const std::string &key, callback_t callback)
{
    void *data_ptr_ = reinterpret_cast<void *>(data_ptr);
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(nbytes, key);
    this->aio->write(this->fd, data_ptr_, bytes, offset, callback);

    this->aio->get_event(NOWAIT);
}

void Offloader::async_read(ull data_ptr, ull nbytes, const std::string &key, callback_t callback)
{
    void *data_ptr_ = reinterpret_cast<void *>(data_ptr);
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(nbytes, key);
    auto fn = std::bind(&Offloader::release, this, offset, bytes, callback);
    this->aio->read(this->fd, data_ptr_, bytes, offset, fn);

    this->aio->get_event(NOWAIT);
}

void Offloader::sync_write(ull data_ptr, ull nbytes, const std::string &key)
{
    void *data_ptr_ = reinterpret_cast<void *>(data_ptr);
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_write(nbytes, key);
    lseek(this->fd, offset, SEEK_SET);
    write(this->fd, data_ptr_, bytes);
    // lseek64(this->fd, offset, SEEK_SET);
    // pwrite64(this->fd, data_ptr_, bytes, offset);

}

void Offloader::sync_read(ull data_ptr, ull nbytes, const std::string &key)
{
    void *data_ptr_ = reinterpret_cast<void *>(data_ptr);
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_read(nbytes, key);
    lseek(this->fd, offset, SEEK_SET);
    read(this->fd, data_ptr_, bytes);
    // lseek64(this->fd, offset, SEEK_SET);
    // pread64(this->fd, data_ptr_, bytes, offset);

    release(offset, bytes);
}

void Offloader::sync_write_events()
{
    this->aio->sync_write_events();
}

void Offloader::sync_read_events()
{
    this->aio->sync_read_events();
}

void Offloader::synchronize()
{
    this->aio->synchronize();
}

Offloader::~Offloader()
{
    errno = 0;
    delete this->aio;
    close(this->fd);
    if (remove(this->filename.c_str()) != 0)
        printf("Remove \"%s\" error(%d): %s\n", this->filename.c_str(), errno, strerror(errno));
}

SpaceInfo Offloader::prepare_writev(const std::vector<ull> &nbytes_list, const std::string &key)
{
    ull total_bytes = 0;
    for (const ull &nbytes : nbytes_list)
    {
        // if (!tensor.is_contiguous() || !tensor.is_cpu())
        //     throw std::runtime_error("Tensor must be contiguous and on cpu");
        total_bytes += nbytes;
    }
    ull offset = this->space_mgr.alloc(total_bytes);
    SpaceInfo space_info(offset, total_bytes);
    this->tensors_info[key] = space_info;
    return space_info;
}

SpaceInfo Offloader::prepare_readv(const std::vector<ull> &nbytes_list, const std::string &key)
{
    ull total_bytes = 0;
    for (const ull &nbytes : nbytes_list)
    {
        // if (!tensor.is_contiguous() || !tensor.is_cpu())
        //     throw std::runtime_error("Tensor must be contiguous and on cpu");
        total_bytes += nbytes;
    }
    if (this->tensors_info.find(key) == this->tensors_info.end())
        throw std::runtime_error("Read error, tensor not found");
    SpaceInfo space_info = this->tensors_info[key];
    if (total_bytes != space_info.second)
        throw std::runtime_error("Read error, tensor shape mismatch");
    this->tensors_info.erase(key);
    return space_info;
}

void Offloader::async_writev(const std::vector<ull> &data_ptr_list, const std::vector<ull> &nbytes_list, const std::string &key, callback_t callback)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_writev(nbytes_list, key);
    iovec *iov = tensors_to_iovec(data_ptr_list, nbytes_list);
    this->aio->writev(this->fd, iov, data_ptr_list.size(), offset, callback);

    this->aio->get_event(NOWAIT);
}

void Offloader::async_readv(const std::vector<ull> &data_ptr_list, const std::vector<ull> &nbytes_list, const std::string &key, callback_t callback)
{

    ull offset, bytes;
    std::tie(offset, bytes) = prepare_readv(nbytes_list, key);
    iovec *iov = tensors_to_iovec(data_ptr_list, nbytes_list);
    auto fn = std::bind(&Offloader::release, this, offset, bytes, callback);
    this->aio->readv(this->fd, iov, data_ptr_list.size(), offset, fn);

    this->aio->get_event(NOWAIT);
}

void Offloader::sync_writev(const std::vector<ull> &data_ptr_list, const std::vector<ull> &nbytes_list, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_writev(nbytes_list, key);
    iovec *iov = tensors_to_iovec(data_ptr_list, nbytes_list);
    lseek(this->fd, offset, SEEK_SET);
    writev(this->fd, iov, data_ptr_list.size());
    // lseek64(this->fd, offset, SEEK_SET);
    // pwritev64(this->fd, iov, data_ptr_list.size(), offset);

    delete iov;
}

void Offloader::sync_readv(const std::vector<ull> &data_ptr_list, const std::vector<ull> &nbytes_list, const std::string &key)
{
    ull offset, bytes;
    std::tie(offset, bytes) = prepare_readv(nbytes_list, key);
    iovec *iov = tensors_to_iovec(data_ptr_list, nbytes_list);
    lseek(this->fd, offset, SEEK_SET);
    readv(this->fd, iov, data_ptr_list.size());
    // lseek64(this->fd, offset, SEEK_SET);
    // preadv64(this->fd, iov, data_ptr_list.size(), offset);

    delete iov;
}

void Offloader::release(ull offset, ull bytes, callback_t callback)
{
    this->space_mgr.free(offset, bytes);
    if (callback != nullptr)
        callback();
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<Offloader>(m, "Offloader")
        .def(py::init<const std::string &, unsigned int, const std::string &>(), py::arg("filename"), py::arg("n_entries"), py::arg("backend") = "uring")
        .def("async_write", &Offloader::async_write, py::arg("data_ptr"), py::arg("nbytes"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_read", &Offloader::async_read, py::arg("data_ptr"), py::arg("nbytes"), py::arg("key"), py::arg("callback") = py::none())
        .def("sync_write", &Offloader::sync_write, py::arg("data_ptr"), py::arg("nbytes"), py::arg("key"))
        .def("sync_read", &Offloader::sync_read, py::arg("data_ptr"), py::arg("nbytes"), py::arg("key"))
        .def("sync_write_events", &Offloader::sync_write_events)
        .def("sync_read_events", &Offloader::sync_write_events)
        .def("synchronize", &Offloader::synchronize)
        .def("async_writev", &Offloader::async_writev, py::arg("data_ptr_list"), py::arg("nbytes_list"), py::arg("key"), py::arg("callback") = py::none())
        .def("async_readv", &Offloader::async_readv, py::arg("data_ptr_list"), py::arg("nbytes_list"), py::arg("key"), py::arg("callback") = py::none())
        .def("sync_writev", &Offloader::sync_writev, py::arg("data_ptr_list"), py::arg("nbytes_list"), py::arg("key"))
        .def("sync_readv", &Offloader::sync_readv, py::arg("data_ptr_list"), py::arg("nbytes_list"), py::arg("key"));
    m.def("get_backends", get_backends);
    m.def("probe_backend", probe_backend, py::arg("backend"));
}