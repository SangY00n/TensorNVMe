import mmap

import numpy as np
import time
from timeit import timeit


from memory_profiler import profile

# %timeit  np.empty((80,80,300000),dtype='float32')
# %timeit  np.zeros((80,80,300000),dtype='float32')
# %timeit  np.ones((80,80,300000),dtype='float32')
# print(timeit("x = np.empty(1024*1024*1024, dtype='float32')", number=1000, globals=globals()))
# print(timeit("x = np.zeros(1024*1024*1024, dtype='float32')", number=1000, globals=globals()))
# print(timeit("x = np.ones(1024*1024*1024, dtype='float32')", number=1000, globals=globals()))

array_size = 1024*1024*1024

@profile
def test_pagesize():
    # start_time = time.time()
    x = np.empty(array_size, dtype='float32')
    # x = np.arange(array_size, dtype='float32')
    print(x.__array_interface__['data'][0])
    print(x.data.contiguous)
    # print("time for np.empty: ", time.time() - start_time, " seconds")
    # start_time = time.time()
    
    # np.add(x, 0, out=x)
    # x[0]=1
    print(x[0:10])
    print(x.__array_interface__['data'][0])
    print(x.data.contiguous)
    # print(x[::mmap.PAGESIZE//4][-1])
    x[::mmap.PAGESIZE//4]=1
    x[::1] = 1
    # x[-10:] = 2
    print(x[-1])
    x[-1]=1
    print(x.__array_interface__['data'][0])
    print(x.data.contiguous)
    # print("time for memory allocation: ", time.time() - start_time, " seconds")
    # print(mmap.PAGESIZE) # 4096

@profile
def test_resize():
    # start_time = time.time()
    x = np.empty(0, dtype='float32')
    # print("time for np.empty: ", time.time() - start_time, " seconds")
    # start_time = time.time()
    
    x.resize(array_size, refcheck=False)
    # print("time for memory resizing: ", time.time() - start_time, " seconds")

def main():
    # test_resize()
    test_pagesize()
    # time_for_resize = timeit("test_resize()", number=1, globals=globals())
    # time_for_pagesize = timeit("test_pagesize()", number=1, globals=globals())
    
    # print("time for pagesize: ", time_for_pagesize, " seconds")
    # print("time for resize: ", time_for_resize, " seconds")


if __name__ == '__main__':
    main()