#include <stdexcept>
#include "space_mgr.h"
#include <stdio.h>
#include <cstdlib>

SpaceManager::SpaceManager(unsigned long long limit) : limit(limit), used_bytes(0)
{
}

SpaceManager::~SpaceManager()
{
}

ull SpaceManager::alloc(ull bytes)
{
    if (bytes == 0)
        throw std::runtime_error("Invalid alloc size (0)");
    auto target_iter = avail_spaces.end();
    ull target_bytes = 0;
    for (auto iter = avail_spaces.begin(); iter != avail_spaces.end(); iter++)
    {
        if (iter->second >= bytes && (target_iter == avail_spaces.end() || iter->second < target_bytes))
        {
            target_iter = iter;
            target_bytes = iter->second;
        }
    }
    // no available space, use new space
    if (target_iter == avail_spaces.end())
    {
        // limit=0 means unlimit
        if (limit > 0 && limit - used_bytes < bytes)
            throw std::runtime_error("File size exceed limit");
        ull offset = used_bytes;
        used_bytes += bytes;
        return offset;
    }
    ull offset = target_iter->first;
    target_iter->first += bytes;
    target_iter->second -= bytes;
    if (target_iter->second == 0)
        avail_spaces.erase(target_iter);
    return offset;
}

void SpaceManager::free(ull offset, ull bytes)
{
    if (bytes == 0)
        throw std::runtime_error("Invalid free size (0)");
    SpaceInfo new_avail_space(offset, bytes);
    for (auto iter = avail_spaces.begin(); iter != avail_spaces.end();)
    {
        if (offset > iter->first && offset - iter->first == iter->second)
        {
            new_avail_space.first = iter->first;
            new_avail_space.second += iter->second;
            iter = avail_spaces.erase(iter);
        }
        else if (offset < iter->first && iter->first - offset == bytes)
        {
            new_avail_space.second += iter->second;
            iter = avail_spaces.erase(iter);
        }
        else
        {
            iter++;
        }
    }
    // 이 코드 사용 시 발생하는 문제: 모든 tensor가 free되고 마지막 파일의 맨 뒷부분에 write되어 있는 tensor에 대해 free할 때,
    // 이 코드로 들어가면서 이전에 free된 tensor들의 공간이 다시 사용되지 못하게 된다.
    // 이 코드를 제대로 활용하려면, if(new_avail_space.first + new_avail_space.second == used_bytes)인 경우에 used_bytes=used_bytes - new_avail_space.second 해주면 될 것 같다.
    // 그런데 이렇게 한다고 해서 더 효율적으로 동작하지는 않을 것 같음.
    // if (offset + bytes == used_bytes) // 파일의 맨 뒷부분 bytes 공간인 경우.. 인데 이걸 왜 해주지? 이것 때문에 오히려 낭비가 일어나고 있는데. 이걸로 뭘 막을 수 있지?
    //     used_bytes = used_bytes - bytes;
    // else
    //     avail_spaces.push_back(new_avail_space);
    avail_spaces.push_back(new_avail_space);
}

void SpaceManager::print()
{
    printf("Used bytes: %lld", used_bytes);
    for (auto iter = avail_spaces.begin(); iter != avail_spaces.end(); iter++)
    {
        printf(", [%lld, %lld)", iter->first, iter->second);
    }
    printf("\n");
}



MemoryPoolManager::MemoryPoolManager(ull limit) : limit(limit), used_bytes(0)
{
    this->memory_pool = malloc(limit);
}

MemoryPoolManager::~MemoryPoolManager()
{
    free(this->memory_pool);
}

void *MemoryPoolManager::memorypool_alloc(ull bytes)
{
    if (bytes == 0)
        throw std::runtime_error("Invalid alloc size (0)");
    auto target_iter = avail_mem_pools.end();
    ull target_bytes = 0;
    for (auto iter = avail_mem_pools.begin(); iter != avail_mem_pools.end(); iter++)
    {
        if (iter->second >= bytes && (target_iter == avail_mem_pools.end() || iter->second < target_bytes))
        {
            target_iter = iter;
            target_bytes = iter->second;
        }
    }
    // no available space, use new space
    if (target_iter == avail_mem_pools.end())
    {
        if (limit - used_bytes < bytes)
            throw std::runtime_error("Memory pool size exceed limit");
        void *ptr = this->memory_pool + used_bytes;
        used_bytes += bytes;
        used_mem_pools.push_back(MemoryPoolInfo(ptr, bytes));
        return ptr;
    }
    void *ptr = target_iter->first;
    target_iter->first = (void *)((char *)target_iter->first + bytes);
    target_iter->second -= bytes;
    if (target_iter->second == 0)
        avail_mem_pools.erase(target_iter);
    used_mem_pools.push_back(MemoryPoolInfo(ptr, bytes));
    return ptr;
}

void MemoryPoolManager::memorypool_free(void *ptr, ull bytes)
{
    if (bytes == 0)
        throw std::runtime_error("Invalid free size (0)");
    MemoryPoolInfo new_avail_mem_pool(ptr, bytes);
    for (auto iter = avail_mem_pools.begin(); iter != avail_mem_pools.end();)
    {
        if (ptr > iter->first && (char *)ptr - (char *)iter->first == iter->second)
        {
            new_avail_mem_pool.first = iter->first;
            new_avail_mem_pool.second += iter->second;
            iter = avail_mem_pools.erase(iter);
        }
        else if (ptr < iter->first && (char *)iter->first - (char *)ptr == bytes)
        {
            new_avail_mem_pool.second += iter->second;
            iter = avail_mem_pools.erase(iter);
        }
        else
        {
            iter++;
        }
    }
    avail_mem_pools.push_back(new_avail_mem_pool);
    
    for (auto iter = used_mem_pools.begin(); iter != used_mem_pools.end();)
    {
        if (iter->first == ptr)
        {
            used_mem_pools.erase(iter);
            break;
        }
        iter++;
    }
}

void MemoryPoolManager::print()
{
    printf("Used MemoryPool bytes: %lld", used_bytes);
    for (auto iter = avail_mem_pools.begin(); iter != avail_mem_pools.end(); iter++)
    {
        printf(", [%p, %lld)", iter->first, iter->second);
    }
    printf("\n");
}