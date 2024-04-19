#pragma once

#include <vector>
#include <utility>

using std::vector;
using ull = unsigned long long;
// (offset, bytes) pair
using SpaceInfo = std::pair<ull, ull>;

class SpaceManager
{
private:
    ull limit, used_bytes;
    vector<SpaceInfo> avail_spaces;

public:
    SpaceManager(ull limit);
    ~SpaceManager();
    ull alloc(ull bytes);
    void free(ull offset, ull bytes);
    void print();
};

// (ptr, bytes) pair
using MemoryPoolInfo = std::pair<void *, ull>;
class MemoryPoolManager
{
    private:
        ull limit, used_bytes;
        vector<MemoryPoolInfo> avail_mem_pools;
        vector<MemoryPoolInfo> used_mem_pools;

        void *memory_pool;
    public:
        MemoryPoolManager(ull limit);
        ~MemoryPoolManager();
        void *memorypool_alloc(ull bytes);
        void memorypool_free(void *ptr, ull bytes);
        void print();
};