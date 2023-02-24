#pragma once
#include <cstdint>
#include <vector>
#include <memory>

namespace ks
{

inline void *alloc_aligned(size_t size, size_t alignment)
{
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__OSX__)
    /* OSX malloc already returns 16-byte aligned data suitable
       for AltiVec and SSE computations */
    return malloc(size);
#else
    return aligned_alloc(alignment, size);
#endif
}

template <typename T>
T *alloc_aligned(size_t count, size_t alignment)
{
    return (T *)alloc_aligned(count * sizeof(T), alignment);
}

inline void free_aligned(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Variable length array on stack. Be very careful about this!!
#define VLA(PTR, TYPE, COUNT)                                                                                          \
    static_assert(std::is_trivially_destructible<TYPE>::value, "Don't use VLA for non trivially destructible types."); \
    TYPE *PTR = (TYPE *)alloca((COUNT) * sizeof(TYPE));                                                                \
    {                                                                                                                  \
        for (uint32_t __i = 0; __i < COUNT; ++__i)                                                                     \
            new (PTR + __i) TYPE();                                                                                    \
    }

class Allocator
{
  public:
    virtual ~Allocator() = default;

    virtual void *allocate(size_t byte_count) = 0;
    virtual void free(void *bytes) = 0;

    template <typename T, typename... Args>
    T *allocate_typed(Args &&...args)
    {
        void *bytes = allocate(sizeof(T));
        T *obj = reinterpret_cast<T *>(bytes);
        return std::construct_at(obj, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    T *allocate_array(size_t count, Args &&...args)
    {
        void *bytes = allocate(sizeof(T) * count);
        T *objs = reinterpret_cast<T *>(bytes);
        for (size_t i = 0; i < count; ++i) {
            std::construct_at(&objs[i], std::forward<Args>(args)...);
        }
        return objs;
    }

    template <typename T>
    void free_typed(T *obj)
    {
        std::destroy_at(obj);
        free(obj);
    }

    template <typename T>
    void free_array(T *objs, size_t count)
    {
        std::destroy_n(objs, count);
        free(objs);
    }
};

class BlockAllocator final : public Allocator
{
  public:
    explicit BlockAllocator(size_t default_block_size = 1024, size_t max_num_blocks = 1024);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator &other) = delete;
    BlockAllocator &operator=(const BlockAllocator &other) = delete;

    BlockAllocator(BlockAllocator &&other);
    BlockAllocator &operator=(BlockAllocator &&other);
    void swap(BlockAllocator &other);

    void *allocate(size_t byte_count) final;
    void free(void *bytes) final;

    void reset();

  private:
    using byteptr_t = uint8_t *;

    size_t default_block_size = 0;
    size_t curr_block_pos = 0;
    size_t curr_alloc_size = 0;
    byteptr_t curr_block = nullptr;
    std::vector<std::pair<size_t, byteptr_t>> used_blocks;
    std::vector<std::pair<size_t, byteptr_t>> free_blocks;
    size_t max_num_blocks;
};

} // namespace ks