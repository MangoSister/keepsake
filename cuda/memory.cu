#include "memory.cuh"

namespace ksc
{

void *cuda_alloc_managed(size_t size)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

void *cuda_alloc_device(size_t size)
{
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }

} // namespace ksc