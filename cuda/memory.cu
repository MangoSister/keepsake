#include "memory.cuh"

namespace ksc
{

void *cuda_alloc_managed(size_t size)
{
    void *ptr = nullptr;
    cuda_check(cudaMallocManaged(&ptr, size));
    return ptr;
}

void *cuda_alloc_device(size_t size)
{
    void *ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void *ptr) { cuda_check(cudaFree(ptr)); }

} // namespace ksc