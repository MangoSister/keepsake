#pragma once
#include "basic.cuh"

namespace ksc
{

#if defined(__linux__)
typedef int ShareableHandle;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef void *ShareableHandle;
#endif

struct CudaShareableLowLevelMemory
{
    CUdeviceptr dptr{};
    ShareableHandle shareable_handle{};
    size_t size = 0;
};

CudaShareableLowLevelMemory cuda_alloc_device_low_level(size_t size, int device = 0);
void cuda_free_device_low_level(const CudaShareableLowLevelMemory &m);

} // namespace ksc