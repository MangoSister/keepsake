#pragma once

#include "../assertion.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <source_location>

inline void cuda_check(cudaError_t error, const std::source_location location = std::source_location::current())
{
    if (error != cudaSuccess) {
        ASSERT(false, "[File: %s (%u:%u), in `%s`] CUDA Runtime API error: %s\n", location.file_name(), location.line(),
               location.column(), location.function_name(), cudaGetErrorString(error));
        std::abort();
    }
}

inline void cu_check(CUresult result, const std::source_location location = std::source_location::current())
{
    if (result != CUDA_SUCCESS) {
        const char *str;
        CUresult get_error_str_result = cuGetErrorString(result, &str);
        ASSERT(CUDA_SUCCESS == get_error_str_result);
        ASSERT(false, "[File: %s (%u:%u), in `%s`] CUDA Driver API error: %s\n", location.file_name(), location.line(),
               location.column(), location.function_name(), str);
        std::abort();
    }
}

inline void init_cuda(int device = 0)
{
    // Is this necessary??
    cudaFree(nullptr);

    int driver_version;
    cuda_check(cudaDriverGetVersion(&driver_version));
    int runtime_version;
    cuda_check(cudaRuntimeGetVersion(&runtime_version));

    int driver_major = driver_version / 1000;
    int driver_minor = (driver_version - driver_major * 1000) / 10;

    int runtime_major = runtime_version / 1000;
    int runtime_minor = (runtime_version - runtime_major * 1000) / 10;

    printf("GPU CUDA driver %d.%d\n", driver_major, driver_minor);
    printf("CUDA runtime %d.%d\n", runtime_major, runtime_minor);

    int n_devices;
    cuda_check(cudaGetDeviceCount(&n_devices));

    for (int i = 0; i < n_devices; ++i) {
        cudaDeviceProp device_prop;
        cuda_check(cudaGetDeviceProperties(&device_prop, i));
        ASSERT(device_prop.canMapHostMemory);

        printf("CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
               "with shader model %d.%d\n",
               i, device_prop.name, device_prop.totalGlobalMem / (1024. * 1024.), device_prop.multiProcessorCount,
               device_prop.clockRate / 1000., device_prop.major, device_prop.minor);
    }

    cuda_check(cudaSetDevice(device));

    int has_unified_addressing;
    cuda_check(cudaDeviceGetAttribute(&has_unified_addressing, cudaDevAttrUnifiedAddressing, device));
    if (!has_unified_addressing) {
        printf("The selected GPU device (%d) does not support unified addressing.\n", device);
        std::exit(EXIT_FAILURE);
    }

    cuda_check(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    size_t stack_size;
    cuda_check(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
    printf("Reset stack size to %zu\n", stack_size);

    cuda_check(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024));

    cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}
