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
