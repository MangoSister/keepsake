#pragma once

#include <cassert>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ksc
{

#define KSC_ASSERT(EXPR)                                                                                               \
    do {                                                                                                               \
        assert(EXPR);                                                                                                  \
    } while (false) /* eat semicolon */

#define KSC_ASSERT_FMT(EXPR, FMT, ...)                                                                                 \
    do {                                                                                                               \
        if (!(EXPR)) {                                                                                                   \
            printf(FMT, __VA_ARGS__);                                                                                  \
        }                                                                                                              \
        assert(EXPR);                                                                                                  \
    } while (false) /* eat semicolon */

#define CUDA_CHECK(EXPR)                                                                                               \
    if (EXPR != cudaSuccess) {                                                                                         \
        cudaError_t error = cudaGetLastError();                                                                        \
        KSC_ASSERT_FMT(false, "CUDA error: %s\n", cudaGetErrorString(error));                                            \
        std::abort();                                                                                                  \
    } else /* eat semicolon */

#define CU_CHECK(EXPR)                                                                                                 \
    do {                                                                                                               \
        CUresult result = EXPR;                                                                                        \
        if (result != CUDA_SUCCESS) {                                                                                  \
            const char *str;                                                                                           \
            CUresult get_error_str_result = cuGetErrorString(result, &str);                                            \
            KSC_ASSERT(CUDA_SUCCESS == get_error_str_result);                                                          \
            KSC_ASSERT_FMT(false, "CUDA error: %s\n", str);                                                              \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (false) /* eat semicolon */

} // namespace ksc