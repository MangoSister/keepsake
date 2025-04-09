#pragma once

#include "api_error.h"

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>

// __CUDACC__ defines whether nvcc is steering compilation or not
// __CUDA_ARCH__ is always undefined when compiling host code, steered by nvcc or not
// __CUDA_ARCH__ is only defined for the device code trajectory of compilation steered by nvcc
// __CUDACC__ +  __CUDA_ARCH__: use nvcc and device code (.cuh/.cu)
// __CUDACC__ +  !__CUDA_ARCH__: use nvcc and host code (.cuh/.cu)
// !__CUDACC__: use normal cpp compiler and host code (.h/.cpp)
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#if defined(__CUDA_ARCH__)
#define CUDA_IS_DEVICE_CODE
#endif
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CPP_CODE_ONLY
#endif

#if defined(CUDA_IS_DEVICE_CODE)
#define CONSTEXPR_VAL __device__ constexpr
#else
#define CONSTEXPR_VAL constexpr
#endif

#ifdef CUDA_IS_DEVICE_CODE
#define CUDA_ASSERT(EXPR) assert(EXPR)

#ifndef NDEBUG
#define CUDA_ASSERT_FMT(EXPR, FMT, ...)                                                                                 \
    do {                                                                                                               \
        if (!(EXPR)) {                                                                                                 \
            printf(FMT, ##__VA_ARGS__);                                                                                \
        }                                                                                                              \
        assert(EXPR);                                                                                                  \
    } while (false) /* eat semicolon */
#else
#define CUDA_ASSERT_FMT(EXPR, FMT, ...) (void(0))
#endif

#else
#include "../assertion.h"
#define CUDA_ASSERT ASSERT
#define CUDA_ASSERT_FMT ASSERT
#endif

namespace ksc
{

template <class To, class From>
constexpr To CUDA_HOST_DEVICE bit_cast(const From &src) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "sizes must match");
    return reinterpret_cast<To const &>(src);
}

} // namespace ksc