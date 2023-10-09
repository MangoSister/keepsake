#pragma once

#include "error.h"

#include <cmath>

namespace ksc
{

// __CUDACC__ defines whether nvcc is steering compilation or not
// __CUDA_ARCH__is always undefined when compiling host code, steered by nvcc or not
// __CUDA_ARCH__is only defined for the device code trajectory of compilation steered by nvcc
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#if defined(__CUDA_ARCH__)
#define CUDA_IS_GPU_CODE
#endif
#else
#define CUDA_HOST_DEVICE
#endif

#if defined(CUDA_IS_GPU_CODE)
#define CONSTEXPR_VAL __device__ constexpr
#else
#define CONSTEXPR_VAL constexpr
#endif

template <class To, class From>
constexpr To CUDA_HOST_DEVICE bit_cast(const From &from) noexcept;

template <class To, class From>
constexpr To CUDA_HOST_DEVICE bit_cast(const From &src) noexcept
{
    static_assert(sizeof(To) == sizeof(From), "sizes must match");
    return reinterpret_cast<To const &>(src);
}

template <class T>
struct numeric_limits;

template <>
struct numeric_limits<int32_t>
{
    CUDA_HOST_DEVICE
    static constexpr int32_t lowest() noexcept { return -2147483647 - 1; }
    CUDA_HOST_DEVICE
    static constexpr int32_t max() noexcept { return 2147483647; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<int16_t>
{
    CUDA_HOST_DEVICE
    static constexpr int16_t lowest() noexcept { return -32768; }
    CUDA_HOST_DEVICE
    static constexpr int16_t max() noexcept { return 32767; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<int8_t>
{
    CUDA_HOST_DEVICE
    static constexpr int8_t lowest() noexcept { return -128; }
    CUDA_HOST_DEVICE
    static constexpr int8_t max() noexcept { return 127; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<uint32_t>
{
    CUDA_HOST_DEVICE
    static constexpr uint32_t lowest() noexcept { return 0; }
    CUDA_HOST_DEVICE
    static constexpr uint32_t max() noexcept { return 4294967295U; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<uint16_t>
{
    CUDA_HOST_DEVICE
    static constexpr uint16_t lowest() noexcept { return 0; }
    CUDA_HOST_DEVICE
    static constexpr uint16_t max() noexcept { return 65535U; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<uint8_t>
{
    CUDA_HOST_DEVICE
    static constexpr uint8_t lowest() noexcept { return 0; }
    CUDA_HOST_DEVICE
    static constexpr uint8_t max() noexcept { return 255U; }
    static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<float>
{
    CUDA_HOST_DEVICE
    static constexpr float infinity() noexcept { return HUGE_VALF; }
    CUDA_HOST_DEVICE
    static constexpr float max() noexcept { return FLT_MAX; }
    static constexpr bool is_integer = false;
    static constexpr bool has_infinity = true;
};

template <typename T>
CUDA_HOST_DEVICE inline void swap(T &a, T &b)
{
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

} // namespace ksc