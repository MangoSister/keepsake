#pragma once
#include "vecmath.cuh"

// Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering, Journal of Computer Graphics Techniques (JCGT), vol.
// 9, no. 3, 21-38, 2020 https://www.shadertoy.com/view/XlGcRh

namespace ksc
{

// Parameters from Numerical Recipes.
CUDA_HOST_DEVICE inline uint32_t lcg(uint32_t p) { return p * 1664525u + 1013904223u; }

// Combine several hash values for hash tables. Based on the algorithm used in Boost.
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t &seed) {}

// TODO: need a device version for std::hash<T>...
template <typename T, typename... Rest>
inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

CUDA_HOST_DEVICE
inline uint32_t xxhash32(uint32_t p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

CUDA_HOST_DEVICE
inline uint32_t xxhash32(vec2u p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

CUDA_HOST_DEVICE
inline uint32_t xxhash32(vec3u p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

CUDA_HOST_DEVICE
inline uint32_t xxhash32(vec4u p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

} // namespace ksc