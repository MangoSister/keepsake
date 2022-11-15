#pragma once

#include "maths.h"
#include <functional>

// https://www.shadertoy.com/view/XlGcRh

template <int N>
inline arr<N> convert_u32_f01(const arru<N> &u32)
{
    return u32.template cast<float>() / float(0xffffffffu);
}

inline uint32_t hash11u(uint32_t p)
{
    // xxhash32
    const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline arr2u hash12u(uint32_t p)
{
    arr2u v;
    v.x() = hash11u(p);
    // some big primes
    v.y() = hash11u(p + 3794833813);
    return v;
}

inline arr2u hash22u(arr2u v)
{
    // pcg2d
    v = v * 1664525u + 1013904223u;

    v.x() += v.y() * 1664525u;
    v.y() += v.x() * 1664525u;

    // v = v ^ (v >> 16u);
    v.x() = v.x() ^ (v.x() >> 16u);
    v.y() = v.y() ^ (v.y() >> 16u);

    v.x() += v.y() * 1664525u;
    v.y() += v.x() * 1664525u;

    // v = v ^ (v >> 16u);
    v.x() = v.x() ^ (v.x() >> 16u);
    v.y() = v.y() ^ (v.y() >> 16u);

    return v;
}

inline arr2 hash22f(arr2u v) { return convert_u32_f01<2>(hash22u(v)); }

inline arr3u hash13u(uint32_t p)
{
    arr3u v;
    v.x() = hash11u(p);
    // some big primes
    v.y() = hash11u(p + 3794833813);
    v.z() = hash11u(p + 4200861443);
    return v;
}

inline arr3u hash33u(arr3u v)
{
    // pcg3d
    v = v * 1664525u + 1013904223u;

    v.x() += v.y() * v.z();
    v.y() += v.z() * v.x();
    v.z() += v.x() * v.y();

    // v ^= v >> 16u;
    v.x() ^= v.x() >> 16u;
    v.y() ^= v.y() >> 16u;
    v.z() ^= v.z() >> 16u;

    v.x() += v.y() * v.z();
    v.y() += v.z() * v.x();
    v.z() += v.x() * v.y();

    return v;
}

inline arr3 hash33f(arr3u v) { return convert_u32_f01<3>(hash33u(v)); }

// One-liner linear congruential generator. Quick but low quality.
constexpr uint32_t lcg(uint32_t p) { return p * 1664525u + 1013904223u; }

// Combine several hash values for hash tables. Based on the algorithm used in Boost.
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t &seed) {}

template <typename T, typename... Rest>
inline void hash_combine(std::size_t &seed, const T &v, Rest... rest)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}