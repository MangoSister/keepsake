#pragma once

#include "maths.h"

// https://www.shadertoy.com/view/XlGcRh

template <int N>
inline arr<N> convert_u32_f01(arru<N> u32)
{
    return u32.cast<float>() / float(0xffffffffu);
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

inline uint32_t xxhash32(uint32_t p)
{
    const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(uint32_t x, uint32_t y)
{
    const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = y + PRIME32_5 + x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}