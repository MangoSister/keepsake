#pragma once

#include "maths.h"
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ks
{

// https://www.shadertoy.com/view/XlGcRh

// One-liner linear congruential generator. Quick but low quality.
constexpr uint32_t lcg(uint32_t p) { return p * 1664525u + 1013904223u; }

template <int N>
inline arr<N> convert_u32_f01(const arru<N> &u32)
{
    return u32.template cast<float>() / float(0xffffffffu);
}

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

inline uint32_t xxhash32(arr2u p)
{
    const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.y() + PRIME32_5 + p.x() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(arr3u p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.z() + PRIME32_5 + p.x() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(arr4u p)
{
    constexpr uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    constexpr uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.w() + PRIME32_5 + p.x() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z() * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline arr2u xxhash32_1to2(uint32_t p)
{
    arr2u v;
    v.x() = xxhash32(p);
    p = lcg(p);
    v.y() = xxhash32(p);
    return v;
}

inline arr3u xxhash32_1to3(uint32_t p)
{
    arr3u v;
    v.x() = xxhash32(p);
    p = lcg(p);
    v.y() = xxhash32(p);
    p = lcg(p);
    v.z() = xxhash32(p);
    return v;
}

inline arr2u pcg2d(arr2u v)
{
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

inline arr2 pcg2d_float(arr2u v) { return convert_u32_f01<2>(pcg2d(v)); }

inline arr3u pcg3d(arr3u v)
{
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

inline arr3 pcg3d_float(arr3u v) { return convert_u32_f01<3>(pcg3d(v)); }

// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
inline uint64_t murmur_hash_64A(const unsigned char *key, size_t len, uint64_t seed)
{
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const unsigned char *end = key + 8 * (len / 8);

    while (key != end) {
        uint64_t k;
        std::memcpy(&k, key, sizeof(uint64_t));
        key += 8;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    switch (len & 7) {
    case 7:
        h ^= uint64_t(key[6]) << 48;
    case 6:
        h ^= uint64_t(key[5]) << 40;
    case 5:
        h ^= uint64_t(key[4]) << 32;
    case 4:
        h ^= uint64_t(key[3]) << 24;
    case 3:
        h ^= uint64_t(key[2]) << 16;
    case 2:
        h ^= uint64_t(key[1]) << 8;
    case 1:
        h ^= uint64_t(key[0]);
        h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

// See latest boost for more information about hash mixing/combining
// https://www.boost.org/doc/libs/1_86_0/boost/intrusive/detail/hash_mix.hpp
inline uint32_t mix_bits_u32(uint32_t x)
{
    constexpr uint32_t m1 = 0x21f0aaad;
    constexpr uint32_t m2 = 0x735a2d97;

    x ^= x >> 16;
    x *= m1;
    x ^= x >> 15;
    x *= m2;
    x ^= x >> 15;

    return x;
}

// (lastest boost now uses a different variant)
// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
inline uint64_t mix_bits(uint64_t v)
{
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

inline void hash_combine(uint64_t &seed, uint64_t value) { seed = mix_bits(seed + 0x9e3779b9 + value); }

inline void hash_combine_u32(uint32_t &seed, uint32_t value) { seed = mix_bits_u32(seed + 0x9e3779b9 + value); }

template <typename T>
inline uint64_t hash_buffer(const T *ptr, size_t size, uint64_t seed = 0)
{
    return murmur_hash_64A((const unsigned char *)ptr, size, seed);
}

template <typename... Args>
inline uint64_t hash(Args... args);

template <typename... Args>
inline void hash_recursive_copy(char *buf, Args...);

template <>
inline void hash_recursive_copy(char *buf)
{}

template <typename T, typename... Args>
inline void hash_recursive_copy(char *buf, T v, Args... args)
{
    std::memcpy(buf, &v, sizeof(T));
    hash_recursive_copy(buf + sizeof(T), args...);
}

template <typename... Args>
inline uint64_t hash(Args... args)
{
    // C++, you never cease to amaze: https://stackoverflow.com/a/57246704
    constexpr size_t sz = (sizeof(Args) + ... + 0);
    constexpr size_t n = (sz + 7) / 8;
    uint64_t buf[n];
    hash_recursive_copy((char *)buf, args...);
    return murmur_hash_64A((const unsigned char *)buf, sz, 0);
}

template <typename... Args>
inline float hash_float(Args... args)
{
    return uint32_t(hash(args...)) * 0x1p-32f;
}

template <typename T>
struct ByteHash
{
    size_t operator()(const T &obj) const { return hash(obj); }
};

template <typename T>
struct ByteEqual
{
    bool operator()(const T &a, const T &b) const { return std::memcmp(&a, &b, sizeof(T)) == 0; }
};

template <typename T>
using ByteOpHashTable = std::unordered_map<T, uint32_t, ByteHash<T>, ByteEqual<T>>;

// heterogeneous lookup: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1690r1.html
// https://www.cppstories.com/2021/heterogeneous-access-cpp20/
struct string_hash
{
    using is_transparent = void;
    size_t operator()(const char *txt) const { return std::hash<std::string_view>{}(txt); }
    size_t operator()(std::string_view txt) const { return std::hash<std::string_view>{}(txt); }
    size_t operator()(const std::string &txt) const { return std::hash<std::string>{}(txt); }
};
template <typename T>
using StringHashTable = std::unordered_map<std::string, T, string_hash, std::equal_to<>>;

} // namespace ks