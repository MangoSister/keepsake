#pragma once

#include "assertion.h"
#include "hash.h"
#include "maths.h"
#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <span>

namespace ks
{

constexpr uint64_t PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL;
constexpr uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
constexpr uint64_t PCG32_MULT = 0x5851f42d4c957f2dULL;

struct RNG
{
    RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}
    RNG(uint64_t seq_index, uint64_t offset) { set_seq(seq_index, offset); }
    explicit RNG(uint64_t seq_index) { set_seq(seq_index); }

    void set_seq(uint64_t sequence_index, uint64_t offset)
    {
        state = 0u;
        inc = (sequence_index << 1u) | 1u;
        next_u32();
        state += offset;
        next_u32();
    }

    void set_seq(uint64_t sequence_index) { set_seq(sequence_index, mix_bits(sequence_index)); }

    uint32_t next_u32()
    {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    uint64_t next_u64()
    {
        uint64_t v0 = next_u32(), v1 = next_u32();
        return (v0 << 32) | v1;
    }

    int32_t next_i32()
    {
        // https://stackoverflow.com/a/13208789
        uint32_t v = next_u32();
        if (v <= (uint32_t)std::numeric_limits<int32_t>::max())
            return int32_t(v);
        ASSERT(v >= (uint32_t)std::numeric_limits<int32_t>::min());
        return int32_t(v - std::numeric_limits<int32_t>::min()) + std::numeric_limits<int32_t>::min();
    }

    int64_t next_i64()
    {
        // https://stackoverflow.com/a/13208789
        uint64_t v = next_u64();
        if (v <= (uint64_t)std::numeric_limits<int64_t>::max())
            // Safe to type convert directly.
            return int64_t(v);
        ASSERT(v >= (uint64_t)std::numeric_limits<int64_t>::min());
        return int64_t(v - std::numeric_limits<int64_t>::min()) + std::numeric_limits<int64_t>::min();
    }

    float next() { return std::min<float>(fp32_before_one, next_u32() * 0x1p-32f); }

    float next_f32() { return next(); }

    double next_f64() { return std::min<double>(fp32_before_one, next_u64() * 0x1p-64); }

    template <int N>
    Eigen::Matrix<float, N, 1> next()
    {
        Eigen::Matrix<float, N, 1> u;
        for (int i = 0; i < N; ++i) {
            u[i] = next();
        }
        return u;
    }

    vec2 next2d() { return next<2>(); }
    vec3 next3d() { return next<3>(); }
    vec4 next4d() { return next<4>(); }

    void advance(int64_t idelta)
    {
        uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u;
        uint64_t acc_plus = 0u, delta = (uint64_t)idelta;
        while (delta > 0) {
            if (delta & 1) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        state = acc_mult * state + acc_plus;
    }

    int64_t operator-(const RNG &other) const
    {
        ASSERT(inc == other.inc);
        uint64_t cur_mult = PCG32_MULT, cur_plus = inc, cur_state = other.state;
        uint64_t the_bit = 1u, distance = 0u;
        while (state != cur_state) {
            if ((state & the_bit) != (cur_state & the_bit)) {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }
            ASSERT(state & the_bit == cur_state & the_bit);
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }
        return (int64_t)distance;
    }

    uint64_t state, inc;
};

template <int base>
constexpr float radical_inverse(uint32_t a)
{
    constexpr float inv_base = 1.0f / (float)base;
    uint32_t reversed_digits = 0;
    float inv_base_n = 1;
    while (a) {
        uint32_t next = a / base;
        uint32_t digit = a - next * base;
        reversed_digits = reversed_digits * base + digit;
        inv_base_n *= inv_base;
        a = next;
    }
    // ASSERT(reversed_digits * inv_base_n < 1.00001);
    return std::min(reversed_digits * inv_base_n, fp32_before_one);
}

template <>
constexpr float radical_inverse<2>(uint32_t bits)
{
    return float(reverse_bits_32(bits)) * 2.3283064365386963e-10; // / 0x100000000
}

inline vec2 hammersley_2d(uint32_t i, uint32_t N) { return vec2(float(i) / float(N), radical_inverse<2>(i)); }
inline vec3 hammersley_3d(uint32_t i, uint32_t N)
{
    return vec3(float(i) / float(N), radical_inverse<2>(i), radical_inverse<3>(i));
}

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://www.shadertoy.com/view/4dtBWH
inline vec2 roberts_qmc_2d(uint32_t n, const vec2 &p0 = vec2::Zero())
{
    vec2 p;
    p.x() = fract(p0.x() + (float)(n * 12664745) / (float)(1 << 24));
    p.y() = fract(p0.y() + (float)(n * 9560333) / (float)(1 << 24));
    return p;
}

inline vec3 roberts_qmc_3d(uint32_t n, const vec3 &p0 = vec3::Zero())
{
    vec3 p;
    p.x() = fract(p0.x() + (float)(n * 13743434) / (float)(1 << 24));
    p.y() = fract(p0.y() + (float)(n * 11258243) / (float)(1 << 24));
    p.z() = fract(p0.z() + (float)(n * 9222443) / (float)(1 << 24));
    return p;
}

inline vec3 spherical_fibonacci(int i, int N)
{
    constexpr float golden_ratio = 1.61803398875f;
    float phi = two_pi * ((float)i / golden_ratio - std::floor((float)i / golden_ratio));
    float z = 1.0f - (float)(2 * i + 1) / (float)N;
    return to_cartesian(phi, std::acos(z));
}

inline vec3 sample_uniform_hemisphere(const vec2 &u)
{
    float z = u.y();
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = two_pi * u.x();
    float sin_phi = std::sin(phi);
    float cos_phi = std::cos(phi);

    return vec3(r * cos_phi, r * sin_phi, z);
}

inline vec2 sample_disk(const vec2 &u)
{
    float a = 2.0f * u[0] - 1.0f;
    float b = 2.0f * u[1] - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        return vec2::Zero();
    }
    float r, phi;
    if (a * a > b * b) {
        r = a;
        phi = 0.25f * pi * (b / a);
    } else {
        r = b;
        phi = half_pi - 0.25f * pi * (a / b);
    }
    return r * vec2(std::cos(phi), std::sin(phi));
}

inline vec3 sample_cosine_hemisphere(const vec2 &u)
{
    vec2 d = sample_disk(u);
    float z = std::sqrt(std::max(0.0f, 1.0f - d.squaredNorm()));
    return vec3(d.x(), d.y(), z);
    // pdf = z / pi;
}

inline vec3 sample_cosine_hemisphere(const vec2 &u, const vec3 &n)
{
    float a = 1.0f - 2.0f * u[0];
    float b = safe_sqrt(1.0f - a * a);
    // Avoid zero vector output in case the sample is in the opposite direction of n.
    a *= 0.999f;
    b *= 0.999f;

    float phi = two_pi * u[1];
    float x = n.x() + b * cos(phi);
    float y = n.y() + b * sin(phi);
    float z = n.z() + a;
    return vec3(x, y, z).normalized();
}

inline vec3 sample_uniform_sphere(const vec2 &u)
{
    float z = 1.0f - 2.0f * u.y();
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = two_pi * u.x();
    float sin_phi = std::sin(phi);
    float cos_phi = std::cos(phi);
    return vec3(r * cos_phi, r * sin_phi, z);
}

inline vec3 sample_uniform_sphere_vol(const vec3 &u)
{
    float phi = u.x() * two_pi;
    float cos_theta = u.y() * 2.0f - 1.0f;
    float sin_theta = std::sqrt(std::max(0.0f, 1.0f - sqr(cos_theta)));
    float r = std::cbrt(u.z()); // cube root
    return r * vec3(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
}

// Gammell, Jonathan D., and Timothy D. Barfoot. "The probability density function of a transformation-based
// hyperellipsoid sampling technique." arXiv preprint arXiv:1404.1347 (2014).
inline vec3 sample_uniform_ellipsoid(const mat3 &L, vec3 center, vec2 u, vec3 *sph = nullptr)
{
    vec3 p = sample_uniform_sphere(u);
    if (sph) {
        *sph = p;
    }
    return L * p + center;
}

// inline float sample_normal(RNG &rng, float mean, float stddev)
//{
//     std::normal_distribution<> normal_dist(mean, stddev);
//     return normal_dist(rng);
// }
//
// template <int N>
// inline std::array<float, N> sample_normal(RNG &rng)
//{
//     std::normal_distribution<> normal_dist(0.0f, 1.0f);
//     std::array<float, N> samples;
//     for (int i = 0; i < N; ++i) {
//         samples[i] = normal_dist(rng);
//     }
//     return samples;
// }

inline bool russian_roulette(color3 &beta, float u)
{
    if (beta.maxCoeff() < 0.05f) {
        float q = std::max(0.05f, 1.0f - beta.maxCoeff());
        if (u < q) {
            return false;
        }
        beta /= 1.0f - q;
    }
    return true;
}

// Better mapping from pbrt4 that both avoids sqrt and folding
inline vec3 sample_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec2 &u, vec2 *bary)
{
    float b0, b1;
    if (u[0] < u[1]) {
        b0 = u[0] / 2;
        b1 = u[1] - b0;
    } else {
        b1 = u[1] / 2;
        b0 = u[0] - b1;
    }
    float b2 = 1.0f - b0 - b1;
    if (bary) {
        (*bary)[0] = b0;
        (*bary)[1] = b1;
    }
    return b0 * v0 + b1 * v1 + b2 * v2;
}

inline vec2 sample_triangle_bary(const vec2 &u)
{
    float b0, b1;
    if (u[0] < u[1]) {
        b0 = u[0] / 2;
        b1 = u[1] - b0;
    } else {
        b1 = u[1] / 2;
        b0 = u[0] - b1;
    }
    return vec2(b0, b1);
}

// Sample a tetrahedron: http://vcg.isti.cnr.it/jgt/tetra.htm
inline vec3 sample_tetrahedron(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v3, const vec3 &xi)
{
    float s = xi[0];
    float t = xi[1];
    float u = xi[2];

    if (s + t > 1.0f) { // cut'n fold the cube into a prism
        s = 1.0f - s;
        t = 1.0f - t;
    }
    if (t + u > 1.0f) { // cut'n fold the prism into a tetrahedron
        float tmp = u;
        u = 1.0f - s - t;
        t = 1.0f - tmp;
    } else if (s + t + u > 1.0f) {
        float tmp = u;
        u = s + t + u - 1.0f;
        s = 1.0f - t - tmp;
    }
    float a = 1.0f - s - t - u; // a,s,t,u are the barycentric coordinates of the random point.
    return v0 * a + v1 * s + v2 * t + v3 * u;
}

inline int sample_small_distrib(std::span<const float> data, float u, float *u_remap = nullptr)
{
    int N = (int)data.size();
    float sum_w = 0.0f;
    int last_positive = 0;
    for (int i = 0; i < N; ++i) {
        sum_w += data[i];
        if (data[i] > 0.0f) {
            last_positive = i;
        }
    }
    ASSERT(sum_w > 0.0f);
    float inv_sum_w = 1.0f / sum_w;

    float cdf = 0.0f;
    int selected = -1;
    for (int i = 0; i < N; ++i) {
        float dcdf = data[i] * inv_sum_w;
        float cdf_next = cdf + dcdf;
        if (u < cdf_next) {
            selected = i;
            if (u_remap) {
                *u_remap = (u - cdf) / (cdf_next - cdf);
            }
            break;
        }
        cdf = cdf_next;
    }
    if (selected == -1) {
        selected = last_positive;
    }
    ASSERT(data[selected] > 0.0f);
    return selected;
}

// Usually the data is not a float array, but an array of struct that includes other fields that the pdfs
// (weights).
template <typename T, auto member_ptr>
inline int sample_small_distrib(const std::span<const T> data, float u, float *u_remap = nullptr)
{
    int N = (int)data.size();
    float sum_w = 0.0f;
    int last_positive = 0;
    for (int i = 0; i < N; ++i) {
        sum_w += data[i].*member_ptr;
        if (data[i].*member_ptr > 0.0f) {
            last_positive = i;
        }
    }
    ASSERT(sum_w > 0.0f);
    float inv_sum_w = 1.0f / sum_w;

    float cdf = 0.0f;
    int selected = -1;
    for (int i = 0; i < N; ++i) {
        float dcdf = data[i].*member_ptr * inv_sum_w;
        float cdf_next = cdf + dcdf;
        if (u < cdf_next) {
            selected = i;
            if (u_remap) {
                *u_remap = (u - cdf) / (cdf_next - cdf);
            }
            break;
        }
        cdf = cdf_next;
    }
    if (selected == -1) {
        selected = last_positive;
    }
    ASSERT(data[selected].*member_ptr > 0.0f);
    return selected;
}

inline float sample_standard_exp(float u) { return -std::log(1.0f - u); }

inline float sample_normal(float u1, float u2, float *n2 = nullptr)
{
    float x1 = 2.0f * u1 - 1.0f;
    float x2 = 2.0f * u2 - 1.0f;
    float r2 = x1 * x1 + x2 * x2;
    r2 = std::clamp(r2, 1e-5f, 1.0f - 1e-5f);

    // Polar method, a more efficient version of the Box-Muller approach.
    float f = std::sqrt(-2.0f * std::log(r2) / r2);
    /* Keep for next call */
    if (n2)
        *n2 = f * x2;
    return f * x1;
}

// beta, gamma distribution sampling ported from numpy.
// https://github.com/numpy/numpy/blob/9ee262b5bf89e8c866a507e4d62b78532361adc2/numpy/random/src/legacy/legacy-distributions.c
float sample_standard_gamma(RNG &rng, float shape);

float sample_beta(RNG &rng, float a, float b);

// https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
// h = (sqrt(1 - u) sin(2 \pi v), sqrt(1 - u) cos(2 \pi v), sqrt(u) sin(2 \pi w), sqrt(u) cos(2 \pi w))
inline ks::quat sample_uniform_rotation(float u, float v, float w)
{
    ks::quat q;
    q.w() = std::sqrt(1 - u) * std::sin(2 * ks::pi * v);
    q.x() = std::sqrt(1 - u) * std::cos(2 * ks::pi * v);
    q.y() = std::sqrt(u) * std::sin(2 * ks::pi * w);
    q.z() = std::sqrt(u) * std::cos(2 * ks::pi * w);
    return q;
}

// https://graphics.stanford.edu/~seander/bithacks.html#SelectPosFromMSBRank
// https://yduf.github.io/bit-select-random-set/
// (counting from the left)
inline uint32_t sample_bitmask_64(uint64_t v, float u)
{
    // Input value to find position with rank r.
    int bitcnt = std::popcount(v);
    // unsigned int r;      // Input: bit's desired rank [1-64].
    unsigned int r = std::clamp((int)std::floor(u * bitcnt), 0, bitcnt - 1) + 1;
    unsigned int s;      // Output: Resulting position of bit with rank r [1-64]
    uint64_t a, b, c, d; // Intermediate temporaries for bit count.
    unsigned int t;      // Bit count temporary.

    // Do a normal parallel bit count for a 64-bit integer,
    // but store all intermediate steps.
    // a = (v & 0x5555...) + ((v >> 1) & 0x5555...);
    a = v - ((v >> 1) & ~0UL / 3);
    // b = (a & 0x3333...) + ((a >> 2) & 0x3333...);
    b = (a & ~0UL / 5) + ((a >> 2) & ~0UL / 5);
    // c = (b & 0x0f0f...) + ((b >> 4) & 0x0f0f...);
    c = (b + (b >> 4)) & ~0UL / 0x11;
    // d = (c & 0x00ff...) + ((c >> 8) & 0x00ff...);
    d = (c + (c >> 8)) & ~0UL / 0x101;
    t = (d >> 32) + (d >> 48);
    // Now do branchless select!
    s = 64;
    // if (r > t) {s -= 32; r -= t;}
    s -= ((t - r) & 256) >> 3;
    r -= (t & ((t - r) >> 8));
    t = (d >> (s - 16)) & 0xff;
    // if (r > t) {s -= 16; r -= t;}
    s -= ((t - r) & 256) >> 4;
    r -= (t & ((t - r) >> 8));
    t = (c >> (s - 8)) & 0xf;
    // if (r > t) {s -= 8; r -= t;}
    s -= ((t - r) & 256) >> 5;
    r -= (t & ((t - r) >> 8));
    t = (b >> (s - 4)) & 0x7;
    // if (r > t) {s -= 4; r -= t;}
    s -= ((t - r) & 256) >> 6;
    r -= (t & ((t - r) >> 8));
    t = (a >> (s - 2)) & 0x3;
    // if (r > t) {s -= 2; r -= t;}
    s -= ((t - r) & 256) >> 7;
    r -= (t & ((t - r) >> 8));
    t = (v >> (s - 1)) & 0x1;
    // if (r > t) s--;
    s -= ((t - r) & 256) >> 8;
    s = 65 - s;

    // Modified to return bit pos [0-63].
    return s - 1;
}

} // namespace ks