#pragma once

#include "basic.cuh"
#include <cuda/std/limits>
#include <cuda/std/span>

namespace ksc
{

CONSTEXPR_VAL float float_before_one = 0x1.fffffep-1;
CONSTEXPR_VAL float pi = 3.14159265358979323846;
CONSTEXPR_VAL float inf = cuda::std::numeric_limits<float>::infinity();

// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH.html#group__CUDA__MATH
// Note also that due to implementation constraints, certain math functions from std:: namespace may be callable in
// device code even via explicitly qualified std:: names. However, such use is discouraged, since this capability is
// unsupported, unverified, undocumented, not portable, and may change without notice.

template <typename T>
CUDA_HOST_DEVICE inline T abs(T x)
{
#ifdef __CUDACC__
    return ::abs(x);
#else
    return std::abs(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T min(T x, T y)
{
#ifdef __CUDACC__
    return ::min(x, y);
#else
    return std::min(x, y);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T max(T x, T y)
{
#ifdef __CUDACC__
    return ::max(x, y);
#else
    return std::max(x, y);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T floor(T x)
{
#ifdef __CUDACC__
    return ::floor(x);
#else
    return std::floor(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T ceil(T x)
{
#ifdef __CUDACC__
    return ::ceil(x);
#else
    return std::ceil(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T sqrt(T x)
{
#ifdef __CUDACC__
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T log(T x)
{
#ifdef __CUDACC__
    return ::log(x);
#else
    return std::log(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T exp(T x)
{
#ifdef __CUDACC__
    return ::exp(x);
#else
    return std::exp(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline T fma(T x, T y, T z)
{
#ifdef __CUDACC__
    return ::fma(x, y, z);
#else
    return std::fma(x, y, z);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline bool isnan(T x)
{
#ifdef __CUDACC__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline bool isfinite(T x)
{
#ifdef __CUDACC__
    return ::isfinite(x);
#else
    return std::isfinite(x);
#endif
}

template <typename T>
CUDA_HOST_DEVICE inline bool signbit(T x)
{
#ifdef __CUDACC__
    return ::signbit(x);
#else
    return std::signbit(x);
#endif
}

template <typename T, typename U, typename V>
CUDA_HOST_DEVICE inline constexpr T clamp(T val, U low, V high)
{
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
}

template <typename T>
CUDA_HOST_DEVICE inline constexpr T saturate(T val)
{
    return clamp(val, T(0), T(1));
}

CUDA_HOST_DEVICE inline float safe_sqrt(float x) { return sqrtf(fmaxf(0.f, x)); }

CUDA_HOST_DEVICE inline float safe_asin(float x) { return ::asinf(clamp(x, -1, 1)); }

CUDA_HOST_DEVICE inline float safe_acos(float x) { return ::acosf(clamp(x, -1, 1)); }

template <typename T>
CUDA_HOST_DEVICE inline constexpr T sqr(T v)
{
    return v * v;
}

// http://www.plunk.org/~hatch/rightway.html
CUDA_HOST_DEVICE inline float sinx_over_x(float x)
{
    if (1 - x * x == 1)
        return 1;
    return ::sin(x) / x;
}

template <typename Float, typename C>
CUDA_HOST_DEVICE inline constexpr Float evaluate_polynomial(Float t, C c)
{
    return c;
}

template <typename Float, typename C, typename... Args>
CUDA_HOST_DEVICE inline constexpr Float evaluate_polynomial(Float t, C c, Args... cRemaining)
{
    return fma(t, evaluate_polynomial(t, cRemaining...), c);
}

template <typename Ta, typename Tb, typename Tc, typename Td>
CUDA_HOST_DEVICE inline auto difference_of_products(Ta a, Tb b, Tc c, Td d)
{
    auto cd = c * d;
    auto differenceOfProducts = fma(a, b, -cd);
    auto error = fma(-c, d, cd);
    return differenceOfProducts + error;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
CUDA_HOST_DEVICE inline auto sum_of_products(Ta a, Tb b, Tc c, Td d)
{
    auto cd = c * d;
    auto sumOfProducts = fma(a, b, cd);
    auto error = fma(c, d, -cd);
    return sumOfProducts + error;
}

// CompensatedFloat Definition
struct CompensatedFloat
{
  public:
    // CompensatedFloat Public Methods
    CUDA_HOST_DEVICE
    CompensatedFloat(float v, float err = 0) : v(v), err(err) {}
    CUDA_HOST_DEVICE
    explicit operator float() const { return v + err; }
    CUDA_HOST_DEVICE
    explicit operator double() const { return double(v) + double(err); }

    float v, err;
};

CUDA_HOST_DEVICE inline CompensatedFloat two_prod(float a, float b)
{
    float ab = a * b;
    return {ab, fma(a, b, -ab)};
}

CUDA_HOST_DEVICE inline CompensatedFloat two_sum(float a, float b)
{
    float s = a + b, delta = s - a;
    return {s, (a - (s - delta)) + (b - delta)};
}

namespace internal
{
// InnerProduct Helper Functions
template <typename Float>
CUDA_HOST_DEVICE inline CompensatedFloat inner_product(Float a, Float b)
{
    return two_prod(a, b);
}

// Accurate dot products with FMA: Graillat et al.,
// https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
//
// Accurate summation, dot product and polynomial evaluation in complex
// floating point arithmetic, Graillat and Menissier-Morain.
template <typename Float, typename... T>
CUDA_HOST_DEVICE inline CompensatedFloat inner_product(Float a, Float b, T... terms)
{
    CompensatedFloat ab = two_prod(a, b);
    CompensatedFloat tp = inner_product(terms...);
    CompensatedFloat sum = two_sum(ab.v, tp.v);
    return {sum.v, ab.err + (tp.err + sum.err)};
}

} // namespace internal

template <typename... T>
CUDA_HOST_DEVICE inline std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, float> inner_product(T... terms)
{
    CompensatedFloat ip = internal::inner_product(terms...);
    return float(ip);
}

template <typename T, typename U, typename V>
    requires std::is_floating_point_v<T> and std::is_floating_point_v<U> and std::is_floating_point_v<V>
CUDA_HOST_DEVICE inline auto _lerp(U t0, V t1, T t)
{
    return (1 - t) * t0 + t * t1;
}

enum class WrapMode
{
    Repeat,
    Clamp,
    Natural,
};

enum class TickMode
{
    Middle,
    Boundary,
};

CUDA_HOST_DEVICE
inline void lerp_helper(float u, int res, WrapMode wrap, TickMode tick, int &u0, int &u1, float &t)
{
    // ASSERT(u >= 0.0f && u <= 1.0f);
    float u_scaled;
    if (tick == TickMode::Middle) {
        u_scaled = u * (float)res - 0.5f;
    } else {
        u_scaled = u * (float)(res - 1);
    }
    u0 = floor(u_scaled);
    u1 = u0 + 1;
    t = u_scaled - u0;

    if (wrap == WrapMode::Repeat) {
        u0 = (u0 + res) % res;
        u1 = (u1 + res) % res;
    } else if (wrap == WrapMode::Clamp) {
        u0 = clamp(u0, 0, res - 1);
        u1 = clamp(u1, 0, res - 1);
    } // else Natural
}

template <int N>
CUDA_HOST_DEVICE inline void lerp_helper(const float *u, const int *res, WrapMode wrap, TickMode tick, int *u0, int *u1,
                                         float *t)
{
    for (int i = 0; i < N; ++i) {
        lerp_helper(u[i], res[i], wrap, tick, u0[i], u1[i], t[i]);
    }
}

template <int N>
CUDA_HOST_DEVICE inline void lerp_helper(const float *u, const int *res, const WrapMode *wrap, const TickMode *tick,
                                         int *u0, int *u1, float *t)
{
    for (int i = 0; i < N; ++i) {
        lerp_helper(u[i], res[i], wrap[i], tick[i], u0[i], u1[i], t[i]);
    }
}

// 0 is the lowest dimension
CUDA_HOST_DEVICE inline void unravel_index(int index, cuda::std::span<const int> dim, cuda::std::span<int> unraveled)
{
    int D = 1;
    for (int i = 0; i < (int)dim.size() - 1; ++i) {
        D *= dim[i];
    }
    for (int i = (int)dim.size() - 1; i >= 0; --i) {
        unraveled[i] = index / D;
        index -= unraveled[i] * D;
        D /= dim[i];
    }
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Also pbrt hair doc
// TODO: as of Haswell, the PEXT instruction could do all this in a
// single instruction.

// "Insert" a 0 bit after each of the 16 low bits of x
CUDA_HOST_DEVICE
constexpr uint32_t part_1_by_1(uint32_t x)
{
    x &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
CUDA_HOST_DEVICE
constexpr uint32_t part_1_by_2(uint32_t x)
{
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

// Inverse of part_1_by_1 - "delete" all odd-indexed bits
CUDA_HOST_DEVICE
constexpr uint32_t compact_1_by_1(uint32_t x)
{
    x &= 0x55555555;                 // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

// Inverse of part_1_by_2 - "delete" all bits not at positions divisible by 3
CUDA_HOST_DEVICE
constexpr uint32_t compact_1_by_2(uint32_t x)
{
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

CUDA_HOST_DEVICE
constexpr uint32_t encode_morton_2(uint32_t x, uint32_t y) { return (part_1_by_1(y) << 1) + part_1_by_1(x); }

CUDA_HOST_DEVICE
constexpr uint32_t encode_morton_3(uint32_t x, uint32_t y, uint32_t z)
{
    return (part_1_by_2(z) << 2) + (part_1_by_2(y) << 1) + part_1_by_2(x);
}

CUDA_HOST_DEVICE
constexpr void decode_morton_2(uint32_t code, uint32_t &x, uint32_t &y)
{
    x = compact_1_by_1(code >> 0);
    y = compact_1_by_1(code >> 1);
}

CUDA_HOST_DEVICE
constexpr void decode_morton_3(uint32_t code, uint32_t &x, uint32_t &y, uint32_t &z)
{
    x = compact_1_by_2(code >> 0);
    y = compact_1_by_2(code >> 1);
    z = compact_1_by_2(code >> 2);
}

CUDA_HOST_DEVICE inline float to_radian(float deg) { return (pi / 180) * deg; }
CUDA_HOST_DEVICE inline float to_degree(float rad) { return (180 / pi) * rad; }

CUDA_HOST_DEVICE
inline float srgb_to_linear(float x)
{
    if (x < 0.04045f) {
        return x / 12.92f;
    } else {
        return pow((x + 0.055f) / 1.055f, 2.4f);
    }
}

CUDA_HOST_DEVICE
inline float linear_to_srgb(float x)
{
    if (x < 0.0031308f) {
        return x * 12.92f;
    } else {
        return pow(x, 1.0f / 2.4f) * 1.055f - 0.055f;
    }
}

CUDA_HOST_DEVICE inline float fast_exp(float x)
{
#ifdef CUDA_IS_DEVICE_CODE
    return __expf(x);
#else
    return exp(x);
#endif
}

// Porting https://registry.khronos.org/OpenGL-Refpages/gl4/html/smoothstep.xhtml
CUDA_HOST_DEVICE inline float smoothstep(float edge0, float edge1, float x)
{
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

CUDA_HOST_DEVICE inline float erfinv(float x)
{
    float w, p;
    x = clamp(x, -.99999f, .99999f);
    w = -log((1 - x) * (1 + x));
    if (w < 5) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

} // namespace ksc