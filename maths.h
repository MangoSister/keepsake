#pragma once

#include <Eigen/Geometry>
#include <concepts>

using vec2 = Eigen::Vector2f;
using vec2i = Eigen::Vector2i;
using vec3 = Eigen::Vector3f;
using vec3i = Eigen::Vector3i;
using vec4 = Eigen::Vector4f;
using vec4i = Eigen::Vector4i;
using vec3d = Eigen::Vector3d;
using mat2 = Eigen::Matrix2f;
using mat3 = Eigen::Matrix3f;
using mat4 = Eigen::Matrix4f;
using quat = Eigen::Quaternionf;
using quatd = Eigen::Quaterniond;
template <int N>
using color = Eigen::Array<float, N, 1>;
using color3 = color<3>;
using color4 = color<4>;
using array3 = color<3>;
using array4 = color<4>;
template <int N>
using colord = Eigen::Array<double, N, 1>;
using color3d = colord<3>;
using color4d = colord<4>;

inline constexpr float pi = 3.14159265359f;
inline constexpr float two_pi = 6.28318530718f;
inline constexpr float half_pi = pi / 2.0f;
inline constexpr float quarter_pi = pi / 4.0f;
inline constexpr float sqrt_2 = 1.41421356237f;
inline constexpr float inv_pi = 1.0f / pi;
inline constexpr float inf = std::numeric_limits<float>::infinity();
inline const float before_one = std::nextafter(1.0f, -std::numeric_limits<float>::infinity());
inline constexpr int int_max = std::numeric_limits<int>::max();

template <typename T>
constexpr T square(const T &x)
{
    return x * x;
}

template <typename T>
constexpr T sign(const T &x)
{
    return T((T(0) < x) - (x < T(0)));
}

template <typename T>
constexpr T saturate(const T &x)
{
    return std::clamp(x, T(0), T(1));
}

template <typename T>
T safe_sqrt(const T &x)
{
    return std::sqrt(std::max(T(0), x));
}

template <typename T>
T safe_acos(T x)
{
    return std::acos(std::clamp(x, T(-1.0), T(1.0)));
}

inline vec3 lerp(const vec3 &v1, const vec3 &v2, float t) { return (1.0f - t) * v1 + t * v2; }
inline color3 lerp(const color3 &v1, const color3 &v2, float t) { return (1.0f - t) * v1 + t * v2; }
inline color4 lerp(const color4 &v1, const color4 &v2, float t) { return (1.0f - t) * v1 + t * v2; }

template <typename T>
inline T mod(T a, T b)
{
    T result = a - std::floor(a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template <typename T>
inline T fract(T x)
{
    return x - std::floor(x);
}

inline float luminance(const color3 &rgb)
{
    constexpr float lum_weight[3] = {0.212671f, 0.715160f, 0.072169f};
    return lum_weight[0] * rgb[0] + lum_weight[1] * rgb[1] + lum_weight[2] * rgb[2];
}

template <typename DerivedV, typename DerivedB>
auto clamp(const Eigen::ArrayBase<DerivedV> &v, const Eigen::ArrayBase<DerivedB> &low,
           const Eigen::ArrayBase<DerivedB> &high)
{
    return v.min(high).max(low);
}

template <typename T>
inline T clamp(const T &val, const T &low, const T &high)
{
    return std::clamp(val, low, high);
}

template <typename Derived>
inline auto clamp_negative(const Eigen::ArrayBase<Derived> &v)
{
    return v.max(v.Zero());
}

template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;
template <arithmetic T>
inline T clamp_negative(const T &v)
{
    return std::max(v, T(0));
}

template <typename T>
constexpr bool is_pow2(T v)
{
    return v && !(v & (v - 1));
}

constexpr int32_t round_up_pow2(int32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline int log2_int(uint32_t v)
{
#if defined(_MSC_VER)
    unsigned long lz = 0;
    if (_BitScanReverse(&lz, v))
        return lz;
    return 0;
#else
    return 31 - __builtin_clz(v);
#endif
}

inline int log2_int(int32_t v) { return log2_int((uint32_t)v); }

constexpr float to_radian(float degree) { return degree / 180.0f * pi; }

constexpr float to_degree(float radian) { return radian / pi * 180.0f; }

inline void to_spherical(const vec3 &dir, float &phi, float &theta)
{
    theta = std::acos(std::clamp(dir.z(), -1.0f, 1.0f));
    phi = std::atan2(dir.y(), dir.x());
    if (phi < 0.0f) {
        phi += two_pi;
    }
}

inline vec3 to_cartesian(float phi, float theta)
{
    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);
    float cos_phi = std::cos(phi);
    float sin_phi = std::sin(phi);
    return vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

inline bool solve_linear_system_2x2(const float A[2][2], const float B[2], float &x0, float &x1)
{
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (std::abs(det) < 1e-10f)
        return false;
    x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
    x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
    if (std::isnan(x0) || std::isnan(x1))
        return false;
    return true;
}

inline bool solve_quadratic(float a, float b, float c, float &t0, float &t1)
{
    // Find quadratic discriminant
    double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
    if (discrim < 0)
        return false;
    discrim = std::sqrt(discrim);

    // Compute quadratic _t_ values
    double q;
    if (b < 0)
        q = -.5 * (b - discrim);
    else
        q = -.5 * (b + discrim);
    t0 = q / a;
    t1 = c / q;
    if (t0 > t1)
        std::swap(t0, t1);
    return true;
}

// Duff, Tom, et al. "Building an orthonormal basis, revisited." Journal of Computer Graphics Techniques Vol 6.1 (2017).
inline void orthonormal_basis(const vec3 &N, vec3 &X, vec3 &Y)
{
    float sign = copysignf(1.0f, N.z());
    const float a = -1.0f / (sign + N.z());
    const float b = N.x() * N.y() * a;
    X = vec3(1.0f + sign * N.x() * N.x() * a, sign * b, -sign * N.x());
    Y = vec3(b, sign + N.y() * N.y() * a, -N.y());
}

struct Frame
{
    Frame()
    {
        t = vec3::UnitX();
        b = vec3::UnitY();
        n = vec3::UnitZ();
    }

    Frame(const vec3 &t, const vec3 &b, const vec3 &n) : t(t), b(b), n(n) {}

    explicit Frame(const vec3 &n) : n(n) { orthonormal_basis(n, t, b); }

    bool valid() const
    {
        return (std::abs(t.squaredNorm() - 1.0f) <= 1e-4f) && (std::abs(b.squaredNorm() - 1.0f) <= 1e-4f) &&
               (std::abs(n.squaredNorm() - 1.0f) <= 1e-4f) && (t.isOrthogonal(b, 1e-4f)) &&
               (b.isOrthogonal(n, 1e-4f)) && (n.isOrthogonal(t, 1e-4f));
    }

    vec3 to_local(const vec3 &w) const { return vec3(t.dot(w), b.dot(w), n.dot(w)); }

    vec3 to_world(const vec3 &w) const { return t * w.x() + b * w.y() + n * w.z(); }

    vec3 t, b, n;
};

inline mat4 scale_rotate_translate(const vec3 &scale, const quat &rototation, const vec3 &translatation)
{
    mat4 S = mat4::Identity();
    for (int i = 0; i < 3; ++i) {
        S(i, i) = scale(i);
    }
    mat4 R = mat4::Identity();
    R.block<3, 3>(0, 0) = rototation.matrix();

    mat4 T = mat4::Identity();
    for (int i = 0; i < 3; ++i) {
        T(i, 3) = translatation(i);
    }
    return T * R * S;
}

constexpr float srgb_to_linear(float x)
{
    if (x < 0.04045f) {
        return x / 12.92f;
    } else {
        return std::pow((x + 0.055f) / 1.055f, 2.4f);
    }
}

// Is this stable?
template <typename T>
inline T sinc(T x)
{
    if (x == T(0)) {
        return T(1);
    }
    using std::sin;
    return sin(x) / x;
}

enum class WrapMode
{
    Repeat,
    Clamp
};

enum class TickMode
{
    Middle,
    Boundary,
};

template <WrapMode wrap_x, WrapMode wrap_y>
inline void bilinear_helper(const vec2 &uv, const vec2i &res, int &u0, int &u1, int &v0, int &v1, float &t0, float &t1)
{
    static_assert(wrap_x == WrapMode::Repeat || wrap_x == WrapMode::Clamp, "Invalid wrap option.");
    static_assert(wrap_y == WrapMode::Repeat || wrap_y == WrapMode::Clamp, "Invalid wrap option.");

    vec2 uv_scaled = uv.cwiseProduct(res.cast<float>()) - vec2::Constant(0.5f);
    u0 = std::floor(uv_scaled.x());
    u1 = u0 + 1;
    v0 = std::floor(uv_scaled.y());
    v1 = v0 + 1;
    t0 = uv_scaled.x() - u0;
    t1 = uv_scaled.y() - v0;

    if constexpr (wrap_x == WrapMode::Repeat) {
        u0 = (u0 + res.x()) % res.x();
        u1 = (u1 + res.x()) % res.x();
    } else { // Clamp
        u0 = std::clamp(u0, 0, res.x() - 1);
        u1 = std::clamp(u1, 0, res.x() - 1);
    }

    if constexpr (wrap_y == WrapMode::Repeat) {
        v0 = (v0 + res.y()) % res.y();
        v1 = (v1 + res.y()) % res.y();
    } else { // Clamp
        v0 = std::clamp(v0, 0, res.y() - 1);
        v1 = std::clamp(v1, 0, res.y() - 1);
    }
}

inline void lerp_helper(float u, int res, WrapMode wrap, TickMode tick, int &u0, int &u1, float &t)
{
    // ASSERT(u >= 0.0f && u <= 1.0f);
    float u_scaled;
    if (tick == TickMode::Middle) {
        u_scaled = u * (float)res - 0.5f;
    } else {
        u_scaled = u * (float)(res - 1);
    }
    u0 = std::floor(u_scaled);
    u1 = u0 + 1;
    t = u_scaled - u0;

    if (wrap == WrapMode::Repeat) {
        u0 = (u0 + res) % res;
        u1 = (u1 + res) % res;
    } else { // Clamp
        u0 = std::clamp(u0, 0, res - 1);
        u1 = std::clamp(u1, 0, res - 1);
    }
}

template <int N>
inline void lerp_helper(const float *u, const int *res, WrapMode wrap, TickMode tick, int *u0, int *u1, float *t)
{
    for (int i = 0; i < N; ++i) {
        lerp_helper(u[i], res[i], wrap, tick, u0[i], u1[i], t[i]);
    }
}

template <int N>
inline void lerp_helper(const float *u, const int *res, const WrapMode *wrap, const TickMode *tick, int *u0, int *u1,
                        float *t)
{
    for (int i = 0; i < N; ++i) {
        lerp_helper(u[i], res[i], wrap[i], tick[i], u0[i], u1[i], t[i]);
    }
}

inline vec3 transform_dir(const mat4 &m, const vec3 &d) { return m.block<3, 3>(0, 0) * d; }

inline vec3 transform_point(const mat4 &m, const vec3 &p) { return (m * p.homogeneous()).head(3); }

inline float int_as_float(int i)
{
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

inline int float_as_int(float f)
{
    int i;
    std::memcpy(&i, &f, 4);
    return i;
}

inline float uint_as_float(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint32_t float_as_uint(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

inline float delta_angle(float alpha, float beta)
{
    float phi = std::fmod(std::abs(beta - alpha), two_pi);
    float distance = phi > pi ? two_pi - phi : phi;
    return distance;
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Also pbrt hair doc
// TODO: as of Haswell, the PEXT instruction could do all this in a
// single instruction.

// "Insert" a 0 bit after each of the 16 low bits of x
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
constexpr uint32_t compact_1_by_2(uint32_t x)
{
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

constexpr uint32_t encode_morton_2(uint32_t x, uint32_t y) { return (part_1_by_1(y) << 1) + part_1_by_1(x); }

constexpr uint32_t encode_morton_3(uint32_t x, uint32_t y, uint32_t z)
{
    return (part_1_by_2(z) << 2) + (part_1_by_2(y) << 1) + part_1_by_2(x);
}

constexpr void decode_morton_2(uint32_t code, uint32_t &x, uint32_t &y)
{
    x = compact_1_by_1(code >> 0);
    y = compact_1_by_1(code >> 1);
}

inline void decode_morton_3(uint32_t code, uint32_t &x, uint32_t &y, uint32_t &z)
{
    x = compact_1_by_2(code >> 0);
    y = compact_1_by_2(code >> 1);
    z = compact_1_by_2(code >> 2);
}