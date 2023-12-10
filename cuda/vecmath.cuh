#pragma once

#include "array.cuh"
#include "math.cuh"
#include "optional.cuh"
#include "span.cuh"

namespace ksc
{

template <typename T>
struct VectorLength
{
    using type = float;
};

template <>
struct VectorLength<double>
{
    using type = double;
};

template <>
struct VectorLength<long double>
{
    using type = long double;
};

template <typename T>
struct Vector2
{
  public:
    static const int nDimensions = 2;

    CUDA_HOST_DEVICE
    static constexpr Vector2 zero() { return Vector2(T(0), T(0)); }

    Vector2() = default;
    CUDA_HOST_DEVICE
    explicit constexpr Vector2(T s) : x(s), y(s) {}
    CUDA_HOST_DEVICE
    constexpr Vector2(T x, T y) : x(x), y(y) {}
    CUDA_HOST_DEVICE
    bool has_nan() const { return IsNaN(x) || IsNaN(y); }

    CUDA_HOST_DEVICE
    bool isfinite() const { return ::isfinite(x) && ::isfinite(y); }

    template <typename U>
    CUDA_HOST_DEVICE auto operator+(Vector2<U> c) const -> Vector2<decltype(T{} + U{})>
    {
        return {x + c.x, y + c.y};
    }
    template <typename U>
    CUDA_HOST_DEVICE Vector2<T> &operator+=(Vector2<U> c)
    {
        x += c.x;
        y += c.y;
        return static_cast<Vector2<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator-(Vector2<U> c) const -> Vector2<decltype(T{} - U{})>
    {
        return {x - c.x, y - c.y};
    }
    template <typename U>
    CUDA_HOST_DEVICE Vector2<T> &operator-=(Vector2<U> c)
    {
        x -= c.x;
        y -= c.y;
        return static_cast<Vector2<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator*(Vector2<U> c) const -> Vector2<decltype(T{} + U{})>
    {
        return {x * c.x, y * c.y};
    }
    template <typename U>
    CUDA_HOST_DEVICE Vector2<T> &operator*=(Vector2<U> c)
    {
        x *= c.x;
        y *= c.y;
        return static_cast<Vector2<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator/(Vector2<U> c) const -> Vector2<decltype(T{} + U{})>
    {
        return {x / c.x, y / c.y};
    }
    template <typename U>
    CUDA_HOST_DEVICE Vector2<T> &operator/=(Vector2<U> c)
    {
        x /= c.x;
        y /= c.y;
        return static_cast<Vector2<T> &>(*this);
    }

    CUDA_HOST_DEVICE
    bool operator==(Vector2<T> c) const { return x == c.x && y == c.y; }
    CUDA_HOST_DEVICE
    bool operator!=(Vector2<T> c) const { return x != c.x || y != c.y; }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator*(U s) const -> Vector2<decltype(T{} * U{})>
    {
        return {s * x, s * y};
    }
    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector2<T> &operator*=(U s)
    {
        x *= s;
        y *= s;
        return static_cast<Vector2<T> &>(*this);
    }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator/(U d) const -> Vector2<decltype(T{} / U{})>
    {
        return {x / d, y / d};
    }
    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector2<T> &operator/=(U d)
    {
        x /= d;
        y /= d;
        return static_cast<Vector2<T> &>(*this);
    }

    CUDA_HOST_DEVICE
    Vector2<T> operator-() const { return {-x, -y}; }

    CUDA_HOST_DEVICE
    T operator[](int i) const { return (i == 0) ? x : y; }

    CUDA_HOST_DEVICE
    T &operator[](int i) { return (i == 0) ? x : y; }

    template <typename U>
    CUDA_HOST_DEVICE explicit operator Vector2<U>() const
    {
        return Vector2<U>((U)x, (U)y);
    }

    T x{}, y{};
};

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator*(U s, Vector2<T> t) -> Vector2<decltype(T{} * U{})>
{
    return t * s;
}

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator/(U s, Vector2<T> t) -> Vector2<decltype(U{} / T{})>
{
    return {s / t.x, s / t.y};
}

template <typename T>
CUDA_HOST_DEVICE inline Vector2<T> min(Vector2<T> t0, Vector2<T> t1)
{
    return {min(t0.x, t1.x), min(t0.y, t1.y)};
}

template <typename T>
CUDA_HOST_DEVICE inline T min_component_value(Vector2<T> t)
{
    return min({t.x, t.y});
}

template <typename T>
CUDA_HOST_DEVICE inline int min_component_index(Vector2<T> t)
{
    return (t.x < t.y) ? 0 : 1;
}

template <typename T>
CUDA_HOST_DEVICE inline Vector2<T> max(Vector2<T> t0, Vector2<T> t1)
{
    return {max(t0.x, t1.x), max(t0.y, t1.y)};
}

template <typename T>
CUDA_HOST_DEVICE inline T max_component_value(Vector2<T> t)
{
    return max(t.x, t.y);
}

template <typename T>
CUDA_HOST_DEVICE inline int max_component_index(Vector2<T> t)
{
    return (t.x > t.y) ? 0 : 1;
}

template <typename T>
CUDA_HOST_DEVICE inline Vector2<T> permute(Vector2<T> t, ksc::array<int, 2> p)
{
    return {t[p[0]], t[p[1]]};
}

template <typename T>
CUDA_HOST_DEVICE inline T hprod(Vector2<T> t)
{
    return t.x * t.y;
}

template <typename T>
CUDA_HOST_DEVICE inline Vector2<T> abs(Vector2<T> t)
{
    return {Abs(t.x), Abs(t.y)};
}

template <typename T>
CUDA_HOST_DEVICE inline auto lerp(float t, Vector2<T> t0, Vector2<T> t1)
{
    return (1 - t) * t0 + t * t1;
}

template <typename T>
CUDA_HOST_DEVICE inline auto lerp(Vector2<T> t, Vector2<T> t0, Vector2<T> t1)
{
    return (Vector2<T>(1) - t) * t0 + t * t1;
}

template <typename T>
CUDA_HOST_DEVICE inline auto dot(Vector2<T> v1, Vector2<T> v2) -> typename VectorLength<T>::type
{
    return sum_of_products(v1.x, v2.x, v1.y, v2.y);
}

template <typename T>
CUDA_HOST_DEVICE inline auto abs_dot(Vector2<T> v1, Vector2<T> v2) -> typename VectorLength<T>::type
{
    return abs(dot(v1, v2));
}

// floating point only

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector2<T> ceil(Vector2<T> t)
{
    return {ceil(t.x), ceil(t.y)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector2<T> floor(Vector2<T> t)
{
    return {floor(t.x), floor(t.y)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector2<T> fma(float a, Vector2<T> b, Vector2<T> c)
{
    return {fma(a, b.x, c.x), fma(a, b.y, c.y)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector2<T> fma(Vector2<T> a, float b, Vector2<T> c)
{
    return fma(b, a, c);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto length_squared(Vector2<T> v) -> typename VectorLength<T>::type
{
    return sqr(v.x) + sqr(v.y);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto length(Vector2<T> v) -> typename VectorLength<T>::type
{
    return sqrt(length_squared(v));
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto normalize(Vector2<T> v)
{
    return v / length(v);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto distance(Vector2<T> p1, Vector2<T> p2) -> typename VectorLength<T>::type
{
    return length(p1 - p2);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto distance_squared(Vector2<T> p1, Vector2<T> p2) -> typename VectorLength<T>::type
{
    return length_squared(p1 - p2);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto cross(Vector2<T> v1, Vector2<T> v2)
{
    return difference_of_products(v1.x, v2.y, v1.y, v2.x);
}

////////////////////////////////////////////////////////////////////////

template <typename T>
struct Vector3
{
  public:
    static const int nDimensions = 3;

    CUDA_HOST_DEVICE
    static constexpr Vector3 zero() { return Vector3(T(0), T(0), T(0)); }

    Vector3() = default;
    CUDA_HOST_DEVICE
    explicit constexpr Vector3(T s) : x(s), y(s), z(s) {}
    CUDA_HOST_DEVICE
    constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}

    CUDA_HOST_DEVICE
    bool has_nan() const { return ::isnan(x) || ::isnan(y) || ::isnan(z); }

    CUDA_HOST_DEVICE
    bool isfinite() const { return ::isfinite(x) && ::isfinite(y) && ::isfinite(z); }

    CUDA_HOST_DEVICE
    T operator[](int i) const
    {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    CUDA_HOST_DEVICE
    T &operator[](int i)
    {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator+(Vector3<U> c) const -> Vector3<decltype(T{} + U{})>
    {
        return {x + c.x, y + c.y, z + c.z};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector3<T> &operator+=(Vector3<U> c)
    {
        x += c.x;
        y += c.y;
        z += c.z;
        return static_cast<Vector3<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator-(Vector3<U> c) const -> Vector3<decltype(T{} - U{})>
    {
        return {x - c.x, y - c.y, z - c.z};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector3<T> &operator-=(Vector3<U> c)
    {
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return static_cast<Vector3<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator*(Vector3<U> c) const -> Vector3<decltype(T{} + U{})>
    {
        return {x * c.x, y * c.y, z * c.z};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector3<T> &operator*=(Vector3<U> c)
    {
        x *= c.x;
        y *= c.y;
        z *= c.z;
        return static_cast<Vector3<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator/(Vector3<U> c) const -> Vector3<decltype(T{} + U{})>
    {
        return {x / c.x, y / c.y, z / c.z};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector3<T> &operator/=(Vector3<U> c)
    {
        x /= c.x;
        y /= c.y;
        z /= c.z;
        return static_cast<Vector3<T> &>(*this);
    }

    CUDA_HOST_DEVICE
    bool operator==(Vector3<T> c) const { return x == c.x && y == c.y && z == c.z; }
    CUDA_HOST_DEVICE
    bool operator!=(Vector3<T> c) const { return x != c.x || y != c.y || z != c.z; }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator*(U s) const -> Vector3<decltype(T{} * U{})>
    {
        return {s * x, s * y, s * z};
    }
    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector3<T> &operator*=(U s)
    {
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Vector3<T> &>(*this);
    }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator/(U d) const -> Vector3<decltype(T{} / U{})>
    {
        return {x / d, y / d, z / d};
    }
    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector3<T> &operator/=(U d)
    {
        x /= d;
        y /= d;
        z /= d;
        return static_cast<Vector3<T> &>(*this);
    }
    CUDA_HOST_DEVICE
    Vector3<T> operator-() const { return {-x, -y, -z}; }

    template <typename U>
    CUDA_HOST_DEVICE explicit operator Vector2<U>() const
    {
        return Vector2<U>((U)x, (U)y);
    }

    T x{}, y{}, z{};
};

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator*(U s, Vector3<T> t) -> Vector3<decltype(T{} * U{})>
{
    return t * s;
}

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator/(U s, Vector3<T> t) -> Vector3<decltype(U{} / T{})>
{
    return {s / t.x, s / t.y, s / t.z};
}

template <typename T>
CUDA_HOST_DEVICE inline Vector3<T> min(Vector3<T> t1, Vector3<T> t2)
{
    return {min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z)};
}

template <typename T>
CUDA_HOST_DEVICE inline T min_component_value(Vector3<T> t)
{
    return min(min(t.x, t.y), t.z);
}

template <typename T>
CUDA_HOST_DEVICE inline int min_component_index(Vector3<T> t)
{
    return (t.x < t.y) ? ((t.x < t.z) ? 0 : 2) : ((t.y < t.z) ? 1 : 2);
}

template <typename T>
CUDA_HOST_DEVICE inline Vector3<T> max(Vector3<T> t1, Vector3<T> t2)
{
    return {max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z)};
}

template <typename T>
CUDA_HOST_DEVICE inline T max_component_value(Vector3<T> t)
{
    return max(max(t.x, t.y), t.z);
}

template <typename T>
CUDA_HOST_DEVICE inline int max_component_index(Vector3<T> t)
{
    return (t.x > t.y) ? ((t.x > t.z) ? 0 : 2) : ((t.y > t.z) ? 1 : 2);
}

template <typename T>
CUDA_HOST_DEVICE inline Vector3<T> permute(Vector3<T> t, ksc::array<int, 3> p)
{
    return {t[p[0]], t[p[1]], t[p[2]]};
}

template <typename T>
CUDA_HOST_DEVICE inline T hprod(Vector3<T> t)
{
    return t.x * t.y * t.z;
}

template <typename T>
CUDA_HOST_DEVICE inline Vector3<T> abs(Vector3<T> t)
{
    return {abs(t.x), abs(t.y), abs(t.z)};
}

template <typename T>
CUDA_HOST_DEVICE inline auto lerp(float t, Vector3<T> t0, Vector3<T> t1)
{
    return (1 - t) * t0 + t * t1;
}

template <typename T>
CUDA_HOST_DEVICE inline auto lerp(Vector3<T> t, Vector3<T> t0, Vector3<T> t1)
{
    return (Vector3<T>(1) - t) * t0 + t * t1;
}

template <typename T>
CUDA_HOST_DEVICE inline auto dot(Vector3<T> v, Vector3<T> w) -> typename VectorLength<T>::type
{
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

template <typename T>
CUDA_HOST_DEVICE inline auto abs_dot(Vector3<T> v1, Vector3<T> v2) -> typename VectorLength<T>::type
{
    return abs(dot(v1, v2));
}

// floating point only

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> ceil(Vector3<T> t)
{
    return {ceil(t.x), ceil(t.y), ceil(t.z)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> floor(Vector3<T> t)
{
    return {floor(t.x), floor(t.y), floor(t.z)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> fma(float a, Vector3<T> b, Vector3<T> c)
{
    return {fma(a, b.x, c.x), fma(a, b.y, c.y), fma(a, b.z, c.z)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> fma(Vector3<T> a, float b, Vector3<T> c)
{
    return fma(b, a, c);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> cross(Vector3<T> v1, Vector3<T> v2)
{
    return {difference_of_products(v1.y, v2.z, v1.z, v2.y), difference_of_products(v1.z, v2.x, v1.x, v2.z),
            difference_of_products(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline T length_squared(Vector3<T> v)
{
    return sqr(v.x) + sqr(v.y) + sqr(v.z);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto length(Vector3<T> v) -> typename VectorLength<T>::type
{
    return sqrt(length_squared(v));
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto distance(Vector3<T> p1, Vector3<T> p2)
{
    return length(p1 - p2);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto distance_squared(Vector3<T> p1, Vector3<T> p2)
{
    return length_squared(p1 - p2);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto normalize(Vector3<T> v)
{
    return v / length(v);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto average(Vector3<T> v) -> typename VectorLength<T>::type
{
    return (v.x + v.y + v.z) / T(3.0);
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// via http://www.plunk.org/~hatch/rightway.html
template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto angle_between(Vector3<T> v1, Vector3<T> v2) -> typename VectorLength<T>::type
{
    if (dot(v1, v2) < 0)
        return pi - 2 * safe_asin(length(v1 + v2) / 2);
    else
        return 2 * safe_asin(length(v2 - v1) / 2);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline Vector3<T> gram_schmidt(Vector3<T> v, Vector3<T> w)
{
    return v - dot(v, w) * w;
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline void orthonormal_basis(Vector3<T> N, Vector3<T> &X, Vector3<T> &Y)
{
    float sign = copysign(float(1), N.z);
    float a = -1 / (sign + N.z);
    float b = N.x * N.y * a;
    X = Vector3<T>(1 + sign * sqr(N.x) * a, sign * b, -sign * N.x);
    Y = Vector3<T>(b, sign + sqr(N.y) * a, -N.y);
}

///////////////////////

template <typename T>
struct Vector4
{
  public:
    static constexpr int nDimensions = 4;

    CUDA_HOST_DEVICE
    static constexpr Vector4 zero() { return Vector4(T(0), T(0), T(0), T(0)); }

    Vector4() = default;
    CUDA_HOST_DEVICE
    explicit constexpr Vector4(T s) : x(s), y(s), z(s), w(s) {}
    CUDA_HOST_DEVICE
    constexpr Vector4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

    CUDA_HOST_DEVICE
    bool has_nan() const { return ::isnan(x) || ::isnan(y) || ::isnan(z) || ::isnan(w); }

    CUDA_HOST_DEVICE
    bool isfinite() const { return ::isfinite(x) && ::isfinite(y) && ::isfinite(z) && ::isfinite(w); }

    CUDA_HOST_DEVICE
    T operator[](int i) const
    {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        if (i == 2)
            return z;
        return w;
    }

    CUDA_HOST_DEVICE
    T &operator[](int i)
    {
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        if (i == 2)
            return z;
        return w;
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator+(Vector4<U> c) const -> Vector4<decltype(T{} + U{})>
    {
        return {x + c.x, y + c.y, z + c.z, w + c.w};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector4<T> &operator+=(Vector4<U> c)
    {
        x += c.x;
        y += c.y;
        z += c.z;
        w += c.w;
        return static_cast<Vector4<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator-(Vector4<U> c) const -> Vector4<decltype(T{} - U{})>
    {
        return {x - c.x, y - c.y, z - c.z, w - c.w};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector4<T> &operator-=(Vector4<U> c)
    {
        x -= c.x;
        y -= c.y;
        z -= c.z;
        w -= c.w;
        return static_cast<Vector4<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator*(Vector4<U> c) const -> Vector4<decltype(T{} + U{})>
    {
        return {x * c.x, y * c.y, z * c.z, w * c.w};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector4<T> &operator*=(Vector4<U> c)
    {
        x *= c.x;
        y *= c.y;
        z *= c.z;
        w *= c.w;
        return static_cast<Vector4<T> &>(*this);
    }

    template <typename U>
    CUDA_HOST_DEVICE auto operator/(Vector4<U> c) const -> Vector4<decltype(T{} + U{})>
    {
        return {x / c.x, y / c.y, z / c.z, w / c.w};
    }

    template <typename U>
    CUDA_HOST_DEVICE Vector4<T> &operator/=(Vector4<U> c)
    {
        x /= c.x;
        y /= c.y;
        z /= c.z;
        w /= c.w;
        return static_cast<Vector4<T> &>(*this);
    }

    CUDA_HOST_DEVICE
    bool operator==(Vector4<T> c) const { return x == c.x && y == c.y && z == c.z && w == c.w; }
    CUDA_HOST_DEVICE
    bool operator!=(Vector4<T> c) const { return x != c.x || y != c.y || z != c.z || w != c.w; }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator*(U s) const -> Vector4<decltype(T{} * U{})>
    {
        return {s * x, s * y, s * z, s * w};
    }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector4<T> &operator*=(U s)
    {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return static_cast<Vector4<T> &>(*this);
    }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE auto operator/(U d) const -> Vector4<decltype(T{} / U{})>
    {
        return {x / d, y / d, z / d, w / d};
    }

    template <typename U>
        requires std::is_arithmetic_v<U>
    CUDA_HOST_DEVICE Vector4<T> &operator/=(U d)
    {
        x /= d;
        y /= d;
        z /= d;
        w /= d;
        return static_cast<Vector4<T> &>(*this);
    }

    CUDA_HOST_DEVICE
    Vector4<T> operator-() const { return {-x, -y, -z, -w}; }

    template <typename U>
    CUDA_HOST_DEVICE explicit operator Vector3<U>() const
    {
        return Vector3<U>((U)x, (U)y, (U)z);
    }

    template <typename U>
    CUDA_HOST_DEVICE explicit operator Vector2<U>() const
    {
        return Vector2<U>((U)x, (U)y);
    }

    T x{}, y{}, z{}, w{};
};

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator*(U s, Vector4<T> t) -> Vector4<decltype(T{} * U{})>
{
    return t * s;
}

template <typename T, typename U>
    requires std::is_arithmetic_v<U>
CUDA_HOST_DEVICE inline auto operator/(U s, Vector4<T> t) -> Vector4<decltype(U{} / T{})>
{
    return {s / t.x, s / t.y, s / t.z, s / t.w};
}

template <typename T>
CUDA_HOST_DEVICE inline T max_component_value(Vector4<T> t)
{
    return max(max(max(t.x, t.y), t.z), t.w);
}

template <typename T>
CUDA_HOST_DEVICE inline int max_component_index(Vector4<T> t)
{
    if (t.x > t.y) {
        if (t.x > t.z) {
            if (t.x > t.w) {
                return 0;
            } else {
                return 3;
            }
        } else if (t.z > t.w) {
            return 2;
        } else {
            return 3;
        }
    } else if (t.y > t.z) {
        if (t.y > t.w) {
            return 1;
        } else {
            return 3;
        }
    } else if (t.z > t.w) {
        return 2;
    } else {
        return 3;
    }
}

template <typename T>
CUDA_HOST_DEVICE inline Vector4<T> abs(Vector4<T> t)
{
    return {abs(t.x), abs(t.y), abs(t.z), abs(t.w)};
}

template <typename T>
CUDA_HOST_DEVICE inline auto dot(Vector4<T> v1, Vector4<T> v2) -> typename VectorLength<T>::type
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

template <typename T>
CUDA_HOST_DEVICE inline auto abs_dot(Vector4<T> v1, Vector4<T> v2) -> typename VectorLength<T>::type
{
    return abs(dot(v1, v2));
}

// floating point only

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto length_squared(Vector4<T> v) -> typename VectorLength<T>::type
{
    return sqr(v.x) + sqr(v.y) + sqr(v.z) + sqr(v.w);
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto length(Vector4<T> v) -> typename VectorLength<T>::type
{
    return sqrt(length_squared(v));
}

template <typename T>
    requires std::is_floating_point_v<T>
CUDA_HOST_DEVICE inline auto normalize(Vector4<T> v)
{
    return v / length(v);
}

///////////////////////

using vec2 = Vector2<float>;
using vec2f = Vector2<float>;
using vec2i = Vector2<int>;

using vec3 = Vector3<float>;
using vec3f = Vector3<float>;
using vec3i = Vector3<int>;

using vec4 = Vector4<float>;
using vec4f = Vector4<float>;
using vec4i = Vector4<int>;

using color3 = Vector3<float>;
using color4 = Vector4<float>;

///////////////////////

namespace
{

template <int N>
CUDA_HOST_DEVICE inline void init(float m[N][N], int i, int j)
{}

template <int N, typename... Args>
CUDA_HOST_DEVICE inline void init(float m[N][N], int i, int j, float v, Args... args)
{
    m[i][j] = v;
    if (++j == N) {
        ++i;
        j = 0;
    }
    init<N>(m, i, j, args...);
}

template <int N>
CUDA_HOST_DEVICE inline void init_diag(float m[N][N], int i)
{}

template <int N, typename... Args>
CUDA_HOST_DEVICE inline void init_diag(float m[N][N], int i, float v, Args... args)
{
    for (int j = 0; j < N; ++j) {
        m[i][j] = (i == j) ? v : 0.0f;
    }
    init_diag<N>(m, i + 1, args...);
}

template <int N>
CUDA_HOST_DEVICE inline void assign_row(float r[N], int i)
{}

template <int N, typename... Args>
CUDA_HOST_DEVICE inline void assign_row(float r[N], int i, float v, Args... args)
{
    r[i] = v;
    assign_row<N>(r, i + 1, args...);
}

template <int N>
CUDA_HOST_DEVICE inline void assign_col(float m[N][N], int col, int i)
{}

template <int N, typename... Args>
CUDA_HOST_DEVICE inline void assign_col(float m[N][N], int col, int i, float v, Args... args)
{
    m[i][col] = v;
    assign_col<N>(m, col, i + 1, args...);
}

} // namespace

// SquareMatrix Definition
template <int N>
struct SquareMatrix
{
    // SquareMatrix Public Methods
    CUDA_HOST_DEVICE
    static SquareMatrix zero()
    {
        SquareMatrix m;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m.m[i][j] = 0;
        return m;
    }

    // SquareMatrix Public Methods
    CUDA_HOST_DEVICE
    static SquareMatrix identity()
    {
        SquareMatrix m;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m.m[i][j] = i == j ? 1 : 0;
        return m;
    }

    SquareMatrix() = default;

    CUDA_HOST_DEVICE
    SquareMatrix(const float mat[N][N])
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = mat[i][j];
    }
    CUDA_HOST_DEVICE
    SquareMatrix(ksc::span<const float> t);
    template <typename... Args>
    CUDA_HOST_DEVICE SquareMatrix(float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N * N, "Incorrect number of values provided to SquareMatrix constructor");
        init<N>(m, 0, 0, v, args...);
    }
    template <typename... Args>
    CUDA_HOST_DEVICE static SquareMatrix diag(float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N, "Incorrect number of values provided to SquareMatrix::Diag");
        SquareMatrix m;
        init_diag<N>(m.m, 0, v, args...);
        return m;
    }

    CUDA_HOST_DEVICE
    SquareMatrix operator+(const SquareMatrix &m) const
    {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] += m.m[i][j];
        return r;
    }

    CUDA_HOST_DEVICE
    SquareMatrix &operator+=(const SquareMatrix &m)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                this->m[i][j] += m.m[i][j];
        return *this;
    }

    CUDA_HOST_DEVICE
    SquareMatrix operator-(const SquareMatrix &m) const
    {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] -= m.m[i][j];
        return r;
    }

    CUDA_HOST_DEVICE
    SquareMatrix &operator-=(const SquareMatrix &m)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                this->m[i][j] -= m.m[i][j];
        return *this;
    }

    CUDA_HOST_DEVICE
    SquareMatrix operator*(float s) const
    {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] *= s;
        return r;
    }

    CUDA_HOST_DEVICE
    SquareMatrix &operator*=(float s)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] *= s;
        return *this;
    }

    CUDA_HOST_DEVICE
    SquareMatrix operator/(float s) const
    {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] /= s;
        return r;
    }

    CUDA_HOST_DEVICE
    SquareMatrix &operator/=(float s)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] /= s;
        return *this;
    }

    CUDA_HOST_DEVICE
    SquareMatrix operator-() const
    {
        SquareMatrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] = -r.m[i][j];
        return r;
    }

    CUDA_HOST_DEVICE
    bool operator==(const SquareMatrix<N> &m2) const
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return false;
        return true;
    }

    CUDA_HOST_DEVICE
    bool operator!=(const SquareMatrix<N> &m2) const
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return true;
        return false;
    }

    CUDA_HOST_DEVICE
    bool operator<(const SquareMatrix<N> &m2) const
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (m[i][j] < m2.m[i][j])
                    return true;
                if (m[i][j] > m2.m[i][j])
                    return false;
            }
        return false;
    }

    CUDA_HOST_DEVICE
    bool is_identity() const;

    CUDA_HOST_DEVICE
    ksc::span<const float> operator[](int i) const { return m[i]; }
    CUDA_HOST_DEVICE
    ksc::span<float> operator[](int i) { return ksc::span<float>(m[i]); }

    template <typename T>
    CUDA_HOST_DEVICE T col(int j) const
    {
        T c{};
        for (int i = 0; i < N; ++i)
            c[i] = m[i][j];
        return c;
    }

    template <typename T>
    CUDA_HOST_DEVICE void set_row(int row, const T &v)
    {
        for (int i = 0; i < N; ++i)
            m[row][i] = v[i];
    }

    template <typename T>
    CUDA_HOST_DEVICE void set_col(int col, const T &v)
    {
        for (int i = 0; i < N; ++i)
            m[i][col] = v[i];
    }

    template <typename... Args>
    CUDA_HOST_DEVICE void assign_row(int row, float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N, "Incorrect number of values provided to SquareMatrix constructor");
        ksc::assign_row<N>(m[row], 0, v, args...);
    }

    template <typename... Args>
    CUDA_HOST_DEVICE void assign_col(int col, float v, Args... args)
    {
        static_assert(1 + sizeof...(Args) == N, "Incorrect number of values provided to SquareMatrix constructor");
        ksc::assign_col<N>(m, col, 0, v, args...);
    }

    CUDA_HOST_DEVICE float minor() const
        requires(N == 1)
    {
        return m[0][0];
    }

    CUDA_HOST_DEVICE SquareMatrix<N - 1> minor() const
        requires(N > 2)
    {
        SquareMatrix<N - 1> mi;
        for (int i = 0; i < N - 1; ++i)
            for (int j = 0; j < N - 1; ++j) {
                mi.m[i][j] = m[i][j];
            }
        return mi;
    }

    CUDA_HOST_DEVICE vec3 translation() const
        requires(N == 4)
    {
        return vec3(m[0][3], m[1][3], m[2][3]);
    }

    float m[N][N];
};

// SquareMatrix Inline Methods
template <int N>
inline bool SquareMatrix<N>::is_identity() const
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                if (m[i][j] != 1)
                    return false;
            } else if (m[i][j] != 0)
                return false;
        }
    return true;
}

// SquareMatrix Inline Functions
template <int N>
CUDA_HOST_DEVICE inline SquareMatrix<N> operator*(float s, const SquareMatrix<N> &m)
{
    return m * s;
}

template <int N, typename T>
CUDA_HOST_DEVICE inline T operator*(const SquareMatrix<N> &m, const T &v)
{
    return mul<T>(m, v);
}

template <typename Tresult, int N, typename T>
CUDA_HOST_DEVICE inline Tresult mul(const SquareMatrix<N> &m, const T &v)
{
    Tresult result;
    for (int i = 0; i < N; ++i) {
        result[i] = 0;
        for (int j = 0; j < N; ++j)
            result[i] += m[i][j] * v[j];
    }
    return result;
}

template <int N>
CUDA_HOST_DEVICE float determinant(const SquareMatrix<N> &m);

template <>
CUDA_HOST_DEVICE inline float determinant(const SquareMatrix<3> &m)
{
    float minor12 = difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    float minor02 = difference_of_products(m[1][0], m[2][2], m[1][2], m[2][0]);
    float minor01 = difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
    return ::fma(m[0][2], minor01, difference_of_products(m[0][0], minor12, m[0][1], minor02));
}

template <int N>
CUDA_HOST_DEVICE inline SquareMatrix<N> transpose(const SquareMatrix<N> &m);
template <int N>
CUDA_HOST_DEVICE ksc::optional<SquareMatrix<N>> inverse(const SquareMatrix<N> &);

template <int N>
CUDA_HOST_DEVICE SquareMatrix<N> invert_or_exit(const SquareMatrix<N> &m)
{
    ksc::optional<SquareMatrix<N>> inv = inverse(m);
    KSC_ASSERT(inv.has_value());
    return *inv;
}

template <int N>
CUDA_HOST_DEVICE inline SquareMatrix<N> transpose(const SquareMatrix<N> &m)
{
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i][j] = m[j][i];
    return r;
}

template <>
CUDA_HOST_DEVICE inline ksc::optional<SquareMatrix<3>> inverse(const SquareMatrix<3> &m)
{
    float det = determinant(m);
    if (det == 0)
        return {};
    float invDet = 1 / det;

    SquareMatrix<3> r;

    r[0][0] = invDet * difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * difference_of_products(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * difference_of_products(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * difference_of_products(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * difference_of_products(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * difference_of_products(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * difference_of_products(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
}

template <>
CUDA_HOST_DEVICE inline SquareMatrix<4> operator*(const SquareMatrix<4> &m1, const SquareMatrix<4> &m2)
{
    SquareMatrix<4> r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r[i][j] = inner_product(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j], m1[i][3], m2[3][j]);
    return r;
}

template <>
CUDA_HOST_DEVICE inline SquareMatrix<3> operator*(const SquareMatrix<3> &m1, const SquareMatrix<3> &m2)
{
    SquareMatrix<3> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] = inner_product(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j]);
    return r;
}

template <int N>
CUDA_HOST_DEVICE inline SquareMatrix<N> operator*(const SquareMatrix<N> &m1, const SquareMatrix<N> &m2)
{
    SquareMatrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            r[i][j] = 0;
            for (int k = 0; k < N; ++k)
                r[i][j] = ::fma(m1[i][k], m2[k][j], r[i][j]);
        }
    return r;
}

template <int N>
CUDA_HOST_DEVICE inline SquareMatrix<N>::SquareMatrix(ksc::span<const float> t)
{
    KSC_ASSERT(N * N == t.size());
    for (int i = 0; i < N * N; ++i)
        m[i / N][i % N] = t[i];
}

template <int N>
CUDA_HOST_DEVICE SquareMatrix<N> operator*(const SquareMatrix<N> &m1, const SquareMatrix<N> &m2);

template <>
CUDA_HOST_DEVICE inline float determinant(const SquareMatrix<1> &m)
{
    return m[0][0];
}

template <>
CUDA_HOST_DEVICE inline float determinant(const SquareMatrix<2> &m)
{
    return difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);
}

template <>
CUDA_HOST_DEVICE inline float determinant(const SquareMatrix<4> &m)
{
    float s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    float s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    float s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    float s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    float s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    float s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    float c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    float c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    float c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    float c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    float c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    float c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    return (difference_of_products(s0, c5, s1, c4) + difference_of_products(s2, c3, -s3, c2) +
            difference_of_products(s5, c0, s4, c1));
}

template <int N>
CUDA_HOST_DEVICE inline float determinant(const SquareMatrix<N> &m)
{
    SquareMatrix<N - 1> sub;
    float det = 0;
    // Inefficient, but we don't currently use N>4 anyway..
    for (int i = 0; i < N; ++i) {
        // Sub-matrix without row 0 and column i
        for (int j = 0; j < N - 1; ++j)
            for (int k = 0; k < N - 1; ++k)
                sub[j][k] = m[j + 1][k < i ? k : k + 1];

        float sign = (i & 1) ? -1 : 1;
        det += sign * m[0][i] * determinant(sub);
    }
    return det;
}

template <>
CUDA_HOST_DEVICE inline ksc::optional<SquareMatrix<4>> inverse(const SquareMatrix<4> &m)
{
    // Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
    // (c) Google, Apache license.

    // For 4x4 do not compute the adjugate as the transpose of the cofactor
    // matrix, because this results in extra work. Several calculations can be
    // shared across the sub-determinants.
    //
    // This approach is explained in David Eberly's Geometric Tools book,
    // excerpted here:
    //   http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    float s0 = difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    float s1 = difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    float s2 = difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    float s3 = difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    float s4 = difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    float s5 = difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    float c0 = difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    float c1 = difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    float c2 = difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    float c3 = difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    float c4 = difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    float c5 = difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    float determinant = inner_product(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0)
        return {};
    float s = 1 / determinant;

    float inv[4][4] = {{s * inner_product(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                        s * inner_product(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                        s * inner_product(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                        s * inner_product(-m[2][1], s5, m[2][2], s4, -m[2][3], s3)},

                       {s * inner_product(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                        s * inner_product(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                        s * inner_product(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                        s * inner_product(m[2][0], s5, m[2][3], s1, -m[2][2], s2)},

                       {s * inner_product(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                        s * inner_product(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                        s * inner_product(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                        s * inner_product(-m[2][0], s4, m[2][1], s2, -m[2][3], s0)},

                       {s * inner_product(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                        s * inner_product(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                        s * inner_product(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                        s * inner_product(m[2][0], s3, m[2][2], s0, -m[2][1], s1)}};

    return SquareMatrix<4>(inv);
}

using mat2 = SquareMatrix<2>;
using mat3 = SquareMatrix<3>;
using mat4 = SquareMatrix<4>;

CUDA_HOST_DEVICE
inline mat4 make_affine(const mat3 &linear, const vec3 &translation)
{
    mat4 A;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            A.m[i][j] = linear.m[i][j];
    }
    A.m[0][3] = translation[0];
    A.m[1][3] = translation[1];
    A.m[2][3] = translation[2];

    A.m[3][0] = 0.0f;
    A.m[3][1] = 0.0f;
    A.m[3][2] = 0.0f;
    A.m[3][3] = 1.0f;
    return A;
}

CUDA_HOST_DEVICE
inline mat4 affine_inverse(const mat4 &A)
{
    mat3 minor_inv = invert_or_exit(A.minor());
    vec3 trans_inv = -minor_inv * A.translation();
    return make_affine(minor_inv, trans_inv);
}

///////////////////////

// Quaternion Definition
struct Quaternion
{
    CUDA_HOST_DEVICE
    static Quaternion identity() { return Quaternion(vec3(0.0f), 1.0f); }

    // Quaternion Public Methods
    Quaternion() = default;

    CUDA_HOST_DEVICE
    Quaternion(Vector3<float> v, float w) : v(v), w(w) {}

    CUDA_HOST_DEVICE
    explicit Quaternion(const mat3 &a)
    {
        // Assume R is a proper rotation matrix.
        float trace = a[0][0] + a[1][1] + a[2][2];
        if (trace > 0) {
            float s = 0.5f / sqrtf(trace + 1.0f);
            w = 0.25f / s;
            v.x = (a[2][1] - a[1][2]) * s;
            v.y = (a[0][2] - a[2][0]) * s;
            v.z = (a[1][0] - a[0][1]) * s;
        } else {
            if (a[0][0] > a[1][1] && a[0][0] > a[2][2]) {
                float s = 2.0f * sqrtf(1.0f + a[0][0] - a[1][1] - a[2][2]);
                w = (a[2][1] - a[1][2]) / s;
                v.x = 0.25f * s;
                v.y = (a[0][1] + a[1][0]) / s;
                v.z = (a[0][2] + a[2][0]) / s;
            } else if (a[1][1] > a[2][2]) {
                float s = 2.0f * sqrtf(1.0f + a[1][1] - a[0][0] - a[2][2]);
                w = (a[0][2] - a[2][0]) / s;
                v.x = (a[0][1] + a[1][0]) / s;
                v.y = 0.25f * s;
                v.z = (a[1][2] + a[2][1]) / s;
            } else {
                float s = 2.0f * sqrtf(1.0f + a[2][2] - a[0][0] - a[1][1]);
                w = (a[1][0] - a[0][1]) / s;
                v.x = (a[0][2] + a[2][0]) / s;
                v.y = (a[1][2] + a[2][1]) / s;
                v.z = 0.25f * s;
            }
        }
    }

    CUDA_HOST_DEVICE
    Quaternion &operator+=(Quaternion q)
    {
        v += q.v;
        w += q.w;
        return *this;
    }

    CUDA_HOST_DEVICE
    Quaternion operator+(Quaternion q) const { return {v + q.v, w + q.w}; }
    CUDA_HOST_DEVICE
    Quaternion &operator-=(Quaternion q)
    {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    CUDA_HOST_DEVICE
    Quaternion operator-() const { return {-v, -w}; }
    CUDA_HOST_DEVICE
    Quaternion operator-(Quaternion q) const { return {v - q.v, w - q.w}; }
    CUDA_HOST_DEVICE
    Quaternion &operator*=(float f)
    {
        v *= f;
        w *= f;
        return *this;
    }
    CUDA_HOST_DEVICE
    Quaternion operator*(float f) const { return {v * f, w * f}; }
    CUDA_HOST_DEVICE
    Quaternion &operator/=(float f)
    {
        v /= f;
        w /= f;
        return *this;
    }
    CUDA_HOST_DEVICE
    Quaternion operator/(float f) const { return {v / f, w / f}; }

    CUDA_HOST_DEVICE
    mat3 to_matrix() const
    {
        float x2 = v.x * v.x;
        float y2 = v.y * v.y;
        float z2 = v.z * v.z;
        float xy = v.x * v.y;
        float xz = v.x * v.z;
        float yz = v.y * v.z;
        float wx = w * v.x;
        float wy = w * v.y;
        float wz = w * v.z;
        // clang-format off
        return mat3(
            1 - 2 * y2 - 2 * z2,    2 * xy - 2 * wz,        2 * xz + 2 * wy,
            2 * xy + 2 * wz,        1 - 2 * x2 - 2 * z2,    2 * yz - 2 * wx,
            2 * xz - 2 * wy,        2 * yz + 2 * wx,        1 - 2 * x2 - 2 * y2);
        // clang-format on
    }

    // Quaternion Public Members
    Vector3<float> v;
    float w = 1;
};

// Quaternion Inline Functions
CUDA_HOST_DEVICE
inline Quaternion operator*(float f, Quaternion q) { return q * f; }

CUDA_HOST_DEVICE inline float dot(Quaternion q1, Quaternion q2) { return dot(q1.v, q2.v) + q1.w * q2.w; }

CUDA_HOST_DEVICE inline float length(Quaternion q) { return std::sqrt(dot(q, q)); }
CUDA_HOST_DEVICE inline Quaternion normalize(Quaternion q) { return q / length(q); }

CUDA_HOST_DEVICE inline float angle_between(Quaternion q1, Quaternion q2)
{
    if (dot(q1, q2) < 0)
        return pi - 2 * safe_asin(length(q1 + q2) / 2);
    else
        return 2 * safe_asin(length(q2 - q1) / 2);
}

// http://www.plunk.org/~hatch/rightway.html
CUDA_HOST_DEVICE inline Quaternion slerp(float t, Quaternion q1, Quaternion q2)
{
    float theta = angle_between(q1, q2);
    float sinThetaOverTheta = sinx_over_x(theta);
    return q1 * (1 - t) * sinx_over_x((1 - t) * theta) / sinThetaOverTheta +
           q2 * t * sinx_over_x(t * theta) / sinThetaOverTheta;
}

using quat = Quaternion;

///////////////////////

// Concentric mapping
CUDA_HOST_DEVICE inline vec3 concentric_square_to_hemisphere(vec2 u)
{
    // Map uniform random numbers to $[-1,1]^2$
    u = 2.0f * u - vec2(1.0f);

    // Handle degeneracy at the origin
    if (u.x == 0.0f && u.y == 0.0f) {
        return vec3(0.0f, 0.0f, 1.0f);
    }

    // Apply concentric mapping to point
    float phi, r;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        phi = (pi * 0.25f) * (u.y / u.x);
    } else {
        r = u.y;
        phi = (pi * 0.5f) - (pi * 0.25f) * (u.x / u.y);
    }

    float r2 = sqr(r);
    float sin_theta = r * sqrt(2.0f - sqr(r));

    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;
    float z = 1.0f - r2;
    return vec3(x, y, z);
}

// Inverse concentric mapping
CUDA_HOST_DEVICE inline vec2 concentric_hemisphere_to_square(const vec3 &w)
{
    float r = safe_sqrt(1.0f - w.z);
    float phi = atan2(w.y, w.x);
    if (phi < -0.25f * pi) {
        phi += 2.0f * pi;
    }
    float x, y;
    if (phi < 0.25f * pi) {
        x = r;
        y = 4.0f / pi * r * phi;
    } else if (phi < 0.75f * pi) {
        x = -4.0f / pi * r * (phi - 0.5f * pi);
        y = r;
    } else if (phi < 1.25f * pi) {
        x = -r;
        y = -4.0f / pi * r * (phi - pi);
    } else {
        x = 4.0f / pi * r * (phi - 1.5f * pi);
        y = -r;
    }
    x = x * 0.5f + 0.5f;
    y = y * 0.5f + 0.5f;
    return vec2(x, y);
}

CUDA_HOST_DEVICE
vec3 equal_area_square_to_sphere(vec2 p);
CUDA_HOST_DEVICE
vec2 equal_area_sphere_to_square(vec3 d);
CUDA_HOST_DEVICE
vec2 wrap_equal_area_square(vec2 uv);
CUDA_HOST_DEVICE
void equal_area_bilerp(vec2 uv, int N, array<vec2i, 4> &idx, array<float, 4> &weight);

CUDA_HOST_DEVICE
inline float luminance(const color3 &rgb)
{
    constexpr float lum_weight[3] = {0.212671f, 0.715160f, 0.072169f};
    return lum_weight[0] * rgb[0] + lum_weight[1] * rgb[1] + lum_weight[2] * rgb[2];
}

CUDA_HOST_DEVICE
inline color3 srgb_to_linear(color3 c) { return color3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z)); }

CUDA_HOST_DEVICE
inline color3 linear_to_srgb(color3 c) { return color3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z)); }

CUDA_HOST_DEVICE
inline vec2 demux_float(float f)
{
    // ASSERT(f >= 0 && f < 1);
    uint64_t v = f * (1ull << 32);
    // ASSERT(v < 0x100000000);
    uint32_t bits[2] = {compact_1_by_1(v), compact_1_by_1(v >> 1)};
    return {bits[0] / float(1 << 16), bits[1] / float(1 << 16)};
}

CUDA_HOST_DEVICE inline float spherical_theta(vec3 v) { return safe_acos(v.z); }

CUDA_HOST_DEVICE inline float spherical_phi(vec3 v)
{
    float p = atan2(v.y, v.x);
    return (p < 0) ? (p + 2.0f * pi) : p;
}

CUDA_HOST_DEVICE inline vec2 to_spherical(vec3 xyz) { return vec2(spherical_phi(xyz), spherical_theta(xyz)); }

CUDA_HOST_DEVICE inline vec3 to_cartesian(vec2 sph)
{
    float phi = sph.x;
    float theta = sph.y;
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    return vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

} // namespace ksc