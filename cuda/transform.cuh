#pragma once
#include "vecmath.cuh"

namespace ksc
{

struct Transform
{
    Transform() = default;
    CUDA_HOST_DEVICE
    explicit Transform(const mat4 &m) : m(m), inv(ksc::inverse(m)) {}
    CUDA_HOST_DEVICE
    Transform(const mat4 &m, const mat4 &inv) : m(m), inv(inv) {}

    CUDA_HOST_DEVICE
    Transform inverse() const { return {inv, m}; }

    CUDA_HOST_DEVICE
    Transform operator*(const Transform &t2) const { return Transform(m * t2.m, t2.inv * inv); }

    CUDA_HOST_DEVICE
    vec3 point(const vec3 &p) const
    {
        float x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        float y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        float z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        return vec3(x, y, z);
    }

    CUDA_HOST_DEVICE
    vec3 inv_point(const vec3 &p) const
    {
        float x = inv[0][0] * p.x + inv[0][1] * p.y + inv[0][2] * p.z + inv[0][3];
        float y = inv[1][0] * p.x + inv[1][1] * p.y + inv[1][2] * p.z + inv[1][3];
        float z = inv[2][0] * p.x + inv[2][1] * p.y + inv[2][2] * p.z + inv[2][3];
        return vec3(x, y, z);
    }

    // vec4 hpoint(const vec3 &p) const { return m * p.homogeneous(); }

    CUDA_HOST_DEVICE
    vec3 point_hdiv(const vec3 &p) const
    {
        float w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        return point(p) / w;
    }

    CUDA_HOST_DEVICE
    vec3 inv_point_hdiv(const vec3 &p) const
    {
        float w = inv[3][0] * p.x + inv[3][1] * p.y + inv[3][2] * p.z + inv[3][3];
        return inv_point(p) / w;
    }

    CUDA_HOST_DEVICE
    vec3 direction(const vec3 &v) const
    {
        float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
        float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
        float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
        return vec3(x, y, z);
    }

    CUDA_HOST_DEVICE
    vec3 inv_direction(const vec3 &v) const
    {
        float x = inv[0][0] * v.x + inv[0][1] * v.y + inv[0][2] * v.z;
        float y = inv[1][0] * v.x + inv[1][1] * v.y + inv[1][2] * v.z;
        float z = inv[2][0] * v.x + inv[2][1] * v.y + inv[2][2] * v.z;
        return vec3(x, y, z);
    }

    CUDA_HOST_DEVICE
    vec3 normal(const vec3 &n) const
    {
        float x = inv[0][0] * n.x + inv[1][0] * n.y + inv[2][0] * n.z;
        float y = inv[0][1] * n.x + inv[1][1] * n.y + inv[2][1] * n.z;
        float z = inv[0][2] * n.x + inv[1][2] * n.y + inv[2][2] * n.z;
        return vec3(x, y, z);
    }

    CUDA_HOST_DEVICE
    vec3 inv_normal(const vec3 &n) const
    {
        float x = m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z;
        float y = m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z;
        float z = m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z;
        return vec3(x, y, z);
    }

    mat4 m = mat4::identity();
    mat4 inv = mat4::identity();
};

CUDA_HOST_DEVICE
inline Transform make_rotate_x(float degree)
{
    float sinTheta = sin(to_radian(degree));
    float cosTheta = cos(to_radian(degree));
    mat4 m(1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1);
    return Transform(m, transpose(m));
}

CUDA_HOST_DEVICE
inline Transform make_rotate_y(float degree)
{
    float sinTheta = sin(to_radian(degree));
    float cosTheta = cos(to_radian(degree));
    mat4 m(cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0, 0, 0, 1);
    return Transform(m, transpose(m));
}

CUDA_HOST_DEVICE
inline Transform make_rotate_z(float degree)
{
    float sinTheta = sin(to_radian(degree));
    float cosTheta = cos(to_radian(degree));
    mat4 m(cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return Transform(m, transpose(m));
}

CUDA_HOST_DEVICE
inline Transform make_rotate(float degree, vec3 axis)
{
    float sinTheta = std::sin(to_radian(degree));
    float cosTheta = std::cos(to_radian(degree));
    vec3 a = normalize(axis);
    mat4 m;
    // Compute rotation of first basis vector
    m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m[0][3] = 0;

    // Compute rotations of second and third basis vectors
    m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m[1][3] = 0;

    m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m[2][3] = 0;

    return Transform(m, transpose(m));
}

} // namespace ksc