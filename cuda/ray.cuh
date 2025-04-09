#pragma once

#include "transform.cuh"
#include "vecmath.cuh"
#include <cuda/std/bit>

namespace ksc
{

struct Ray
{
    Ray() = default;
    CUDA_HOST_DEVICE
    Ray(const vec3 &o, const vec3 &d, float tmin, float tmax) : origin(o), dir(d), tmin(tmin), tmax(tmax) {}

    CUDA_HOST_DEVICE
    bool has_ray_diffs() const { return rx_dir != vec3::zero(); };

    CUDA_HOST_DEVICE
    vec3 operator()(float t) const { return origin + t * dir; }

    vec3 origin = vec3::zero();
    float tmin = 0.0f;
    vec3 dir = vec3::zero();
    float tmax = 0.0f;

    vec3 rx_origin = vec3::zero();
    vec3 rx_dir = vec3::zero();
    vec3 ry_origin = vec3::zero();
    vec3 ry_dir = vec3::zero();
};

CUDA_HOST_DEVICE inline Ray transform_ray(const Transform &t, const Ray &r)
{
    Ray r_out;
    r_out.origin = t.point(r.origin);
    r_out.dir = t.direction(r.dir);
    r_out.tmin = r.tmin;
    r_out.tmax = r.tmax;
    return r_out;
}

// Normal points outward for rays exiting the surface, else is flipped.
CUDA_HOST_DEVICE inline vec3 offset_ray(const vec3 &p, const vec3 &n)
{
    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;

    vec3i of_i(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    // using cuda::std::bit_cast

    vec3 p_i(bit_cast<float, int>(bit_cast<int, float>(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
             bit_cast<float, int>(bit_cast<int, float>(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
             bit_cast<float, int>(bit_cast<int, float>(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return vec3(fabsf(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                fabsf(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                fabsf(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

CUDA_HOST_DEVICE inline Ray spawn_ray(vec3 origin, const vec3 &dir, const vec3 &ng, float tnear, float tfar)
{
    origin = offset_ray(origin, dot(dir, ng) > 0.0f ? ng : -ng);
    return Ray(origin, dir, tnear, tfar);
}

} // namespace ksc