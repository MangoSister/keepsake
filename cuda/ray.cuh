#pragma once

#include "transform.cuh"
#include "vecmath.cuh"

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

} // namespace ksc