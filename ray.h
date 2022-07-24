#pragma once

#include "maths.h"

struct Ray
{
    Ray() = default;
    Ray(const vec3 &o, const vec3 &d, float tmin, float tmax) : origin(o), dir(d), tmin(tmin), tmax(tmax) {}

    vec3 origin;
    float tmin;
    vec3 dir;
    float tmax;

    vec3 operator()(float t) const { return origin + t * dir; }
};

inline Ray transform_ray(const mat4 &m, const Ray &r)
{
    Ray r_out;
    r_out.origin = transform_point(m, r.origin);
    r_out.dir = transform_dir(m, r.dir);
    r_out.tmin = r.tmin;
    r_out.tmax = r.tmax;
    return r_out;
}

struct Ray2
{
    Ray2() = default;
    Ray2(const vec2 &o, const vec2 &d, float tmin, float tmax) : origin(o), dir(d), tmin(tmin), tmax(tmax) {}

    vec2 origin;
    vec2 dir;
    float tmin;
    float tmax;

    vec2 operator()(float t) const { return origin + t * dir; }
};

struct Intersection
{
    vec3 normal;
    float thit;
};