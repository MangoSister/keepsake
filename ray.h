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

    void *extra = nullptr;

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

inline Ray transform_ray(const Transform &t, const Ray &r)
{
    Ray r_out;
    r_out.origin = t.point(r.origin);
    r_out.dir = t.direction(r.dir);
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

// Normal points outward for rays exiting the surface, else is flipped.
inline vec3 offset_ray(const vec3 &p, const vec3 &n)
{
    constexpr float origin = 1.0f / 32.0f;
    constexpr float float_scale = 1.0f / 65536.0f;
    constexpr float int_scale = 256.0f;

    vec3i of_i(int_scale * n.x(), int_scale * n.y(), int_scale * n.z());
    vec3 p_i(int_as_float(float_as_int(p.x()) + ((p.x() < 0) ? -of_i.x() : of_i.x())),
             int_as_float(float_as_int(p.y()) + ((p.y() < 0) ? -of_i.y() : of_i.y())),
             int_as_float(float_as_int(p.z()) + ((p.z() < 0) ? -of_i.z() : of_i.z())));
    return vec3(fabsf(p.x()) < origin ? p.x() + float_scale * n.x() : p_i.x(),
                fabsf(p.y()) < origin ? p.y() + float_scale * n.y() : p_i.y(),
                fabsf(p.z()) < origin ? p.z() + float_scale * n.z() : p_i.z());
}

enum class OffsetType
{
    Shadow,
    NextBounce,
};

template <OffsetType type>
inline Ray spawn_ray(vec3 origin, const vec3 &dir, const vec3 &ng, float tnear, float tfar)
{
    if constexpr (type == OffsetType::Shadow) {
        origin = offset_ray(origin, ng);
    } else {
        origin = offset_ray(origin, dir.dot(ng) > 0.0f ? ng : -ng);
    }
    return Ray(origin, dir, tnear, tfar);
}

struct Intersection
{
    float thit;
    vec3 p;
    Frame frame;
    void *extra = nullptr;
};

inline Intersection transform_it(const mat4 &m, const Intersection &it)
{
    Intersection it_out;
    it_out.thit = it.thit;
    it_out.p = transform_point(m, it.p);
    it_out.frame = Frame(transform_dir(m, it.frame.t), transform_dir(m, it.frame.b));
    return it_out;
}

inline Intersection transform_it(const Transform &t, const Intersection &it)
{
    Intersection it_out;
    it_out.thit = it.thit;
    it_out.p = t.point(it.p);
    it_out.frame = Frame(t.direction(it.frame.t), t.direction(it.frame.b));
    return it_out;
}