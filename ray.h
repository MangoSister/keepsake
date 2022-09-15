#pragma once

#include "maths.h"

struct Ray
{
    Ray() = default;
    Ray(const vec3 &o, const vec3 &d, float tmin, float tmax) : origin(o), dir(d), tmin(tmin), tmax(tmax) {}

    bool has_ray_diffs() const { return !rx_dir.isZero(); };
    vec3 operator()(float t) const { return origin + t * dir; }

    vec3 origin = vec3::Zero();
    float tmin = 0.0f;
    vec3 dir = vec3::Zero();
    float tmax = 0.0f;

    vec3 rx_origin = vec3::Zero();
    vec3 rx_dir = vec3::Zero();
    vec3 ry_origin = vec3::Zero();
    vec3 ry_dir = vec3::Zero();

    void *extra = nullptr;
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
    void compute_partials(const Ray &ray);

    // Attempts to alleviate the usual shading normal / normal map problems
    // by forcing the vector to stay in the same hemisphere before/after transform.
    vec3 sh_vector_to_local(const vec3 &world) const
    {
        bool hw = frame.n.dot(world) >= 0.0f;
        vec3 local = sh_frame.to_local(world);
        bool hl = local.z() >= 0.0f;
        if (hw != hl) {
            local.z() *= -1.0f;
        }
        return local;
    }

    vec3 sh_vector_to_world(const vec3 &local) const
    {
        bool hl = local.z() >= 0.0f;
        vec3 world = sh_frame.to_world(local);
        bool hw = frame.n.dot(world) >= 0.0f;
        if (hl != hw) {
            world -= 2.0f * frame.n.dot(world) * frame.n;
        }
        return world;
    }

    float thit = 0.0f;
    vec3 p = vec3::Zero();
    // NOTE: dpdu and dpdv are not necessarily orthogonal
    vec3 dpdu = vec3::Zero(), dpdv = vec3::Zero();
    Frame frame;
    Frame sh_frame;

    // uv and partials
    vec2 uv = vec2::Zero();
    vec3 dpdx = vec3::Zero(), dpdy = vec3::Zero();
    float dudx = 0.0f, dvdx = 0.0f;
    float dudy = 0.0f, dvdy = 0.0f;

    void *extra = nullptr;
};

inline Intersection transform_it(const mat4 &m, const Intersection &it)
{
    Intersection it_out;
    it_out.thit = it.thit;
    it_out.p = transform_point(m, it.p);
    it_out.dpdu = transform_dir(m, it.dpdu);
    it_out.dpdv = transform_dir(m, it.dpdv);
    it_out.frame = transform_frame(m, it.frame);
    it_out.sh_frame = transform_frame(m, it.sh_frame);
    it_out.uv = it.uv;
    it_out.dpdx = transform_dir(m, it.dpdx);
    it_out.dpdy = transform_dir(m, it.dpdy);
    it_out.dudx = it.dudx;
    it_out.dvdx = it.dvdx;
    it_out.dudy = it.dudy;
    it_out.dvdy = it.dvdy;

    return it_out;
}

inline Intersection transform_it(const Transform &t, const Intersection &it)
{
    Intersection it_out;
    it_out.thit = it.thit;
    it_out.p = t.point(it.p);
    it_out.dpdu = t.direction(it.dpdu);
    it_out.dpdv = t.direction(it.dpdv);
    it_out.frame = t.frame(it.frame);
    it_out.sh_frame = t.frame(it.sh_frame);
    it_out.uv = it.uv;
    it_out.dpdx = t.direction(it.dpdx);
    it_out.dpdy = t.direction(it.dpdy);
    it_out.dudx = it.dudx;
    it_out.dvdx = it.dvdx;
    it_out.dudy = it.dudy;
    it_out.dvdy = it.dvdy;

    return it_out;
}