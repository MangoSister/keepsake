#pragma once

#include "aabb.h"
#include "assertion.h"
#include "ray.h"
#include <array>
#include <utility>

#include <embree3/rtcore.h>

namespace ks
{

struct EmbreeDevice
{
    explicit EmbreeDevice(const std::string &device_config = "");
    ~EmbreeDevice();

    operator RTCDevice() const { return device; }

    RTCDevice device;
};

typedef void (*IntersectContextFilter)(const RTCFilterFunctionNArguments *args, void *payload);

inline void intersect_context_filter_toplevel(const RTCFilterFunctionNArguments *args);

struct IntersectContext
{
    IntersectContext()
    {
        rtcInitIntersectContext(&context);
        context.filter = intersect_context_filter_toplevel;
    }

    void add_filter(IntersectContextFilter fn, void *payload) const
    {
        ASSERT(n_filters < max_n_filters);
        filters[n_filters++] = {fn, payload};
    }

    RTCIntersectContext context;
    //
    static constexpr uint32_t max_n_filters = 16;
    mutable std::array<std::pair<IntersectContextFilter, void *>, max_n_filters> filters;
    mutable uint32_t n_filters = 0;
};

inline void intersect_context_filter_toplevel(const RTCFilterFunctionNArguments *args)
{
    const IntersectContext *ctx = reinterpret_cast<const IntersectContext *>(args->context);
    for (uint32_t i = 0; i < ctx->n_filters; ++i) {
        auto [fn, payload] = ctx->filters[i];
        fn(args, payload);
    }
}

inline RTCRay spawn_ray(const vec3 &origin, const vec3 &dir, float tnear, float tfar)
{
    RTCRay ray;
    ray.org_x = origin.x();
    ray.org_y = origin.y();
    ray.org_z = origin.z();
    ray.dir_x = dir.x();
    ray.dir_y = dir.y();
    ray.dir_z = dir.z();
    ray.tnear = tnear;
    ray.tfar = tfar;
    ray.time = uint_as_float(~0);
    ray.mask = ~0;
    ray.id = ~0;
    ray.flags = 0;
    return ray;
}

template <OffsetType type>
inline RTCRay spawn_rtcray(vec3 origin, const vec3 &dir, const vec3 &ng, float tnear, float tfar)
{
    if constexpr (type == OffsetType::Shadow) {
        origin = offset_ray(origin, ng);
    } else {
        origin = offset_ray(origin, dir.dot(ng) > 0.0f ? ng : -ng);
    }
    return spawn_ray(origin, dir, tnear, tfar);
}

inline RTCRayHit spawn_rtcrayhit(const vec3 &origin, const vec3 &dir, float tnear, float tfar)
{
    RTCRayHit rayhit;
    rayhit.ray = spawn_ray(origin, dir, tnear, tfar);
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    return rayhit;
}

template <OffsetType type>
inline RTCRayHit spawn_rtcrayhit(vec3 origin, const vec3 &dir, const vec3 &ng, float tnear, float tfar)
{
    RTCRayHit rayhit;
    rayhit.ray = spawn_ray<type>(origin, dir, ng, tnear, tfar);
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    return rayhit;
}

inline RTCRay to_rtcray(const Ray &ray) { return spawn_ray(ray.origin, ray.dir, ray.tmin, ray.tmax); }

inline Ray from_rtcray(const RTCRay &ray)
{
    vec3 o(ray.org_x, ray.org_y, ray.org_z);
    vec3 d(ray.dir_x, ray.dir_y, ray.dir_z);
    return Ray(o, d, ray.tnear, ray.tfar);
}

inline RTCRayHit to_rtcrayhit(const Ray &ray)
{
    RTCRayHit rayhit;
    rayhit.ray = spawn_ray(ray.origin, ray.dir, ray.tmin, ray.tmax);
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    return rayhit;
}

inline Ray from_rayhit(const RTCRayHit &rayhit) { return from_rtcray(rayhit.ray); }

inline bool intersect1(RTCScene scene, RTCRayHit &rayhit)
{
    RTCIntersectContext ctx;
    rtcInitIntersectContext(&ctx);
    ASSERT(rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID);
    rtcIntersect1(scene, &ctx, &rayhit);
    return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

inline bool intersect1(RTCScene scene, const IntersectContext &ctx, RTCRayHit &rayhit)
{
    ASSERT(rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID);
    rtcIntersect1(scene, (RTCIntersectContext *)&ctx, &rayhit);
    return rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID;
}

inline bool occlude1(RTCScene scene, RTCRay &ray)
{
    RTCIntersectContext ctx;
    rtcInitIntersectContext(&ctx);
    ASSERT(ray.tfar != -inf);
    rtcOccluded1(scene, &ctx, &ray);
    return ray.tfar == -inf;
}

inline bool occlude1(RTCScene scene, const IntersectContext &ctx, RTCRay &ray)
{
    ASSERT(ray.tfar != -inf);
    rtcOccluded1(scene, (RTCIntersectContext *)&ctx, &ray);
    return ray.tfar == -inf;
}

inline AABB3 scene_bound(RTCScene scene)
{
    ASSERT(scene);
    RTCBounds b;
    rtcGetSceneBounds(scene, &b);
    return AABB3(vec3(b.lower_x, b.lower_y, b.lower_z), vec3(b.upper_x, b.upper_y, b.upper_z));
}

inline void filter_backface_culling(const RTCFilterFunctionNArguments *args)
{
    uint32_t N = args->N;
    int *valid = args->valid;
    RTCRayN *ray = args->ray;
    RTCHitN *hit = args->hit;
    for (uint32_t i = 0; i < N; ++i) {
        if (valid[i] != 0) {
            vec3 rd(RTCRayN_dir_x(ray, N, i), RTCRayN_dir_y(ray, N, i), RTCRayN_dir_z(ray, N, i));
            vec3 ng(RTCHitN_Ng_x(hit, N, i), RTCHitN_Ng_y(hit, N, i), RTCHitN_Ng_z(hit, N, i));
            if (rd.dot(ng) >= 0.0f) {
                valid[i] = 0;
            }
        }
    }
}

inline void filter_local_geometry(const RTCFilterFunctionNArguments *args, void *payload)
{
    uint32_t geom_id = *(uint32_t *)payload;

    uint32_t N = args->N;
    int *valid = args->valid;
    RTCRayN *ray = args->ray;
    RTCHitN *hit = args->hit;
    for (uint32_t i = 0; i < N; ++i) {
        if (valid[i] != 0) {
            if (geom_id != RTCHitN_geomID(hit, N, i))
                valid[i] = 0;
        }
    }
}

inline void filter_exclude_local_geometry(const RTCFilterFunctionNArguments *args, void *payload)
{
    uint32_t geom_id = *(uint32_t *)payload;

    uint32_t N = args->N;
    int *valid = args->valid;
    RTCRayN *ray = args->ray;
    RTCHitN *hit = args->hit;
    for (uint32_t i = 0; i < N; ++i) {
        if (valid[i] != 0) {
            if (geom_id == RTCHitN_geomID(hit, N, i))
                valid[i] = 0;
        }
    }
}

} // namespace ks