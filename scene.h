#pragma once
#include "embree_util.h"
#include "geometry.h"
#include <memory>

struct BSDF;
struct BSSRDF;
struct Material;

struct SceneHit
{
    Intersection it;
    const Material *material = nullptr;
    uint32_t geom_id = 0; //
};

struct Scene
{
    ~Scene();
    Scene() = default;
    Scene(Scene &&other);

    void create_rtc_scene(const EmbreeDevice &device);
    AABB3 bound() const;
    bool intersect1(const Ray &ray, SceneHit &hit, const IntersectContext &ctx = IntersectContext()) const;
    bool occlude1(const Ray &ray, const IntersectContext &ctx = IntersectContext()) const;

    std::vector<std::unique_ptr<Geometry>> geometries;
    std::vector<const Material *> materials;

    RTCScene rtcscene = nullptr;
};