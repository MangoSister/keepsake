#pragma once
#include "embree_util.h"
#include "geometry.h"
#include <memory>

namespace ks
{

struct BSDF;
struct BSSRDF;
struct Material;

struct SceneHit
{
    Intersection it;
    const Material *material = nullptr;
    uint32_t subscene_id = 0;
    uint32_t geom_id = 0; //
};

struct SubScene
{
    ~SubScene();
    SubScene() = default;
    SubScene(SubScene &&other);
    SubScene &operator=(SubScene &&other);

    void create_rtc_scene(const EmbreeDevice &device);

    std::vector<std::unique_ptr<Geometry>> geometries;
    std::vector<const Material *> materials;

    RTCScene rtcscene = nullptr;
};

struct SubSceneInstance
{
    uint32_t prototype = 0;
    Transform transform;
    RTCGeometry rtgeom_inst = nullptr;
};

struct Scene
{
    ~Scene();
    Scene() = default;
    Scene(Scene &&other);
    Scene &operator=(Scene &&other);

    void add_subscene(SubScene && subscene);
    void add_instance(const EmbreeDevice &device, uint32_t subscene_id, const Transform &transform);
    void create_rtc_scene(const EmbreeDevice &device);
    AABB3 bound() const;
    bool intersect1(const Ray &ray, SceneHit &hit, const IntersectContext &ctx = IntersectContext()) const;
    bool occlude1(const Ray &ray, const IntersectContext &ctx = IntersectContext()) const;

    // Convenience methods
    const Transform &get_instance_transform(uint32_t inst_id) const;
    const MeshGeometry &get_prototype_mesh_geometry(uint32_t subscene_id, uint32_t geom_id) const;
    const Material &get_prototype_material(uint32_t subscene_id, uint32_t geom_id) const;
    bool are_material_assigned() const;

    std::vector<std::unique_ptr<SubScene>> subscenes;
    std::vector<SubSceneInstance> instances;

    RTCScene rtcscene = nullptr;
};

struct LocalGeometry
{
    const Scene *scene = nullptr;
    uint32_t geom_id = 0;

    bool intersect1(const Ray &ray, SceneHit &hit) const;
};

} // namespace ks