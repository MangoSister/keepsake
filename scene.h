#pragma once
#include "config.h"
#include "embree_util.h"
#include "geometry.h"
#include "light.h"
#include <memory>
#include <optional>

namespace ks
{

struct BSDF;
struct BSSRDF;
struct Material;

struct SceneHit
{
    Intersection it;
    const Material *material = nullptr;
    uint32_t inst_id = 0;
    uint32_t subscene_id = 0;
    uint32_t geom_id = 0; //
    uint32_t prim_id = 0;
};

struct SubScene
{
    ~SubScene();
    SubScene() = default;
    SubScene(SubScene &&other);
    SubScene &operator=(SubScene &&other);

    void create_rtc_scene(const EmbreeDevice &device);
    AABB3 bound() const;

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

    void add_subscene(SubScene &&subscene);
    void add_instance(const EmbreeDevice &device, uint32_t subscene_id, const Transform &transform,
                      bool local_intersector = false);
    void create_rtc_scene(const EmbreeDevice &device);
    AABB3 bound() const;
    bool intersect1(const Ray &ray, SceneHit &hit, const IntersectContext &ctx = IntersectContext()) const;
    bool occlude1(const Ray &ray, const IntersectContext &ctx = IntersectContext()) const;
    bool intersect1_local(uint32_t inst_id, const Ray &ray, SceneHit &hit,
                          const IntersectContext &ctx = IntersectContext()) const;
    bool occlude1_local(uint32_t inst_id, const Ray &ray, const IntersectContext &ctx = IntersectContext()) const;

    // Convenience methods
    const Transform &get_instance_transform(uint32_t inst_id) const;
    const MeshGeometry &get_prototype_mesh_geometry(uint32_t subscene_id, uint32_t geom_id) const;
    const Material &get_prototype_material(uint32_t subscene_id, uint32_t geom_id) const;
    bool are_material_assigned() const;

    void build_mesh_lights();

    std::vector<std::unique_ptr<SubScene>> subscenes;
    std::vector<SubSceneInstance> instances;

    // Follow Blender and create a dedicated "local" intersector for each subsurface object or object with similar
    // needs: https://developer.blender.org/docs/features/cycles/bvh/
    // (inst_id, local_scene)
    std::unordered_map<uint32_t, RTCScene> per_instance_intersectors;

    // TODO: instanced mesh lights
    std::vector<std::unique_ptr<MeshLightShared>> mesh_lights;

    RTCScene rtcscene = nullptr;
};

struct LocalGeometry
{
    const Scene *scene = nullptr;
    uint32_t geom_id = 0;
    uint32_t inst_id = (uint32_t)(~0);

    bool intersect1(const Ray &ray, SceneHit &hit) const;
};

Scene create_scene(const ConfigArgs &args, const EmbreeDevice &device);

} // namespace ks