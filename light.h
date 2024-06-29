#pragma once
#include "aabb.h"
#include "barray.h"
#include "config.h"
#include "distrib.h"
#include "geometry.h"
#include "maths.h"
#include "shader_field.h"
#include <filesystem>
namespace fs = std::filesystem;

namespace ks
{

//-----------------------------------------------------------------------------
// [Different types of lights]
//-----------------------------------------------------------------------------
// TODO: area lights???

struct Light
{
    virtual ~Light() = default;
    virtual bool delta_position() const = 0;
    virtual bool delta_direction() const = 0;
    bool delta() const { return delta_direction() || delta_position(); };

    // NOTE: the point passed in is the shading point
    // NOTE: return throughput weight: (L / pdf)
    virtual color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const = 0;
    virtual float pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const = 0;
    virtual color3 power(const AABB3 &scene_bound) const = 0;
};

struct SkyLight : public Light
{
    SkyLight(const fs::path &path, const Transform &l2w, bool transform_y_up, float strength = 1.0f);
    // Shortcut for ambient light.
    explicit SkyLight(const color3 &ambient);
    bool delta_position() const { return false; };
    bool delta_direction() const { return false; };

    // NOTE: the shading point is ignored for skylight
    color3 eval(const vec3 &p_shade, const vec3 &wi) const;
    // NOTE: return throughput weight: (L / pdf)
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const;
    color3 power(const AABB3 &scene_bound) const;

    DistribTable2D distrib;
    BlockedArray<color3> map;
    Transform l2w;
    bool transform_y_up = true;
    float strength = 1.0f;
};

struct DirectionalLight : public Light
{
    DirectionalLight() = default;
    DirectionalLight(const color3 &L, const vec3 &dir) : L(L), dir(dir){};

    bool delta_position() const { return false; };
    bool delta_direction() const { return true; };
    // NOTE: return throughput weight: (L / pdf)
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const;
    color3 power(const AABB3 &scene_bound) const;

    color3 L;
    vec3 dir;
};

struct PointLight : public Light
{
    PointLight() = default;
    PointLight(const color3 &I, const vec3 &pos) : I(I), pos(pos){};

    bool delta_position() const { return true; };
    bool delta_direction() const { return false; };
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const;
    color3 power(const AABB3 &scene_bound) const;

    color3 I;
    vec3 pos;
};

struct MeshTriLight;

// TODO: currently mesh lights cannot be instanced because triangle areas will change with non-similar transforms. How
// to properly support instancing?
struct MeshLightShared
{
    MeshLightShared() = default;
    MeshLightShared(uint32_t inst_id, uint32_t geom_id, const MeshGeometry &geom, const Transform &transform,
                    const ShaderField3 &emission, const ShaderField1 *opacity_map);

    uint32_t inst_id = 0;
    uint32_t geom_id = 0;
    Transform transform;
    const MeshGeometry *geom = nullptr;
    const ShaderField3 *emission = nullptr;
    const ShaderField1 *opacity_map = nullptr;

    std::vector<MeshTriLight> lights;
    std::vector<uint32_t> prim_ids;
    std::vector<float> importance;
    std::vector<float> prim_areas;
};

struct MeshTriLight : public Light
{
    bool delta_position() const { return false; };
    bool delta_direction() const { return false; };

    color3 eval(const vec3 &p_shade, const vec3 &wi, float &wi_dist) const
    {
        // TODO
        ASSERT(false);
        return color3::Zero();
    }
    color3 eval(const Intersection &it) const;
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const;
    color3 power(const AABB3 &scene_bound) const;

    uint32_t idx;
    const MeshLightShared *shared = nullptr;
};

std::unique_ptr<Light> create_light(const ConfigArgs &args);
std::unique_ptr<SkyLight> create_sky_light(const ConfigArgs &args);
std::unique_ptr<DirectionalLight> create_directional_light(const ConfigArgs &args);
std::unique_ptr<PointLight> create_point_light(const ConfigArgs &args);

//-----------------------------------------------------------------------------
// [Light Samplers]
//-----------------------------------------------------------------------------

// TODO: Light BVH, or just ReSTIR, etc...

struct MeshTriIndex
{
    uint32_t inst_id;
    uint32_t geom_id;
    uint32_t prim_id;
};

struct MeshTriIndexHash
{
    std::size_t operator()(const MeshTriIndex &idx) const { return hash(idx.inst_id, idx.geom_id, idx.prim_id); }
};
struct MeshTriIndexEqual
{
    bool operator()(const MeshTriIndex &x, const MeshTriIndex &y) const
    {
        return x.inst_id == y.inst_id && x.geom_id == y.geom_id && x.prim_id == y.prim_id;
    }
};

struct MeshLightShared;

struct LightPointers
{
    std::vector<const Light *> lights; // all non-mesh lights.
    std::vector<const MeshLightShared *> mesh_lights;
};

struct LightSampler
{
    virtual ~LightSampler() = default;

    virtual void build(LightPointers light_ptrs) = 0;
    virtual std::pair<uint32_t, const Light *> sample(float u, float &pr) const = 0;
    virtual float probability(uint32_t light_index) const = 0;
    virtual const Light *get(uint32_t light_index) const = 0;
    virtual uint32_t light_count() const = 0;
    virtual std::span<const std::pair<uint32_t, const SkyLight *>> get_sky_lights() const = 0;
    virtual std::pair<const MeshTriLight *, float> get_mesh_light(MeshTriIndex index) const = 0;
};

struct PowerLightSampler : public LightSampler
{
    explicit PowerLightSampler(const ks::AABB3 &scene_bound) : scene_bound(scene_bound) {}

    void build(LightPointers light_ptrs) final;
    std::pair<uint32_t, const Light *> sample(float u, float &pr) const final;
    float probability(uint32_t light_index) const final;
    const Light *get(uint32_t light_index) const final;
    uint32_t light_count() const { return index_psum.back(); }

    std::span<const std::pair<uint32_t, const SkyLight *>> get_sky_lights() const { return sky_lights; }

    std::pair<const MeshTriLight *, float> get_mesh_light(MeshTriIndex index) const
    {
        auto it = mesh_light_map.find(index);
        if (it != mesh_light_map.end()) {
            uint32_t light_index = it->second;
            return {(const MeshTriLight *)get(light_index), probability(light_index)};
        } else {
            return {nullptr, 0.0f};
        }
    }

    ks::AABB3 scene_bound;

    std::vector<uint32_t> index_psum;
    std::vector<const Light *> lights;
    std::vector<const MeshLightShared *> mesh_lights;

    std::vector<std::pair<uint32_t, const SkyLight *>> sky_lights;
    std::unordered_map<MeshTriIndex, uint32_t, MeshTriIndexHash, MeshTriIndexEqual> mesh_light_map;

    DistribTable power_distrib;
};

} // namespace ks