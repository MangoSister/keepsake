#pragma once

#include "aabb.h"
#include "distrib.h"

namespace ks
{
struct Scene;
struct RNG;

struct SurfaceSample
{
    uint32_t inst_id;
    uint32_t subscene_id;
    uint32_t geom_id;
    uint32_t prim_id;
    ks::vec2 bary;

    ks::vec3 position;
    ks::vec2 texcoord;
    ks::vec3 geometry_normal;
    ks::vec3 sh_normal;
};

struct SceneSurfaceSampler
{
    explicit SceneSurfaceSampler(const ks::Scene &scene);
    void construct_unbounded();

    bool empty() const { return prim_tables.empty(); }

    SurfaceSample sample_unbounded(ks::RNG &rng, bool sample_both_sides) const;
    std::vector<SurfaceSample> sample_unbounded(int N, ks::RNG &rng, bool sample_both_sides) const;
    // TODO: how to efficiently support opacity mask?

    const ks::Scene *scene = nullptr;

    std::vector<ks::vec3i> map_to_geom;
    ks::DistribTable geom_table;
    std::vector<ks::DistribTable> prim_tables;

    double total_area = 0.0; // When assets are really large...
};

struct BoundedSurfaceSampler
{
    BoundedSurfaceSampler(const ks::Scene &scene, const ks::AABB3 &bound);

    bool empty() const { return clipped_index_buffer.empty(); }
    std::vector<SurfaceSample> sample(int N, ks::RNG &rng, bool sample_both_sides) const;

    const ks::Scene *scene = nullptr;
    ks::AABB3 sample_bound;
    std::vector<ks::vec2> clipped_bary_buffer;
    std::vector<ks::vec3i> clipped_index_buffer;
    std::vector<ks::vec4i> clipped_id_buffer;

    ks::DistribTable distrib;
    double total_area = 0.0f; // When assets are really large...

    std::unordered_map<uint32_t, uint64_t> clipped_tri_opacity_mask;
};

} // namespace ks