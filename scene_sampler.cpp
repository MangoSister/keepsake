#include "scene_sampler.h"
#include "material.h"
#include "opacity_map.h"
#include "rng.h"
#include "scene.h"
#include "tri_box.h"

#include <bit>

namespace ks
{

SceneSurfaceSampler::SceneSurfaceSampler(const Scene &scene) : scene(&scene) { construct_unbounded(); }

void SceneSurfaceSampler::construct_unbounded()
{
    for (int inst_id = 0; inst_id < scene->instances.size(); ++inst_id) {
        int subscene_id = scene->instances[inst_id].prototype;
        for (int geom_id = 0; geom_id < scene->subscenes[subscene_id]->geometries.size(); ++geom_id)
            map_to_geom.push_back(vec3i(subscene_id, inst_id, geom_id));
    }

    int flat_count = (int)map_to_geom.size();

    prim_tables.resize(flat_count);
    std::vector<float> geom_areas(flat_count);
    total_area = 0.0;
    for (int flat_idx = 0; flat_idx < map_to_geom.size(); ++flat_idx) {
        const vec3i &idx = map_to_geom[flat_idx];
        int subscene_id = idx[0];
        int inst_id = idx[1];
        int geom_id = idx[2];
        const Transform &transform = scene->get_instance_transform(inst_id);
        const MeshGeometry &geom = scene->get_prototype_mesh_geometry(subscene_id, geom_id);

        int n_prims = (int)geom.data->indices.size() / 3;
        std::vector<float> prim_areas(n_prims);
        geom_areas[flat_idx] = 0.0f;
        for (int prim_id = 0; prim_id < n_prims; ++prim_id) {
            int i0 = geom.data->indices[3 * prim_id];
            int i1 = geom.data->indices[3 * prim_id + 1];
            int i2 = geom.data->indices[3 * prim_id + 2];
            vec3 v0 = geom.data->get_pos(i0);
            vec3 v1 = geom.data->get_pos(i1);
            vec3 v2 = geom.data->get_pos(i2);
            v0 = transform.point(v0);
            v1 = transform.point(v1);
            v2 = transform.point(v2);
            vec3 e01 = v1 - v0;
            vec3 e12 = v2 - v1;
            vec3 ng = e01.cross(e12);
            prim_areas[prim_id] = 0.5f * ng.norm();
            geom_areas[flat_idx] += prim_areas[prim_id];
        }
        prim_tables[flat_idx] = DistribTable(prim_areas.data(), n_prims);
        total_area += geom_areas[flat_idx];
    }
    geom_table = DistribTable(geom_areas.data(), flat_count);
}

SurfaceSample SceneSurfaceSampler::sample_unbounded(ks::RNG &rng, bool sample_both_sides) const
{
    SurfaceSample s;

    float prob;
    uint32_t flat_id = geom_table.sample(rng.next(), prob);
    uint32_t prim_id = prim_tables[flat_id].sample(rng.next(), prob);
    vec3i idx = map_to_geom[flat_id];
    int subscene_id = idx[0];
    int inst_id = idx[1];
    int geom_id = idx[2];
    const Transform &transform = scene->get_instance_transform(inst_id);
    const MeshGeometry &geom = scene->get_prototype_mesh_geometry(subscene_id, geom_id);

    int i0 = geom.data->indices[3 * prim_id];
    int i1 = geom.data->indices[3 * prim_id + 1];
    int i2 = geom.data->indices[3 * prim_id + 2];
    vec3 v0 = geom.data->get_pos(i0);
    vec3 v1 = geom.data->get_pos(i1);
    vec3 v2 = geom.data->get_pos(i2);

    vec2 bary = sample_triangle_bary(rng.next2d());
    s.inst_id = inst_id;
    s.subscene_id = subscene_id;
    s.geom_id = geom_id;
    s.prim_id = prim_id;
    s.bary = bary;

    s.position = geom.interpolate_position(s.prim_id, s.bary);
    s.texcoord = geom.interpolate_texcoord(s.prim_id, s.bary);
    s.sh_normal = geom.interpolate_vertex_normal(s.prim_id, s.bary, &s.geometry_normal);
    if (geom.data->twosided && sample_both_sides) {
        if (rng.next() < 0.5f) {
            s.geometry_normal = -s.geometry_normal;
            s.sh_normal = -s.sh_normal;
        }
    }

    s.position = transform.point(s.position);
    s.sh_normal = transform.normal(s.sh_normal);
    s.geometry_normal = transform.normal(s.geometry_normal);

    return s;
}

std::vector<SurfaceSample> SceneSurfaceSampler::sample_unbounded(int N, RNG &rng, bool sample_both_sides) const
{
    std::vector<SurfaceSample> samples(N);
    for (int i = 0; i < N; ++i) {
        samples[i] = sample_unbounded(rng, sample_both_sides);
    }
    return samples;
}

//////////////////////////////////////////////////////////////////////////

enum ClipFlag
{
    Left = 1 << 0,
    Right = 1 << 1,
    Bottom = 1 << 2,
    Top = 1 << 3,
    Near = 1 << 4,
    Far = 1 << 5,
};

static constexpr int max_clip_vertex_count = 9 + 1;

template <uint32_t axis, bool side>
static inline bool inside(const vec3 &v)
{
    if constexpr (side) {
        return v[axis] <= 1.0f;
    } else {
        return v[axis] >= 0.0f;
    }
}

template <uint32_t axis, bool side>
static inline float lerp_factor(const vec3 &h1, const vec3 &h2)
{
    float t = 0.0f;
    if constexpr (side) {
        t = (1.0f - h1[axis]) / (h2[axis] - h1[axis]);
    } else {
        t = (0.0f - h1[axis]) / (h2[axis] - h1[axis]);
    }
    t = std::clamp(t, 0.0f, 1.0f);
    return t;
}

template <uint32_t axis, bool side>
static inline void clip(const vec3 &h1, const vec2 &bary1, const vec3 &h2, const vec2 &bary2, vec3 &ho, vec2 &baryo)
{
    float t = lerp_factor<axis, side>(h1, h2);
    ho = lerp(h1, h2, t);
    ho[axis] = side ? 1.0f : 0.0f;
    baryo = lerp(bary1, bary2, t);
}

template <uint32_t axis, bool side>
static inline bool sutherland_hodgman_pass(vec3 *pos1, vec2 *bary1, int n1, vec3 *pos2, vec2 *bary2, int &n2)
{
    n2 = 0;
    pos1[n1] = pos1[0];
    bary1[n1] = bary1[0];
    bool insideStart = inside<axis, side>(pos1[0]);
    for (int j = 0; j < n1; ++j) {
        bool insideEnd = inside<axis, side>(pos1[j + 1]);
        if (insideStart && insideEnd) {
            pos2[n2] = pos1[j + 1];
            bary2[n2] = bary1[j + 1];
            if (n2 == 0 || (pos2[n2] != pos2[n2 - 1] && pos2[n2] != pos2[0]))
                ++n2;
        } else if (insideStart && !insideEnd) {
            clip<axis, side>(pos1[j], bary1[j], pos1[j + 1], bary1[j + 1], pos2[n2], bary2[n2]);
            if (n2 == 0 || (pos2[n2] != pos2[n2 - 1] && pos2[n2] != pos2[0]))
                ++n2;
        } else if (!insideStart && insideEnd) {
            clip<axis, side>(pos1[j + 1], bary1[j + 1], pos1[j], bary1[j], pos2[n2], bary2[n2]);
            if (n2 == 0 || (pos2[n2] != pos2[n2 - 1] && pos2[n2] != pos2[0]))
                ++n2;

            pos2[n2] = pos1[j + 1];
            bary2[n2] = bary1[j + 1];
            if (n2 == 0 || (pos2[n2] != pos2[n2 - 1] && pos2[n2] != pos2[0]))
                ++n2;
        }

        // !insideStart && !insideEnd: do nothing.
        insideStart = insideEnd;
    }
    return n2 >= 3;
}

int clip_tri_by_box(const std::array<vec3, 3> &tri, const AABB3 &box, std::span<vec2> ret_bary)
{
    // transform to unit box [0,1]^3
    vec3 box_center = box.center();
    vec3 box_ext = box.extents();

    std::array<vec3, max_clip_vertex_count> pos1;
    std::array<vec2, max_clip_vertex_count> bary1;
    pos1[0] = (tri[0] - box_center).cwiseQuotient(box_ext) + vec3::Constant(0.5f);
    pos1[1] = (tri[1] - box_center).cwiseQuotient(box_ext) + vec3::Constant(0.5f);
    pos1[2] = (tri[2] - box_center).cwiseQuotient(box_ext) + vec3::Constant(0.5f);
    bary1[0] = vec2(1.0f, 0.0f);
    bary1[1] = vec2(0.0f, 1.0f);
    bary1[2] = vec2(0.0f, 0.0f);
    std::array<vec3, max_clip_vertex_count> pos2;
    std::array<vec2, max_clip_vertex_count> bary2;

    int nin = 3;
    vec3 *in_pos = pos1.data();
    vec2 *in_bary = bary1.data();

    int nout = 0;
    vec3 *out_pos = pos2.data();
    vec2 *out_bary = bary2.data();

    auto swap_buffers = [&]() {
        std::swap(nin, nout);
        std::swap(in_pos, out_pos);
        std::swap(in_bary, out_bary);
    };

    uint8_t vert_clip_flags[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        vert_clip_flags[i] |= inside<0, true>(in_pos[i]) ? 0 : ClipFlag::Right;
        vert_clip_flags[i] |= inside<0, false>(in_pos[i]) ? 0 : ClipFlag::Left;
        vert_clip_flags[i] |= inside<1, true>(in_pos[i]) ? 0 : ClipFlag::Top;
        vert_clip_flags[i] |= inside<1, false>(in_pos[i]) ? 0 : ClipFlag::Bottom;
        vert_clip_flags[i] |= inside<2, true>(in_pos[i]) ? 0 : ClipFlag::Far;
        vert_clip_flags[i] |= inside<2, false>(in_pos[i]) ? 0 : ClipFlag::Near;
    }
    if (vert_clip_flags[0] & vert_clip_flags[1] & vert_clip_flags[2]) {
        return 0; // All out.
    }
    uint8_t clip_flags = vert_clip_flags[0] | vert_clip_flags[1] | vert_clip_flags[2];
    if (clip_flags & ClipFlag::Right) {
        if (!sutherland_hodgman_pass<0, true>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    if (clip_flags & ClipFlag::Left) {
        if (!sutherland_hodgman_pass<0, false>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    if (clip_flags & ClipFlag::Top) {
        if (!sutherland_hodgman_pass<1, true>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    if (clip_flags & ClipFlag::Bottom) {
        if (!sutherland_hodgman_pass<1, false>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    if (clip_flags & ClipFlag::Far) {
        if (!sutherland_hodgman_pass<2, true>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    if (clip_flags & ClipFlag::Near) {
        if (!sutherland_hodgman_pass<2, false>(in_pos, in_bary, nin, out_pos, out_bary, nout))
            return 0;
        swap_buffers();
    }
    swap_buffers();

    std::copy(out_bary, out_bary + nout, ret_bary.begin());
    return nout;
}

BoundedSurfaceSampler::BoundedSurfaceSampler(const Scene &scene, const AABB3 &sample_bound)
    : scene(&scene), sample_bound(sample_bound)
{
    auto query_func = [](RTCPointQueryFunctionArguments *args) -> bool {
        BoundedSurfaceSampler &self = *(BoundedSurfaceSampler *)(args->userPtr);

        uint32_t inst_id = args->context->instID[0];
        uint32_t subscene_id = self.scene->instances[inst_id].prototype;
        uint32_t geom_id = args->geomID;
        uint32_t prim_id = args->primID;

        const Transform &transform = self.scene->get_instance_transform(inst_id);
        const MeshGeometry &geom = self.scene->get_prototype_mesh_geometry(subscene_id, geom_id);

        // NOTE: Embree may rotate the barycentric coordinates between vertices.
        // So we get the vertices by interpolate_position to be consistent.
        std::array<vec3, 3> tri;
        tri[0] = geom.interpolate_position(prim_id, vec2(1, 0));
        tri[1] = geom.interpolate_position(prim_id, vec2(0, 1));
        tri[2] = geom.interpolate_position(prim_id, vec2(0, 0));
        tri[0] = transform.point(tri[0]);
        tri[1] = transform.point(tri[1]);
        tri[2] = transform.point(tri[2]);

        if (tri_box_overlap(tri, self.sample_bound)) {
            std::array<vec2, max_clip_vertex_count> clipped_bary;
            int nclip = clip_tri_by_box(tri, self.sample_bound, clipped_bary);
            int nprev = self.clipped_bary_buffer.size();
            for (int i = 0; i < nclip; ++i) {
                self.clipped_bary_buffer.push_back(clipped_bary[i]);
            }
            for (int i = 0; i < nclip - 2; ++i) {
                int i0 = nprev;
                int i1 = nprev + i + 1;
                int i2 = nprev + i + 2;

                const Material &mat = *self.scene->subscenes[subscene_id]->materials[geom_id];
                if (!mat.opacity_map) {
                    self.clipped_index_buffer.push_back(vec3i(i0, i1, i2));
                    self.clipped_id_buffer.push_back(vec4i(inst_id, subscene_id, geom_id, prim_id));
                } else {
                    vec2 b0 = self.clipped_bary_buffer[i0];
                    vec2 b1 = self.clipped_bary_buffer[i1];
                    vec2 b2 = self.clipped_bary_buffer[i2];

                    uint64_t opacity_mask = 0;
                    for (int j = 0; j < 64; ++j) {
                        vec2 bary_nested = sample_triangle_bary(hammersley_2d(j, 64));
                        vec2 bary = bary_nested[0] * b0 + bary_nested[1] * b1 +
                                    saturate(1.0f - bary_nested[0] - bary_nested[1]) * b2;
                        vec2 texcoord = geom.interpolate_texcoord(prim_id, bary);
                        bool valid = mat.opacity_map->eval(texcoord) > 0.5f;
                        opacity_mask |= ((uint64_t)(valid) << j);
                    }
                    if (opacity_mask) {
                        self.clipped_tri_opacity_mask.insert(
                            {(uint32_t)self.clipped_index_buffer.size(), opacity_mask});
                        self.clipped_index_buffer.push_back(vec3i(i0, i1, i2));
                        self.clipped_id_buffer.push_back(vec4i(inst_id, subscene_id, geom_id, prim_id));
                    }
                    // Otherwise, we ignore this triangle.
                    // TODO: clipped_bary_buffer may have some wasted residual vertices.
                }
            }
        }
        return false;
    };

    vec3 sample_bound_center = sample_bound.center();
    float bound_radius = sample_bound.extents().norm();
    RTCPointQuery pq;
    pq.x = sample_bound_center.x();
    pq.y = sample_bound_center.y();
    pq.z = sample_bound_center.z();
    pq.radius = bound_radius;
    RTCPointQueryContext ctx;
    rtcInitPointQueryContext(&ctx);
    // TODO: double check -- will the same primitive get invoked multiple times?
    rtcPointQuery(scene.rtcscene, &pq, &ctx, query_func, (void *)this);

    std::vector<float> areas(clipped_index_buffer.size());
    for (int i = 0; i < clipped_index_buffer.size(); ++i) {
        uint32_t inst_id = clipped_id_buffer[i][0];
        uint32_t subscene_id = clipped_id_buffer[i][1];
        uint32_t geom_id = clipped_id_buffer[i][2];
        uint32_t prim_id = clipped_id_buffer[i][3];

        const Transform &transform = scene.get_instance_transform(inst_id);
        const MeshGeometry &geom = scene.get_prototype_mesh_geometry(subscene_id, geom_id);

        uint32_t i0 = clipped_index_buffer[i][0];
        uint32_t i1 = clipped_index_buffer[i][1];
        uint32_t i2 = clipped_index_buffer[i][2];
        vec2 b0 = clipped_bary_buffer[i0];
        vec2 b1 = clipped_bary_buffer[i1];
        vec2 b2 = clipped_bary_buffer[i2];

        vec3 v0 = geom.interpolate_position(prim_id, b0);
        vec3 v1 = geom.interpolate_position(prim_id, b1);
        vec3 v2 = geom.interpolate_position(prim_id, b2);
        v0 = transform.point(v0);
        v1 = transform.point(v1);
        v2 = transform.point(v2);

        vec3 e01 = v1 - v0;
        vec3 e12 = v2 - v1;
        vec3 ng = e01.cross(e12);

        float area = 0.5f * ng.norm();
        if (auto it = clipped_tri_opacity_mask.find(i); it != clipped_tri_opacity_mask.end()) {
            area *= (float)std::popcount(it->second) / 64.0f;
        }
        areas[i] = area;
        total_area += areas[i];
    }
    distrib = DistribTable(areas.data(), areas.size());
}

std::vector<SurfaceSample> BoundedSurfaceSampler::sample(int N, RNG &rng, bool sample_both_sides) const
{
    std::vector<SurfaceSample> samples(N);

    for (int i = 0; i < N; ++i) {
        float prob;
        uint32_t p = distrib.sample(rng.next(), prob);

        uint32_t inst_id = clipped_id_buffer[p][0];
        uint32_t subscene_id = clipped_id_buffer[p][1];
        uint32_t geom_id = clipped_id_buffer[p][2];
        uint32_t prim_id = clipped_id_buffer[p][3];

        const Transform &transform = scene->get_instance_transform(inst_id);
        const MeshGeometry &geom = scene->get_prototype_mesh_geometry(subscene_id, geom_id);

        uint32_t i0 = clipped_index_buffer[p][0];
        uint32_t i1 = clipped_index_buffer[p][1];
        uint32_t i2 = clipped_index_buffer[p][2];
        vec2 b0 = clipped_bary_buffer[i0];
        vec2 b1 = clipped_bary_buffer[i1];
        vec2 b2 = clipped_bary_buffer[i2];

        vec2 bary_nested;
        if (auto it = clipped_tri_opacity_mask.find(p); it != clipped_tri_opacity_mask.end()) {
            // This triangle is opacity-mapped. Sample discrete sub-tri points.
            // Also with some jittering (with side length 0.5/sqrt(64)) to avoid 0 determinant for ellispoids
            // (jitter may introduce some error)
            uint32_t bitpos = sample_bitmask_64(it->second, rng.next());
            vec2 u = hammersley_2d(bitpos, 64);
            u.x() += std::lerp(-0.5f, 0.5f, rng.next()) / 8.0f;
            u.x() = saturate(u.x());
            u.y() += std::lerp(-0.5f, 0.5f, rng.next()) / 8.0f;
            u.y() = saturate(u.y());
            bary_nested = sample_triangle_bary(u);
        } else {
            bary_nested = sample_triangle_bary(rng.next2d());
        }

        samples[i].inst_id = inst_id;
        samples[i].subscene_id = subscene_id;
        samples[i].geom_id = geom_id;
        samples[i].prim_id = prim_id;
        samples[i].bary =
            bary_nested[0] * b0 + bary_nested[1] * b1 + saturate(1.0f - bary_nested[0] - bary_nested[1]) * b2;

        samples[i].position = geom.interpolate_position(samples[i].prim_id, samples[i].bary);
        samples[i].texcoord = geom.interpolate_texcoord(samples[i].prim_id, samples[i].bary);
        samples[i].sh_normal =
            geom.interpolate_vertex_normal(samples[i].prim_id, samples[i].bary, &samples[i].geometry_normal);
        if (geom.data->twosided && sample_both_sides) {
            if (rng.next() < 0.5f) {
                samples[i].geometry_normal = -samples[i].geometry_normal;
                samples[i].sh_normal = -samples[i].sh_normal;
            }
        }

        samples[i].position = transform.point(samples[i].position);
        samples[i].sh_normal = transform.normal(samples[i].sh_normal);
        samples[i].geometry_normal = transform.normal(samples[i].geometry_normal);

        AABB3 extended_bound = sample_bound;
        extended_bound.min -= vec3::Constant(0.001f);
        extended_bound.max += vec3::Constant(0.001f);
        ASSERT(extended_bound.contain(samples[i].position),
               "Sampled position (%f, %f, %f) should be within the bound [(%f, %f, %f), (%f, %f, %f)]!",
               samples[i].position.x(), samples[i].position.y(), samples[i].position.z(), sample_bound.min.x(),
               sample_bound.min.y(), sample_bound.min.z(), sample_bound.max.x(), sample_bound.max.y(),
               sample_bound.max.z());
    }

    return samples;
}

} // namespace ks