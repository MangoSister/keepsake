#include "light.h"
#include "image_util.h"
#include "parallel.h"
#include "ray.h"
#include "rng.h"
#include "sat.h"
#include <atomic>
#include <tuple>
#include <utility>

namespace ks
{

//-----------------------------------------------------------------------------
// [Different types of lights]
//-----------------------------------------------------------------------------

SkyLight::SkyLight(const fs::path &path, const Transform &l2w, bool transform_y_up, float strength)
    : l2w(l2w), transform_y_up(transform_y_up), strength(strength)
{
    int width, height;
    std::unique_ptr<color3[]> pixels = load_from_hdr<3>(path, width, height);
    map = BlockedArray<color3>(width, height, 1, pixels.get());
    std::vector<float> lum(width * height);
    std::atomic<float> lum_sum(0.0);
    parallel_for((int)lum.size(), [&](int i) {
        int y = i / width;
        float theta = (y + 0.5f) / (float)height * pi;
        float sin_theta = std::sin(theta);
        lum[i] = sin_theta * luminance(pixels[i]);
        lum_sum.fetch_add(lum[i]);
    });

    // Normal PMF table
    // distrib = DistribTable2D(lum.data(), map.ures, map.vres);

    // Thresholded PMF table. MIS compensation
    float average = lum_sum.load() / lum.size();
    parallel_for((int)lum.size(), [&](int i) { lum[i] = std::max(lum[i] - average, 0.0f); });
    distrib = DistribTable2D(lum.data(), map.ures, map.vres);
}

SkyLight::SkyLight(const color3 &ambient)
{
    map = BlockedArray<color3>(1, 1, 1, &ambient);
    float lum = luminance(ambient);
    distrib = DistribTable2D(&lum, map.ures, map.vres);
}

color3 SkyLight::eval(const vec3 &p, const vec3 &wi, float &wi_dist) const
{
    vec3 wi_local = l2w.inverse().direction(wi);

    float phi, theta;
    if (transform_y_up) {
        to_spherical_yup(wi_local, phi, theta);
    } else {
        to_spherical(wi_local, phi, theta);
    }
    vec2 uv;
    uv[0] = phi * inv_pi * 0.5f;
    uv[1] = theta * inv_pi;
    vec2i res(map.ures, map.vres);
    WrapMode wrap[2] = {WrapMode::Repeat, WrapMode::Clamp};
    TickMode tick[2] = {TickMode::Middle, TickMode::Middle};
    vec2i uv0, uv1;
    vec2 t;
    lerp_helper<2>(uv.data(), res.data(), wrap, tick, uv0.data(), uv1.data(), t.data());
    color3 L = color3::Zero();
    L += (1.0f - t[0]) * (1.0f - t[1]) * map(uv0[0], uv0[1]);
    L += (1.0f - t[0]) * t[1] * map(uv0[0], uv1[1]);
    L += t[0] * (1.0f - t[1]) * map(uv1[0], uv0[1]);
    L += t[0] * t[1] * map(uv1[0], uv1[1]);
    L *= strength;

    wi_dist = inf;
    return L;
}

color3 SkyLight::sample(const vec3 &p, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const
{
    vec2 uv = distrib.sample_linear(u, pdf);
    if (pdf == 0.0f) {
        return color3::Zero();
    }
    float phi = uv[0] * 2.0f * pi;
    float theta = uv[1] * pi;
    vec3 wi_local;
    if (transform_y_up) {
        wi_local = to_cartesian_yup(phi, theta);
    } else {
        wi_local = to_cartesian(phi, theta);
    }
    float sin_theta = std::sin(theta);
    pdf /= (2.0f * pi * pi * sin_theta);
    if (sin_theta == 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }
    wi = l2w.direction(wi_local);
    return eval(p, wi, wi_dist) / pdf;
}

float SkyLight::pdf(const vec3 &p, const vec3 &wi, float wi_dist) const
{
    vec3 wi_local = l2w.inverse().direction(wi);
    float phi, theta;
    if (transform_y_up) {
        to_spherical_yup(wi_local, phi, theta);
    } else {
        to_spherical(wi_local, phi, theta);
    }
    float sin_theta = std::sin(theta);
    if (sin_theta == 0.0f) {
        return 0.0f;
    }

    vec2 uv;
    uv[0] = phi * inv_pi * 0.5f;
    uv[1] = theta * inv_pi;

    float pdf = distrib.pdf(uv);
    pdf /= (2.0f * pi * pi * sin_theta);
    return pdf;
}

color3 SkyLight::power(const AABB3 &scene_bound) const
{
    std::array<std::atomic<float>, 3> sum{0.0f, 0.0f, 0.0f};

    parallel_for(map.vres, [&](int row) {
        float theta = (row + 0.5f) / (float)map.vres * pi;
        float sin_theta = std::sin(theta);

        color3 row_sum = color3::Zero();
        for (int col = 0; col < map.ures; ++col) {
            float phi = (col + 0.5f) / (float)map.ures * two_pi;
            vec3 dir = to_cartesian(phi, theta);
            float wi_dist;
            row_sum += eval(vec3::Zero(), dir, wi_dist) * sin_theta;
        }
        for (int c = 0; c < 3; ++c)
            sum[c].fetch_add(row_sum[c]);
    });

    color3 Phi(sum[0], sum[1], sum[2]);
    Phi *= (pi / map.vres) * (two_pi / map.ures);
    float scene_radius = 0.5f * scene_bound.extents().norm();
    Phi *= pi * sqr(scene_radius);
    return Phi;
}

color3 DirectionalLight::eval(const vec3 &p_shade, const vec3 &wi, float &wi_dist) const
{
    wi_dist = inf;
    return color3::Zero();
}

color3 DirectionalLight::sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const
{
    wi = dir;
    wi_dist = inf;
    pdf = 1.0f;
    return L;
}

float DirectionalLight::pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const { return 0.0f; }

color3 DirectionalLight::power(const AABB3 &scene_bound) const
{
    float scene_radius = 0.5f * scene_bound.extents().norm();
    return L * pi * sqr(scene_radius);
}

color3 PointLight::eval(const vec3 &p_shade, const vec3 &wi, float &wi_dist) const
{
    wi_dist = (pos - p_shade).norm();
    return color3::Zero();
}

color3 PointLight::sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const
{
    wi = pos - p_shade;
    float l2 = wi.squaredNorm();
    if (l2 == 0.0f) {
        return color3(0.0f);
    }
    float l = std::sqrt(l2);
    wi /= l;
    wi_dist = l;
    pdf = 1.0f;
    return I / l2;
}

float PointLight::pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const { return 0.0f; }

color3 PointLight::power(const AABB3 &scene_bound) const { return 4.0f * pi * I; }

std::unique_ptr<Light> create_light(const ConfigArgs &args)
{
    std::string light_type = args.load_string("type");
    std::unique_ptr<Light> light;
    if (light_type == "directional") {
        light = create_directional_light(args);
    } else if (light_type == "point") {
        light = create_point_light(args);
    } else if (light_type == "sky") {
        light = create_sky_light(args);
    }
    return light;
}

std::unique_ptr<SkyLight> create_sky_light(const ConfigArgs &args)
{
    if (args.contains("ambient")) {
        color3 ambient = args.load_vec3("ambient").array();
        return std::make_unique<SkyLight>(ambient);
    } else {
        fs::path map = args.load_path("map");
        Transform to_world;
        if (args.contains("to_world"))
            to_world = args.load_transform("to_world");
        bool transform_y_up = args.load_bool("transform_y_up", true);
        float strength = args.load_float("strength", 1.0f);
        return std::make_unique<SkyLight>(map, to_world, transform_y_up, strength);
    }
}

std::unique_ptr<DirectionalLight> create_directional_light(const ConfigArgs &args)
{
    color3 L = args.load_vec3("L").array();
    vec3 dir = args.load_vec3("dir", true);
    return std::make_unique<DirectionalLight>(L, dir);
}

std::unique_ptr<PointLight> create_point_light(const ConfigArgs &args)
{
    color3 I = args.load_vec3("I").array();
    vec3 pos = args.load_vec3("pos", false);
    return std::make_unique<PointLight>(I, pos);
}

static std::pair<std::vector<float>, std::vector<float>> compute_mesh_light_importance(const MeshGeometry &geom,
                                                                                       const Transform &to_world,
                                                                                       const ShaderField3 &emission,
                                                                                       const ShaderField1 *opacity_map)
{
    bool fallback = false;
    bool uniform = (dynamic_cast<const ConstantField<color<3>> *>(&emission)) &&
                   (!opacity_map || dynamic_cast<const ConstantField<color<1>> *>(opacity_map));
    bool textured = !uniform &&
                    (dynamic_cast<const ConstantField<color<3>> *>(&emission) ||
                     dynamic_cast<const TextureField<3> *>(&emission)) &&
                    (!opacity_map || dynamic_cast<const ConstantField<color<1>> *>(opacity_map) ||
                     dynamic_cast<const TextureField<1> *>(opacity_map));
    if (uniform) {
        fallback = true;
    }
    if (!uniform && !textured) {
        // TODO: don't have a very good way to handle uv scale or just arbitrary procedural fields...
        fprintf(stderr,
                "[MeshLight] driven by non constant nor textured fields. Fall back to naive sampling strategy.\n");
        fallback = true;
    }
    vec2i res(1, 1);
    std::optional<TextureWrapMode> wrap_mode_u;
    std::optional<TextureWrapMode> wrap_mode_v;
    std::optional<bool> flip_v;
    std::optional<vec2> uv_scale;
    std::optional<vec2> uv_offset;
    if (const TextureField<3> *tex = (dynamic_cast<const TextureField<3> *>(&emission)); tex) {
        res = vec2i(tex->texture->width, tex->texture->height);
        wrap_mode_u = tex->sampler->wrap_mode_u;
        wrap_mode_v = tex->sampler->wrap_mode_v;
        flip_v = tex->flip_v;
        uv_scale = tex->uv_scale;
        uv_offset = tex->uv_offset;
    }
    if (const TextureField<1> *tex = (dynamic_cast<const TextureField<1> *>(opacity_map)); tex) {
        res = res.cwiseMax(vec2i(tex->texture->width, tex->texture->height));
        if (!wrap_mode_u) {
            wrap_mode_u = tex->sampler->wrap_mode_u;
        } else if (wrap_mode_u && wrap_mode_u != tex->sampler->wrap_mode_u) {
            fallback = true;
        }
        if (!wrap_mode_v) {
            wrap_mode_v = tex->sampler->wrap_mode_v;
        } else if (wrap_mode_v && wrap_mode_v != tex->sampler->wrap_mode_v) {
            fallback = true;
        }
        if (!flip_v) {
            flip_v = tex->flip_v;
        } else if (flip_v && flip_v != tex->flip_v) {
            fallback = true;
        }
        if (!uv_scale) {
            uv_scale = tex->uv_scale;
        } else if (uv_scale && uv_scale != tex->uv_scale) {
            fallback = true;
        }
        if (!uv_offset) {
            uv_offset = tex->uv_offset;
        } else if (uv_offset && uv_offset != tex->uv_offset) {
            fallback = true;
        }
        if (fallback) {
            fprintf(stderr,
                    "[MeshLight] textured fields have different settings. Fall back to naive sampling strategy.\n");
        }
    }
    std::vector<float> importance_uv(res.x() * res.y());
    std::atomic<float> sum_importance(0.0f);
    parallel_tile_2d(res.x(), res.y(), [&](int x, int y) {
        Intersection it;
        it.uv = vec2((x + 0.5f) / (float)res.x(), (y + 0.5f) / (float)res.y());
        float importance = luminance(emission(it));
        importance *= opacity_map ? (*opacity_map)(it)[0] : 1.0f;
        importance_uv[y * res.x() + x] = importance;
        sum_importance.fetch_add(importance);
    });
    float avg_importance = sum_importance.load() / importance_uv.size();

    if (fallback) {
        int n_prims = (int)geom.data->indices.size() / 3;
        std::vector<float> importance(n_prims);
        std::vector<float> prim_areas(n_prims);
        parallel_for(n_prims, [&](int prim_id) {
            int i0 = geom.data->indices[3 * prim_id];
            int i1 = geom.data->indices[3 * prim_id + 1];
            int i2 = geom.data->indices[3 * prim_id + 2];
            vec3 v0 = geom.data->get_pos(i0);
            vec3 v1 = geom.data->get_pos(i1);
            vec3 v2 = geom.data->get_pos(i2);
            v0 = to_world.point(v0);
            v1 = to_world.point(v1);
            v2 = to_world.point(v2);
            vec3 e01 = v1 - v0;
            vec3 e12 = v2 - v1;
            vec3 ng = e01.cross(e12);
            float area = 0.5f * ng.norm();
            importance[prim_id] = two_pi * area * avg_importance;
            prim_areas[prim_id] = area;
        });
        return {importance, prim_areas};
    }

    SummedAreaTable importance_sat(importance_uv, res);
    // Query the "importance" of each triangle
    // weight = (triangle area world space) * (average intensity covered by triangle uv space)
    // But approximated by checking AABB, but I guess good enough for now.
    int n_prims = (int)geom.data->indices.size() / 3;
    std::vector<float> importance(n_prims);
    std::vector<float> prim_areas(n_prims);
    parallel_for(n_prims, [&](int prim_id) {
        int i0 = geom.data->indices[3 * prim_id];
        int i1 = geom.data->indices[3 * prim_id + 1];
        int i2 = geom.data->indices[3 * prim_id + 2];
        vec3 v0 = geom.data->get_pos(i0);
        vec3 v1 = geom.data->get_pos(i1);
        vec3 v2 = geom.data->get_pos(i2);
        v0 = to_world.point(v0);
        v1 = to_world.point(v1);
        v2 = to_world.point(v2);
        vec3 e01 = v1 - v0;
        vec3 e12 = v2 - v1;
        vec3 ng = e01.cross(e12);
        float pa = 0.5f * ng.norm();

        vec2 tc0 = geom.data->get_texcoord(i0);
        vec2 tc1 = geom.data->get_texcoord(i1);
        vec2 tc2 = geom.data->get_texcoord(i2);
        if (flip_v) {
            tc0[1] = 1.0f - tc0[1];
            tc1[1] = 1.0f - tc1[1];
            tc2[1] = 1.0f - tc2[1];
        }
        tc0 = tc0.cwiseProduct(*uv_scale) + (*uv_offset);
        tc1 = tc1.cwiseProduct(*uv_scale) + (*uv_offset);
        tc2 = tc2.cwiseProduct(*uv_scale) + (*uv_offset);

        AABB2 tc_bound;
        tc_bound.expand(tc0);
        tc_bound.expand(tc1);
        tc_bound.expand(tc2);
        // Need to handle wrapping...
        int txmin = (int)std::floor(tc_bound.min.x());
        int txmax = (int)std::ceil(tc_bound.max.x());
        int tymin = (int)std::floor(tc_bound.min.y());
        int tymax = (int)std::ceil(tc_bound.max.y());
        importance[prim_id] = 0.0f;
        float sum_b_area = 0.0f;
        for (int ty = tymin; ty < tymax; ++ty) {
            for (int tx = txmin; tx < txmax; ++tx) {
                AABB2 b = intersect(tc_bound, AABB2(vec2(tx, ty), vec2(tx + 1.0f, ty + 1.0f)));
                float b_area = b.area();
                if (b_area == 0.0f) {
                    continue;
                }
                switch (*wrap_mode_u) {
                case TextureWrapMode::Repeat: {
                    b.min.x() -= std::floor(b.min.x());
                    b.max.x() -= std::floor(b.max.x());
                    break;
                }
                case TextureWrapMode::Clamp: {
                    b.min.x() = saturate(b.min.x());
                    b.max.x() = saturate(b.max.x());
                    break;
                }
                default:
                    ASSERT(false, "Invalid texture u wrap mode.");
                }
                switch (*wrap_mode_v) {
                case TextureWrapMode::Repeat: {
                    b.min.y() -= std::floor(b.min.y());
                    b.max.y() -= std::floor(b.max.y());
                    break;
                }
                case TextureWrapMode::Clamp: {
                    b.min.y() = saturate(b.min.y());
                    b.max.y() = saturate(b.max.y());
                    break;
                }
                default:
                    ASSERT(false, "Invalid texture v wrap mode.");
                }
                // If clamped, we need to make b single-pixel wide/high.
                if (b.min.x() == 0.0f && b.max.x() == 0.0f) {
                    b.max.x() = 1.0f / (res.x() + 1);
                }
                if (b.min.x() == 1.0f && b.max.x() == 1.0f) {
                    b.min.x() = 1.0f - 1.0f / (res.x() + 1);
                }
                if (b.min.y() == 0.0f && b.max.y() == 0.0f) {
                    b.max.y() = 1.0f / (res.y() + 1);
                }
                if (b.min.y() == 1.0f && b.max.y() == 1.0f) {
                    b.min.y() = 1.0f - 1.0f / (res.y() + 1);
                }
                float b_area_wrapped = b.area();
                ASSERT(b_area_wrapped > 0.0f);
                // Re-weight by unclamped area.
                importance[prim_id] += importance_sat.sum(b) / b_area_wrapped * b_area;
                sum_b_area += b_area;
            }
        }
        if (sum_b_area > 0.0f) {
            importance[prim_id] /= sum_b_area;
        }
        importance[prim_id] *= (two_pi * pa);
        prim_areas[prim_id] = pa;
    });
    return {importance, prim_areas};
}

MeshLightShared::MeshLightShared(uint32_t inst_id, uint32_t geom_id, const MeshGeometry &geom,
                                 const Transform &transform, const ShaderField3 &emission,
                                 const ShaderField1 *opacity_map)
    : inst_id(inst_id), geom_id(geom_id), geom(&geom), transform(transform), emission(&emission),
      opacity_map(opacity_map)
{
    std::tie(importance, prim_areas) = compute_mesh_light_importance(geom, transform, emission, opacity_map);
    uint32_t N = (uint32_t)importance.size();
    prim_ids.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        prim_ids[i] = i;
    }
    for (uint32_t i = 0; i < N;) {
        if (importance[i] == 0.0f) {
            std::swap(prim_ids[N - 1], prim_ids[i]);
            std::swap(importance[N - 1], importance[i]);
            std::swap(prim_areas[N - 1], prim_areas[i]);
            --N;
        } else {
            ++i;
        }
    }
    prim_ids.resize(N);
    prim_ids.shrink_to_fit();
    importance.resize(N);
    importance.shrink_to_fit();
    prim_areas.resize(N);
    prim_areas.shrink_to_fit();
    lights.resize(N);
    lights.shrink_to_fit();
    for (uint32_t i = 0; i < N; ++i) {
        lights[i].idx = i;
        lights[i].shared = this;
    }
}

color3 MeshTriLight::eval(const Intersection &it) const
{
    color3 Le = (*shared->emission)(it);
    // NOTE: assume opacity map is already handled by external intersection test.
    return Le;
}

color3 MeshTriLight::sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &wi_dist, float &pdf) const
{
    // TODO: switch to solid angle sampling.
    vec2 bary = sample_triangle_bary(u);
    uint32_t prim_id = shared->prim_ids[idx];
    const MeshGeometry &geom = *shared->geom;
    const Transform &transform = shared->transform;
    vec3 position = geom.interpolate_position(prim_id, bary);
    vec2 texcoord = geom.interpolate_texcoord(prim_id, bary);
    vec3 n = geom.compute_geometry_normal(prim_id);
    position = transform.point(position);
    n = transform.normal(n);
    wi = position - p_shade;
    float wi_dist2 = wi.squaredNorm();
    if (wi_dist2 == 0.0f) {
        wi_dist = 0.0f;
        pdf = 0.0f;
        return color3::Zero();
    }
    wi_dist = std::sqrt(wi_dist2);
    wi /= wi_dist;

    pdf = 1.0f / shared->prim_areas[idx];
    float absdot = std::abs(n.dot(wi));
    if (absdot == 0.0f) {
        wi_dist = 0.0f;
        pdf = 0.0f;
        return color3::Zero();
    }
    pdf *= (wi_dist2 / absdot);

    Intersection it;
    it.uv = texcoord;
    return eval(it) / pdf;
}

float MeshTriLight::pdf(const vec3 &p_shade, const vec3 &wi, float wi_dist) const
{
    if (wi_dist == 0.0f) {
        return 0.0f;
    }
    vec3 p = p_shade + wi * wi_dist;

    uint32_t prim_id = shared->prim_ids[idx];
    const MeshGeometry &geom = *shared->geom;
    const Transform &transform = shared->transform;
    vec3 n = geom.compute_geometry_normal(prim_id);
    p = transform.point(p);
    n = transform.normal(n);

    float pdf = 1.0f / shared->prim_areas[idx];
    float absdot = std::abs(n.dot(wi));
    if (absdot == 0.0f) {
        return 0.0f;
    }
    pdf *= (sqr(wi_dist) / absdot);
    return pdf;
}

color3 MeshTriLight::power(const AABB3 &scene_bound) const { return color3::Constant(shared->importance[idx]); }

//-----------------------------------------------------------------------------
// [Light Samplers]
//-----------------------------------------------------------------------------

void PowerLightSampler::build(LightPointers light_ptrs)
{
    lights = std::move(light_ptrs.lights);
    mesh_lights = std::move(light_ptrs.mesh_lights);

    index_psum.clear();
    sky_lights.clear();
    mesh_light_map.clear();

    index_psum.resize(1 + mesh_lights.size() + 1);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < (uint32_t)index_psum.size(); ++i) {
        index_psum[i] = sum;
        if (i < (uint32_t)index_psum.size()) {
            sum += (i == 0 ? (uint32_t)lights.size() : (uint32_t)mesh_lights[i - 1]->lights.size());
        }
    }

    std::vector<float> powers(sum);
    uint32_t flat_idx = 0;
    for (uint32_t i = 0; i < (uint32_t)lights.size(); ++i) {
        const SkyLight *sky = dynamic_cast<const SkyLight *>(lights[i]);
        if (sky) {
            sky_lights.emplace_back(i, sky);
        }
        powers[flat_idx++] = lights[i]->power(scene_bound).mean();
    }

    for (uint32_t i = 0; i < (uint32_t)mesh_lights.size(); ++i) {
        MeshTriIndex mesh_tri_idx;
        mesh_tri_idx.inst_id = mesh_lights[i]->inst_id;
        mesh_tri_idx.geom_id = mesh_lights[i]->geom_id;
        for (uint32_t j = 0; j < (uint32_t)mesh_lights[i]->lights.size(); ++j) {
            mesh_tri_idx.prim_id = mesh_lights[i]->prim_ids[mesh_lights[i]->lights[j].idx];
            mesh_light_map.insert({mesh_tri_idx, flat_idx});

            powers[flat_idx++] = mesh_lights[i]->lights[j].power(scene_bound).mean();
        }
    }

    power_distrib = DistribTable(powers.data(), lights.size());
}

std::pair<uint32_t, const Light *> PowerLightSampler::sample(float u, float &pr) const
{
    uint32_t index = power_distrib.sample(u, pr);
    return {index, lights[index]};
}

float PowerLightSampler::probability(uint32_t light_index) const
{
    return power_distrib.pdf(light_index) / (float)lights.size();
}

const Light *PowerLightSampler::get(uint32_t light_index) const
{
    ASSERT(light_index < light_count());
    auto it = std::upper_bound(index_psum.begin(), index_psum.end(), light_index);
    uint32_t group_index = (uint32_t)std::distance(index_psum.begin(), std::prev(it));
    uint32_t offset = light_index - (*std::prev(it));
    if (group_index == 0) {
        return lights[offset];
    } else {
        return &mesh_lights[group_index - 1]->lights[offset];
    }
}

} // namespace ks