#include "light.h"
#include "image_util.h"
#include "parallel.h"
#include "ray.h"
#include "rng.h"

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
    parallel_for((int)lum.size(), [&](int i) {
        int y = i / width;
        float theta = (y + 0.5f) / (float)height * pi;
        float sin_theta = std::sin(theta);
        lum[i] = sin_theta * luminance(pixels[i]);
    });
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

//-----------------------------------------------------------------------------
// [Light Samplers]
//-----------------------------------------------------------------------------

void LightSampler::build(std::span<const Light *> lights)
{
    skylights.clear();
    for (uint32_t i = 0; i < (uint32_t)lights.size(); ++i) {
        const Light *l = lights[i];
        const SkyLight *sky = dynamic_cast<const SkyLight *>(l);
        if (sky) {
            skylights.emplace_back(i, sky);
        }
    }
}

void UniformLightSampler::build(std::span<const Light *> lights)
{
    LightSampler::build(lights);

    this->lights.resize(lights.size());
    std::copy(lights.begin(), lights.end(), this->lights.begin());
}

std::pair<uint32_t, const Light *> UniformLightSampler::sample(float u, float &pr) const
{
    uint32_t N = (uint32_t)lights.size();
    uint32_t index = std::min((uint32_t)std::floor(u * N), N - 1);
    pr = 1.0f / (float)N;
    return {index, lights[index]};
}

float UniformLightSampler::probability(uint32_t light_index) const { return 1.0f / (float)lights.size(); }

const Light *UniformLightSampler::get(uint32_t light_index) const { return lights[light_index]; }

void PowerLightSampler::build(std::span<const Light *> lights)
{
    LightSampler::build(lights);

    this->lights.resize(lights.size());
    std::copy(lights.begin(), lights.end(), this->lights.begin());
    std::vector<float> powers(lights.size());
    for (uint32_t i = 0; i < (uint32_t)lights.size(); ++i) {
        powers[i] = lights[i]->power(scene_bound).mean();
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

const Light *PowerLightSampler::get(uint32_t light_index) const { return lights[light_index]; }

} // namespace ks