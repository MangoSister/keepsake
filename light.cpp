#include "light.h"
#include "image_util.h"
#include "parallel.h"
#include "ray.h"
#include "rng.h"

KS_NAMESPACE_BEGIN

SkyLight::SkyLight(const fs::path &path, const Transform &l2w, float strength) : l2w(l2w), strength(strength)
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

color3 SkyLight::eval(const vec3 &p, const vec3 &wi) const
{
    vec3 wi_local = l2w.inverse().direction(wi);

    float phi, theta;
    to_spherical_yup(wi_local, phi, theta);
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
    return L;
}

color3 SkyLight::sample(const vec3 &p, const vec2 &u, vec3 &wi, float &pdf) const
{
    vec2 uv = distrib.sample_linear(u, pdf);
    if (pdf == 0.0f) {
        return color3::Zero();
    }
    float phi = uv[0] * 2.0f * pi;
    float theta = uv[1] * pi;
    vec3 wi_local = to_cartesian_yup(phi, theta);
    float sin_theta = std::sin(theta);
    pdf /= (2.0f * pi * pi * sin_theta);
    if (sin_theta == 0.0f) {
        pdf = 0.0f;
    }
    wi = l2w.direction(wi_local);
    return eval(p, wi) / pdf;
}

float SkyLight::pdf(const vec3 &p, const vec3 &wi) const
{
    vec3 wi_local = l2w.inverse().direction(wi);
    float phi, theta;
    to_spherical_yup(wi_local, phi, theta);
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

color3 DirectionalLight::eval(const vec3 &p_shade, const vec3 &wi) const { return color3::Zero(); }

color3 DirectionalLight::sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &pdf) const
{
    wi = dir;
    pdf = 1.0f;
    return L;
}

float DirectionalLight::pdf(const vec3 &p_shade, const vec3 &wi) const { return 0.0f; }

std::unique_ptr<Light> create_light(const ConfigArgs &args)
{
    std::string light_type = args.load_string("type");
    std::unique_ptr<Light> light;
    if (light_type == "directional") {
        light = create_directional_light(args);
    } else if (light_type == "sky") {
        light = create_sky_light(args);
    }
    return light;
}

std::unique_ptr<SkyLight> create_sky_light(const ConfigArgs &args)
{
    fs::path map = args.load_path("map");
    Transform to_world;
    if (args.contains("to_world"))
        to_world = args.load_transform("to_world");
    float strength = args.load_float("strength", 1.0f);
    return std::make_unique<SkyLight>(map, to_world, strength);
}

std::unique_ptr<DirectionalLight> create_directional_light(const ConfigArgs &args)
{
    color3 L = args.load_vec3("L").array();
    vec3 dir = args.load_vec3("dir", true);
    return std::make_unique<DirectionalLight>(L, dir);
}

KS_NAMESPACE_END