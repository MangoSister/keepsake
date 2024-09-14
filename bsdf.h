#pragma once
#include "config.h"
#include "maths.h"
#include "shader_field.h"

namespace ks
{

struct BSDF : public Configurable
{
    virtual ~BSDF() = default;
    virtual bool delta() const = 0;
    // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
    virtual color3 eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const = 0;
    // NOTE: Instead of the standard 2D rnd, we use 3D. 1 rnd for choosing "lobe" or whatever mode, and 2 for actually
    // sampling wi. However for simple BSDF u_lobe is usually ignored. Something better?
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    virtual color3 sample(const vec3 &wo, vec3 &wi, const Intersection &it, float u_lobe, const vec2 &u_wi,
                          float &pdf) const = 0;
    virtual float pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const = 0;
    // NOTE: MIS usually requires these together. This provides room of optimization for implementations.
    virtual std::pair<color3, float> eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const;
};

struct Lambertian : public BSDF
{
    Lambertian() = default;
    Lambertian(const color3 &albedo) { this->albedo = std::make_unique<ConstantField<color3>>(albedo); }
    Lambertian(std::unique_ptr<ShaderField<color3>> &&albedo) : albedo(std::move(albedo)){};

    bool delta() const { return false; }
    // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
    color3 eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const;
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    color3 sample(const vec3 &wo, vec3 &wi, const Intersection &it, float u_lobe, const vec2 &u_wi, float &pdf) const;
    float pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const;

    std::unique_ptr<ShaderField<color3>> albedo;
};

std::unique_ptr<Lambertian> create_lambertian(const ConfigArgs &args);
std::unique_ptr<BSDF> create_bsdf(const ConfigArgs &args);

/* Given cosine between rays, return probability density that a photon bounces
 * to that direction. The g parameter controls how different it is from the
 * uniform sphere. g=0 uniform diffuse-like, g=1 close to sharp single ray. */
inline float single_peaked_henyey_greenstein(float cos_theta, float g)
{
    return ((1.0f - g * g) / safe_pow(1.0f + g * g - 2.0f * g * cos_theta, 1.5f)) * (inv_pi * 0.25f);
};

vec3 sample_henyey_greenstein(const vec3 &D, float g, float randu, float randv, float *pdf);

} // namespace ks