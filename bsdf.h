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
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    virtual color3 sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const = 0;
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
    color3 sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const;
    float pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const;

    std::unique_ptr<ShaderField<color3>> albedo;
};

std::unique_ptr<Lambertian> create_lambertian(const ConfigArgs &args);
std::unique_ptr<BSDF> create_bsdf(const ConfigArgs &args);

} // namespace ks