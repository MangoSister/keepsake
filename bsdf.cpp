#include "bsdf.h"
#include "rng.h"

color3 Lambertian::eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wo.z() <= 0.0f || wi.z() <= 0.0f)
        return color3::Zero();
    return inv_pi * albedo * wi.z();
}

color3 Lambertian::sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const
{
    if (wo.z() <= 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }
    wi = sample_cosine_hemisphere(u);
    pdf = wi.z() * inv_pi;
    return albedo;
}

float Lambertian::pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wo.z() <= 0.0f || wi.z() <= 0.0f)
        return 0.0f;
    return wi.z() * inv_pi;
}

std::unique_ptr<Lambertian> create_lambertian(const ConfigArgs &args)
{
    color3 albedo = args.load_vec3("albedo").array();
    return std::make_unique<Lambertian>(albedo);
}

std::unique_ptr<BSDF> create_bsdf(const ConfigArgs &args)
{
    std::string bsdf_type = args.load_string("type");
    std::unique_ptr<BSDF> bsdf;
    if (bsdf_type == "lambertian") {
        bsdf = create_lambertian(args);
    }
    return bsdf;
}
