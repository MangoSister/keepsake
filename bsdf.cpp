#include "bsdf.h"
#include "rng.h"

std::pair<color3, float> BSDF::eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    return {eval(wo, wi, it), pdf(wo, wi, it)};
}

color3 Lambertian::eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wo.z() <= 0.0f || wi.z() <= 0.0f)
        return color3::Zero();
    color3 albedo_color = (*albedo)(it);
    albedo_color = clamp(albedo_color, color3::Zero(), color3::Ones());
    return inv_pi * albedo_color * wi.z();
}

color3 Lambertian::sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const
{
    if (wo.z() <= 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }
    wi = sample_cosine_hemisphere(u);
    pdf = wi.z() * inv_pi;
    color3 albedo_color = (*albedo)(it);
    albedo_color = clamp(albedo_color, color3::Zero(), color3::Ones());
    return albedo_color;
}

float Lambertian::pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wo.z() <= 0.0f || wi.z() <= 0.0f)
        return 0.0f;
    return wi.z() * inv_pi;
}

std::unique_ptr<Lambertian> create_lambertian(const ConfigArgs &args)
{
    std::unique_ptr<ShaderField3> albedo =
        args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["albedo"]);
    return std::make_unique<Lambertian>(std::move(albedo));
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