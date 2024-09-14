#include "bsdf.h"
#include "principled_bsdf.h"
#include "rng.h"

namespace ks
{

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

color3 Lambertian::sample(const vec3 &wo, vec3 &wi, const Intersection &it, float u_lobe, const vec2 &u_wi,
                          float &pdf) const
{
    if (wo.z() <= 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }
    wi = sample_cosine_hemisphere(u_wi);
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
    } else if (bsdf_type == "principled_brdf") {
        bsdf = create_principled_brdf(args);
    } else if (bsdf_type == "principled_bsdf") {
        bsdf = create_principled_bsdf(args);
    }
    return bsdf;
}

vec3 sample_henyey_greenstein(const vec3 &D, float g, float randu, float randv, float *pdf)
{
    /* match pdf for small g */
    float cos_theta;
    bool isotropic = fabsf(g) < 1e-3f;

    if (isotropic) {
        cos_theta = (1.0f - 2.0f * randu);
        if (pdf) {
            *pdf = inv_pi * 0.25f;
        }
    } else {
        float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * randu);
        cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
        if (pdf) {
            *pdf = single_peaked_henyey_greenstein(cos_theta, g);
        }
    }

    float sin_theta = safe_sqrt(1.0f - cos_theta * cos_theta);
    float phi = two_pi * randv;
    vec3 dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    vec3 T, B;
    orthonormal_basis(D, T, B);
    dir = dir.x() * T + dir.y() * B + dir.z() * D;

    return dir;
}

} // namespace ks