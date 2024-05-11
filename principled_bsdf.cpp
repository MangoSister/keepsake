#include "principled_bsdf.h"
#include "fresnel.h"
#include "rng.h"

namespace ks
{

MicrofacetAdapterDerived<GGX> ggx_adapter;
MicrofacetAdapterDerived<Beckmann> beckmann_adapter;

PrincipledBRDF::Closure PrincipledBRDF::eval_closure(const Intersection &it) const
{
    Closure closure;
    closure.basecolor = (*basecolor)(it);
    closure.basecolor = clamp(closure.basecolor, color3::Zero(), color3::Ones());
    closure.ax = (*roughness)(it)[0];
    closure.ax = clamp(closure.ax, 0.0f, 1.0f);
    closure.ax = sqr(closure.ax);
    closure.ay = closure.ax;
    closure.metallic = (*metallic)(it)[0];
    closure.metallic = clamp(closure.metallic, 0.0f, 1.0f);
    closure.specular = (*specular)(it)[0];
    closure.specular = clamp(closure.specular, 0.0f, 1.0f) * 0.08f;
    if (microfacet == MicrofacetType::GGX) {
        closure.microfacet = &ggx_adapter;
    } else {
        closure.microfacet = &beckmann_adapter;
    }
    return closure;
}

color3 PrincipledBRDF::eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    return internal::eval(wo, wi, closure);
}

color3 PrincipledBRDF::sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const
{
    Closure closure = eval_closure(it);
    return internal::sample(wo, wi, closure, u, pdf);
}

float PrincipledBRDF::pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    return internal::pdf(wo, wi, closure);
}

std::pair<color3, float> PrincipledBRDF::eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    color3 f = internal::eval(wo, wi, closure);
    if (f.isZero()) {
        return {f, 0.0f};
    }
    float pdf = internal::pdf(wo, wi, closure);
    return {f, pdf};
}

color3 PrincipledBRDF::internal::eval(const vec3 &wo, const vec3 &wi, const Closure &closure)
{
    if (wo.z() == 0.0f || wi.z() == 0.0f) {
        return color3::Zero();
    }

    color3 f = color3::Zero();
    if (wo.z() > 0.0f && wi.z() > 0.0f) {
        f += eval_diffuse(wo, wi, closure);
        f += eval_specular(wo, wi, closure);
    }
    return f;
}

color3 PrincipledBRDF::internal::sample(const vec3 &wo, vec3 &wi, const Closure &closure, const vec2 &u, float &pdf)
{
    if (wo.z() == 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }

    vec2 sample_weights = lobe_sample_weights(wo, closure);
    if (sample_weights.isZero()) {
        pdf = 0.0f;
        return color3::Zero();
    }
    float u0_remap;
    int lobe = sample_small_distrib({sample_weights.data(), 2}, u[0], &u0_remap);
    vec2 u_remap(u0_remap, u[1]);

    vec2 pdf_lobe = vec2::Zero();
    if (lobe == 0) {
        wi = sample_diffuse(wo, closure, u_remap, pdf_lobe[0]);
        if (wi.isZero() || pdf_lobe[0] == 0.0f) {
            pdf = 0.0f;
            return color3::Zero();
        }
        if (wo.z() > 0.0f && wi.z() > 0.0f)
            pdf_lobe[1] = pdf_specular(wo, wi, closure);
    } else if (lobe == 1 || pdf_lobe[1] == 0.0f) {
        wi = sample_specular(wo, closure, u_remap, pdf_lobe[1]);
        if (wi.isZero()) {
            pdf = 0.0f;
            return color3::Zero();
        }
        if (wo.z() > 0.0f && wi.z() > 0.0f)
            pdf_lobe[0] = pdf_diffuse(wo, wi, closure);
    }

    pdf = pdf_lobe.dot(sample_weights);
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return eval(wo, wi, closure) / pdf;
}

float PrincipledBRDF::internal::pdf(const vec3 &wo, const vec3 &wi, const Closure &closure)
{
    if (wo.z() == 0.0f || wi.z() == 0.0f) {
        return 0.0f;
    }

    vec2 weights = lobe_sample_weights(wo, closure);
    if (weights.isZero()) {
        return 0.0f;
    }
    if (wo.z() <= 0.0f || wi.z() <= 0.0f) {
        return 0.0f;
    }

    vec2 pdf_lobe = vec2::Zero();
    if (wo.z() > 0.0f && wi.z() > 0.0f) {
        pdf_lobe[0] = pdf_diffuse(wo, wi, closure);
        pdf_lobe[1] = pdf_specular(wo, wi, closure);
    }

    return pdf_lobe.dot(weights);
}

color3 PrincipledBRDF::internal::eval_diffuse(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    float lobe_weight = (1.0f - c.metallic);
    if (lobe_weight == 0.0f) {
        return color3::Zero();
    }
    return lobe_weight * c.basecolor * inv_pi * wi.z();
}

color3 PrincipledBRDF::internal::eval_specular(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    vec3 wh = (wo + wi).normalized();
    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G = c.microfacet->G2(c.ax, c.ay, wo, wi);

    color3 R0 = lerp(color3::Constant(c.specular), c.basecolor, c.metallic);
    color3 Fr = lerp(R0, color3::Ones(), fresnel_schlick(wo.dot(wh)));
    color3 f = D * G * Fr / (4.0f * wo.z());
    ASSERT(f.allFinite() && (f >= 0.0f).all());
    return f;
}

vec2 PrincipledBRDF::internal::lobe_sample_weights(const vec3 &wo, const Closure &c)
{
    float lum_basecolor = luminance(c.basecolor);
    float lum_R0 = std::lerp(c.specular, lum_basecolor, c.metallic);

    float weight_diffuse = wo.z() > 0.0f ? (1.0f - c.metallic) * lum_basecolor * inv_pi : 0.0f;
    // wh isn't available at this point...
    float weight_specular = wo.z() > 0.0f ? std::lerp(lum_R0, 1.0f, fresnel_schlick(wo.z())) : 0.0f;

    float sum = weight_diffuse + weight_specular;
    if (sum == 0.0f) {
        return vec2::Zero();
    }
    float inv_sum = 1.0f / sum;
    weight_diffuse = weight_diffuse * inv_sum;
    weight_specular = weight_specular * inv_sum;

    vec2 weights(weight_diffuse, weight_specular);
    ASSERT(weights.allFinite() && (weights.array() >= 0.0f).all());
    return weights;
}

vec3 PrincipledBRDF::internal::sample_diffuse(const vec3 &wo, const Closure &closure, const vec2 &u, float &pdf)
{
    vec3 wi = sample_cosine_hemisphere(u);
    pdf = wi.z() * inv_pi;
    return wi;
}

vec3 PrincipledBRDF::internal::sample_specular(const vec3 &wo, const Closure &c, const vec2 &u, float &pdf)
{
    if (wo.z() == 0.0f) {
        pdf = 0.0f;
        return vec3::Zero();
    }

    vec3 wh = c.microfacet->sample(c.ax, c.ay, sgn(wo.z()) * wo, u);
    vec3 wi = reflect(wo, wh);
    // side check
    if (wo.z() * wi.z() < 0.0f) {
        pdf = 0.0f;
        return vec3::Zero();
    }

    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);
    pdf = D * G1 / (4.0f * std::abs(wo.z()));
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return wi;
}

float PrincipledBRDF::internal::pdf_diffuse(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    return wi.z() * inv_pi;
}

float PrincipledBRDF::internal::pdf_specular(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    vec3 wh = (wo + wi).normalized();
    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);
    float pdf = D * G1 / (4.0f * std::abs(wo.z()));
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return pdf;
}

std::unique_ptr<PrincipledBRDF> create_principled_brdf(const ConfigArgs &args)
{
    auto bsdf = std::make_unique<PrincipledBRDF>();

    bsdf->basecolor = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["basecolor"]);
    bsdf->roughness = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["roughness"]);
    bsdf->metallic = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["metallic"]);
    bsdf->specular = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["specular"]);
    std::string m = args.load_string("microfacet", "ggx");
    if (m == "ggx") {
        bsdf->microfacet = MicrofacetType::GGX;
    } else if (m == "beckmann") {
        bsdf->microfacet = MicrofacetType::Beckmann;
    }

    return bsdf;
}

////////////////////////////////////////////////////////////////////////

PrincipledBSDF::Closure PrincipledBSDF::eval_closure(const Intersection &it) const
{
    Closure closure;
    closure.basecolor = (*basecolor)(it);
    closure.basecolor = clamp(closure.basecolor, color3::Zero(), color3::Ones());
    closure.ax = (*roughness)(it)[0];
    closure.ax = clamp(closure.ax, 0.0f, 1.0f);
    closure.ax = sqr(closure.ax);
    closure.ay = closure.ax;
    closure.metallic = (*metallic)(it)[0];
    closure.metallic = clamp(closure.metallic, 0.0f, 1.0f);
    closure.ior = (*ior)(it)[0];
    closure.ior = clamp(closure.ior, 1.0f, 4.0f);
    // Remap [0, 1] to [1, 2] (actually allows going above 1 for even more reflection).
    closure.specular_r0_mul = (*specular_r0_mul)(it)[0] * 2.0f;
    closure.specular_r0_mul = std::max(closure.specular_r0_mul, 0.0f);
    closure.specular_trans = (*specular_trans)(it)[0];
    closure.specular_trans = clamp(closure.specular_trans, 0.0f, 1.0f);
    closure.emissive = (*emissive)(it);
    closure.emissive = closure.emissive.cwiseMax(0.0f);
    if (microfacet == MicrofacetType::GGX) {
        closure.microfacet = &ggx_adapter;
    } else {
        closure.microfacet = &beckmann_adapter;
    }
    return closure;
}

color3 PrincipledBSDF::eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    return internal::eval(wo, wi, closure);
}

color3 PrincipledBSDF::sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u, float &pdf) const
{
    Closure closure = eval_closure(it);
    return internal::sample(wo, wi, closure, u, pdf);
}

float PrincipledBSDF::pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    return internal::pdf(wo, wi, closure);
}

std::pair<color3, float> PrincipledBSDF::eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    Closure closure = eval_closure(it);
    color3 f = internal::eval(wo, wi, closure);
    if (f.isZero()) {
        return {f, 0.0f};
    }
    float pdf = internal::pdf(wo, wi, closure);
    return {f, pdf};
}

ks::color3 PrincipledBSDF::internal::eval(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure)
{
    if (wo.z() == 0.0f || wi.z() == 0.0f) {
        return color3::Zero();
    }

    color3 f = color3::Zero();
    if (wo.z() > 0.0f && wi.z() > 0.0f) {
        f += eval_diffuse(wo, wi, closure);
        f += eval_metallic_specular(wo, wi, closure);
    }
    f += eval_dielectric_specular(wo, wi, closure);
    return f;
}

ks::color3 PrincipledBSDF::internal::sample(const ks::vec3 &wo, ks::vec3 &wi, const Closure &closure, const ks::vec2 &u,
                                            float &pdf)
{
    if (wo.z() == 0.0f) {
        pdf = 0.0f;
        return color3::Zero();
    }

    vec3 sample_weights = lobe_sample_weights(wo, closure);
    if (sample_weights.isZero()) {
        pdf = 0.0f;
        return color3::Zero();
    }
    float u0_remap;
    int lobe = sample_small_distrib({sample_weights.data(), 3}, u[0], &u0_remap);
    vec2 u_remap(u0_remap, u[1]);

    vec3 pdf_lobe = vec3::Zero();
    if (lobe == 0) {
        wi = sample_diffuse(wo, closure, u_remap, pdf_lobe[0]);
        if (wi.isZero() || pdf_lobe[0] == 0.0f) {
            pdf = 0.0f;
            return color3::Zero();
        }
        if (wo.z() > 0.0f && wi.z() > 0.0f)
            pdf_lobe[1] = pdf_metallic_specular(wo, wi, closure);
        pdf_lobe[2] = pdf_dielectric_specular(wo, wi, closure);
    } else if (lobe == 1) {
        wi = sample_metallic_specular(wo, closure, u_remap, pdf_lobe[1]);
        if (wi.isZero() || pdf_lobe[1] == 0.0f) {
            pdf = 0.0f;
            return color3::Zero();
        }
        if (wo.z() > 0.0f && wi.z() > 0.0f)
            pdf_lobe[0] = pdf_diffuse(wo, wi, closure);
        pdf_lobe[2] = pdf_dielectric_specular(wo, wi, closure);
    } else {
        wi = sample_dielectric_specular(wo, closure, u_remap, pdf_lobe[2]);
        if (wi.isZero() || pdf_lobe[2] == 0.0f) {
            pdf = 0.0f;
            return color3::Zero();
        }
        if (wo.z() > 0.0f && wi.z() > 0.0f) {
            pdf_lobe[0] = pdf_diffuse(wo, wi, closure);
            pdf_lobe[1] = pdf_metallic_specular(wo, wi, closure);
        }
    }

    pdf = pdf_lobe.dot(sample_weights);
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return eval(wo, wi, closure) / pdf;
}

float PrincipledBSDF::internal::pdf(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure)
{
    if (wo.z() == 0.0f || wi.z() == 0.0f) {
        return 0.0f;
    }

    vec3 weights = lobe_sample_weights(wo, closure);
    if (weights.isZero()) {
        return 0.0f;
    }
    if (wo.z() <= 0.0f || wi.z() <= 0.0f) {
        weights[0] = 0.0f;
        weights[1] = 0.0f;
        weights[2] = 1.0f;
    }

    vec3 pdf_lobe = vec3::Zero();
    if (wo.z() > 0.0f && wi.z() > 0.0f) {
        pdf_lobe[0] = pdf_diffuse(wo, wi, closure);
        pdf_lobe[1] = pdf_metallic_specular(wo, wi, closure);
    }
    pdf_lobe[2] = pdf_dielectric_specular(wo, wi, closure);

    return pdf_lobe.dot(weights);
}

color3 PrincipledBSDF::internal::eval_diffuse(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    float lobe_weight = (1.0f - c.metallic) * (1.0f - c.specular_trans);
    if (lobe_weight == 0.0f) {
        return color3::Zero();
    }
    return lobe_weight * c.basecolor * inv_pi * wi.z();
}

color3 PrincipledBSDF::internal::eval_metallic_specular(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    float lobe_weight = c.metallic;
    if (lobe_weight == 0.0f) {
        return color3::Zero();
    }
    vec3 wh = (wo + wi).normalized();
    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G = c.microfacet->G2(c.ax, c.ay, wo, wi);
    color3 Fr = lerp(c.basecolor, color3::Ones(), fresnel_schlick(wo.dot(wh)));
    color3 f = lobe_weight * D * G * Fr / (4.0f * wo.z());
    ASSERT(f.allFinite() && (f >= 0.0f).all());
    return f;
}

color3 PrincipledBSDF::internal::eval_dielectric_specular(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    float lobe_weight = (1.0f - c.metallic);
    if (lobe_weight == 0.0f) {
        return color3::Zero();
    }

    bool reflect = wo.z() * wi.z() >= 0.0f;
    float eta = wo.z() >= 0.0f ? c.ior : (1.0f / c.ior);
    vec3 wh;
    if (reflect) {
        wh = (wo + wi).normalized();
    } else {
        wh = (wo + wi * eta).normalized();
        lobe_weight *= c.specular_trans;
    }

    if (wh.z() < 0.0f)
        wh = -wh;

    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G = c.microfacet->G2(c.ax, c.ay, wo, wi);
    float abs_wo_dot_wh = std::abs(wo.dot(wh));
    float Fr = fresnel_dielectric(abs_wo_dot_wh, 1.0f / eta);

    color3 f;
    if (reflect) {
        float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh, c);
        f = color3::Constant(D * G * (Fr * Fr_adj) / (4.0f * std::abs(wo.z())));
    } else {
        float denom = sqr(wo.dot(wh) + eta * wi.dot(wh));
        float specular = D * G * (1.0f - Fr) * wo.dot(wh) * wi.dot(wh) / (denom * wo.z());
        specular = std::abs(specular); // Flip all the negative signs.
        // TODO: if we ever need BDPT:
        // https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
        // https://github.com/mitsuba-renderer/mitsuba/blob/master/src/bsdfs/roughdielectric.cpp
        /* Missing term in the original paper: account for the solid angle
           compression when tracing radiance -- this is necessary for
           bidirectional methods */
        // float factor = (mode == TransportMode::Radiance) ? eta : 1;
        // f *= sqr(1.0 / eta) * sqr(factor);
        f = color3::Constant(specular);
    }
    f *= lobe_weight;
    ASSERT(f.allFinite() && (f >= 0.0f).all());
    return f;
}

float PrincipledBSDF::internal::dielectric_specular_adjust(float wo_dot_wh, const Closure &closure)
{
    float R0 = sqr((1.0f - closure.ior) / (1.0f + closure.ior));
    float schlick = fresnel_schlick(wo_dot_wh);
    return std::lerp(closure.specular_r0_mul * R0, 1.0f, schlick) / std::lerp(R0, 1.0f, schlick);
}

vec3 PrincipledBSDF::internal::lobe_sample_weights(const vec3 &wo, const Closure &c)
{
    float lum_basecolor = luminance(c.basecolor);

    float weight_diffuse =
        wo.z() > 0.0f ? (1.0f - c.metallic) * (1.0f - c.specular_trans) * lum_basecolor * inv_pi : 0.0f;
    // wh isn't available at this point...
    float weight_metallic_specular =
        wo.z() > 0.0f ? c.metallic * std::lerp(lum_basecolor, 1.0f, fresnel_schlick(wo.z())) : 0.0f;
    float eta = wo.z() >= 0.0f ? c.ior : (1.0f / c.ior);
    // TODO: incorporate specular_r0_mul here? in general need a better heuristic...
    float weight_dielectric_specular = (1.0f - c.metallic) * fresnel_dielectric(std::abs(wo.z()), 1.0f / eta);

    float sum = weight_diffuse + weight_metallic_specular + weight_dielectric_specular;
    if (sum == 0.0f) {
        return vec3::Zero();
    }
    float inv_sum = 1.0f / sum;
    weight_diffuse = weight_diffuse * inv_sum;
    weight_metallic_specular = weight_metallic_specular * inv_sum;
    weight_dielectric_specular = weight_dielectric_specular * inv_sum;

    vec3 weights(weight_diffuse, weight_metallic_specular, weight_dielectric_specular);
    ASSERT(weights.allFinite() && (weights.array() >= 0.0f).all());
    return weights;
}

vec3 PrincipledBSDF::internal::sample_diffuse(const vec3 &wo, const Closure &closure, const vec2 &u, float &pdf)
{
    vec3 wi = sample_cosine_hemisphere(u);
    pdf = wi.z() * inv_pi;
    return wi;
}

vec3 PrincipledBSDF::internal::sample_metallic_specular(const vec3 &wo, const Closure &c, const vec2 &u, float &pdf)
{
    if (wo.z() == 0.0f) {
        pdf = 0.0f;
        return vec3::Zero();
    }

    vec3 wh = c.microfacet->sample(c.ax, c.ay, sgn(wo.z()) * wo, u);
    vec3 wi = reflect(wo, wh);
    // side check
    if (wo.z() * wi.z() < 0.0f) {
        pdf = 0.0f;
        return vec3::Zero();
    }

    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);
    pdf = D * G1 / (4.0f * std::abs(wo.z()));
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return wi;
}

vec3 PrincipledBSDF::internal::sample_dielectric_specular(const vec3 &wo, const Closure &c, const vec2 &u, float &pdf)
{
    if (wo.z() == 0.0f) {
        pdf = 0.0f;
        return vec3::Zero();
    }

    vec2 u_ndf = demux_float(u[0]);
    float u_fr = u[1];

    vec3 wh = c.microfacet->sample(c.ax, c.ay, sgn(wo.z()) * wo, u_ndf);
    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);

    float eta = wo.z() >= 0.0f ? c.ior : (1.0f / c.ior);
    float abs_wo_dot_wh = std::abs(wo.dot(wh));
    float Fr = fresnel_dielectric(std::abs(wo.dot(wh)), 1.0f / eta);
    float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh, c);
    float Fr_refl = Fr * Fr_adj;
    float Fr_refr = (1.0f - Fr) * c.specular_trans;
    if (Fr_refr == 0.0f) {
        Fr_refl = 1.0f;
    }

    vec3 wi;
    if (u_fr < Fr_refl / (Fr_refl + Fr_refr)) {
        // sample reflection
        wi = reflect(wo, wh);
        // side check
        if (wo.z() * wi.z() < 0.0f) {
            pdf = 0.0f;
            return vec3::Zero();
        }
        pdf = D * G1 / (4.0f * std::abs(wo.z()));
        pdf *= Fr_refl / (Fr_refl + Fr_refr);
        ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    } else {
        // sample refraction
        if (!refract(wo, sgn(wo.z()) * wh, 1.0f / eta, wi)) {
            // total internal reflection
            pdf = 0.0f;
            return vec3::Zero();
        }
        // side check
        if (wo.z() * wi.z() > 0.0f) {
            pdf = 0.0f;
            return vec3::Zero();
        }
        float denom = sqr(wo.dot(wh) + eta * wi.dot(wh));
        float jacobian = eta * eta * std::abs(wi.dot(wh)) / denom;
        pdf = D * G1 * std::abs(wo.dot(wh)) / std::abs(wo.z()) * jacobian;
        pdf *= Fr_refr / (Fr_refl + Fr_refr);
        ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    }

    return wi;
}

float PrincipledBSDF::internal::pdf_diffuse(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    return wi.z() * inv_pi;
}

float PrincipledBSDF::internal::pdf_metallic_specular(const vec3 &wo, const vec3 &wi, const Closure &c)
{
    vec3 wh = (wo + wi).normalized();
    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);
    float pdf = D * G1 / (4.0f * std::abs(wo.z()));
    ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    return pdf;
}

float PrincipledBSDF::internal::pdf_dielectric_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &c)
{
    bool reflect = wo.z() * wi.z() >= 0.0f;
    float eta = wo.z() >= 0.0f ? c.ior : (1.0f / c.ior);
    vec3 wh;
    if (reflect) {
        wh = (wo + wi).normalized();
    } else {
        wh = (wo + wi * eta).normalized();
    }

    if (wh.z() < 0.0f)
        wh = -wh;

    float D = c.microfacet->D(c.ax, c.ay, wh);
    float G1 = c.microfacet->G1(c.ax, c.ay, wo);
    float abs_wo_dot_wh = std::abs(wo.dot(wh));
    float Fr = fresnel_dielectric(std::abs(wo.dot(wh)), 1.0f / eta);
    float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh, c);
    float Fr_refl = Fr * Fr_adj;
    float Fr_refr = (1.0f - Fr) * c.specular_trans;
    if (Fr_refr == 0.0f) {
        Fr_refl = 1.0f;
    }

    float pdf;
    if (reflect) {
        pdf = D * G1 / (4.0f * std::abs(wo.z()));
        pdf *= Fr_refl / (Fr_refl + Fr_refr);
        ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    } else {
        float denom = sqr(wo.dot(wh) + eta * wi.dot(wh));
        float jacobian = eta * eta * std::abs(wi.dot(wh)) / denom;
        pdf = D * G1 * std::abs(wo.dot(wh)) / std::abs(wo.z()) * jacobian;
        pdf *= Fr_refr / (Fr_refl + Fr_refr);
        ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
    }
    return pdf;
}

std::unique_ptr<PrincipledBSDF> create_principled_bsdf(const ConfigArgs &args)
{
    printf("The current implementation of Principled B[S]DF may require some rework!!!\n");

    auto bsdf = std::make_unique<PrincipledBSDF>();

    bsdf->basecolor = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["basecolor"]);
    bsdf->roughness = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["roughness"]);
    bsdf->metallic = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["metallic"]);
    bsdf->ior = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["ior"]);
    bsdf->specular_r0_mul = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["specular_r0_mul"]);
    bsdf->specular_trans = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["specular_trans"]);
    bsdf->emissive = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["emissive"]);

    std::string m = args.load_string("microfacet", "ggx");
    if (m == "ggx") {
        bsdf->microfacet = MicrofacetType::GGX;
    } else if (m == "beckmann") {
        bsdf->microfacet = MicrofacetType::Beckmann;
    }

    return bsdf;
}

} // namespace ks