#pragma once
#include "bsdf.h"
#include "config.h"
#include "microfacet.h"
#include "shader_field.h"

namespace ks
{

enum class MicrofacetType
{
    GGX,
    Beckmann,
};

extern MicrofacetAdapterDerived<GGX> ggx_adapter;
extern MicrofacetAdapterDerived<Beckmann> beckmann_adapter;

struct PrincipledBRDF : public ks::BSDF
{
    bool delta() const { return false; }

    struct Closure
    {
        ks::color3 basecolor;
        float ax;
        float ay;
        float metallic;
        float specular;
        const ks::MicrofacetAdapter *microfacet;
    };
    Closure eval_closure(const ks::Intersection &it) const;

    // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
    ks::color3 eval(const ks::vec3 &wo, const ks::vec3 &wi, const ks::Intersection &it) const;
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    ks::color3 sample(const ks::vec3 &wo, ks::vec3 &wi, const ks::Intersection &it, float u_lobe, const ks::vec2 &u_wi,
                      float &pdf) const;
    float pdf(const ks::vec3 &wo, const ks::vec3 &wi, const ks::Intersection &it) const;
    std::pair<color3, float> eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const override;

    struct internal
    {
        // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
        static ks::color3 eval(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
        static ks::color3 sample(const ks::vec3 &wo, ks::vec3 &wi, const Closure &closure, float u_lobe,
                                 const ks::vec2 &u_wi, float &pdf);
        static float pdf(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);

        static ks::color3 eval_diffuse(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static ks::color3 eval_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);

        static ks::vec2 lobe_sample_weights(const ks::vec3 &wo, const Closure &closure);
        static ks::vec3 sample_diffuse(const ks::vec3 &wo, const Closure &closure, const ks::vec2 &u, float &pdf);
        static ks::vec3 sample_specular(const ks::vec3 &wo, const Closure &closure, const ks::vec2 &u, float &pdf);

        static float pdf_diffuse(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static float pdf_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
    };

    std::unique_ptr<ks::ShaderField3> basecolor;
    std::unique_ptr<ks::ShaderField1> roughness;
    std::unique_ptr<ks::ShaderField1> metallic;
    std::unique_ptr<ks::ShaderField1> specular;

    MicrofacetType microfacet;
};

std::unique_ptr<PrincipledBRDF> create_principled_brdf(const ks::ConfigArgs &args);

////////////////////////////////////////////////////////////////////////

struct PrincipledBSDF : public ks::BSDF
{
    bool delta() const { return false; }

    struct Closure
    {
        ks::color3 basecolor;
        float roughness;
        float ax;
        float ay;
        float metallic;
        float ior;
        float specular_r0_mul;
        float specular_trans;
        float diffuse_trans;
        float diffuse_trans_fwd;
        ks::color3 emissive;

        const ks::MicrofacetAdapter *microfacet;
    };
    Closure eval_closure(const ks::Intersection &it) const;

    // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
    ks::color3 eval(const ks::vec3 &wo, const ks::vec3 &wi, const ks::Intersection &it) const;
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    ks::color3 sample(const ks::vec3 &wo, ks::vec3 &wi, const ks::Intersection &it, float u_lobe, const ks::vec2 &u,
                      float &pdf) const;
    float pdf(const ks::vec3 &wo, const ks::vec3 &wi, const ks::Intersection &it) const;
    std::pair<color3, float> eval_and_pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const override;

    struct internal
    {
        // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
        static ks::color3 eval(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
        static ks::color3 sample(const ks::vec3 &wo, ks::vec3 &wi, const Closure &closure, float u_lobe,
                                 const ks::vec2 &u_wi, float &pdf);
        static float pdf(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);

        static ks::color3 eval_diffuse(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static ks::color3 eval_diffuse_transmission(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static ks::color3 eval_metallic_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static ks::color3 eval_dielectric_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static float dielectric_specular_adjust(float wo_dot_wh, const Closure &closure);
        static ks::vec4 lobe_sample_weights(const ks::vec3 &wo, const Closure &closure);
        static ks::vec3 sample_diffuse(const Closure &closure, const ks::vec2 &u, float &pdf);
        static ks::vec3 sample_diffuse_transmission(const ks::vec3 &wo, const Closure &closure, const ks::vec2 &u,
                                                    float &pdf);
        static ks::vec3 sample_metallic_specular(const ks::vec3 &wo, const Closure &closure, const ks::vec2 &u,
                                                 float &pdf);
        static ks::vec3 sample_dielectric_specular(const ks::vec3 &wo, const Closure &closure, float u_lobe,
                                                   const ks::vec2 &u_wi, float &pdf);

        static float pdf_diffuse(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static float pdf_diffuse_transmission(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static float pdf_metallic_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
        static float pdf_dielectric_specular(const ks::vec3 &wo, const ks::vec3 &wi, const Closure &closure);
    };

    std::unique_ptr<ks::ShaderField3> basecolor;
    std::unique_ptr<ks::ShaderField1> roughness;
    std::unique_ptr<ks::ShaderField1> metallic;
    std::unique_ptr<ks::ShaderField1> ior;
    std::unique_ptr<ks::ShaderField1> specular_r0_mul;
    std::unique_ptr<ks::ShaderField1> specular_trans;
    // diffuse_trans transfers energy from the diffuse reflection lobe to the diffuse transmission lobe;
    std::unique_ptr<ks::ShaderField1> diffuse_trans;
    // HG lobe g parameter but forward only ([0, 1]).
    std::unique_ptr<ks::ShaderField1> diffuse_trans_fwd;
    // TODO: make emission at material level, not bsdf level.
    std::unique_ptr<ks::ShaderField3> emissive;

    MicrofacetType microfacet;
};

std::unique_ptr<PrincipledBSDF> create_principled_bsdf(const ks::ConfigArgs &args);

} // namespace ks