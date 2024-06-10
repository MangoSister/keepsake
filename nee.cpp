#include "nee.h"
#include "bsdf.h"
#include "light.h"
#include "ray.h"
#include "scene.h"
#include "sobol.h"

namespace ks
{

static color3 sample_direct(const Light &light, const BSDF &bsdf, const Intersection &hit, const Scene &geom,
                            const vec3 &wo, const vec2 &u_light, float u_bsdf_lobe, const vec2 &u_bsdf_wi)
{
    color3 Ld = color3::Zero();
    vec3 wo_local = hit.sh_vector_to_local(wo);

    bool delta_bsdf = bsdf.delta();
    bool delta_light = light.delta();

    if (!delta_bsdf) {
        vec3 wi;
        float wi_dist;
        float pdf_light;
        color3 L_beta = light.sample(hit.p, u_light, wi, wi_dist, pdf_light);
        if (pdf_light > 0.0f && !L_beta.isZero()) {
            vec3 wi_local = hit.sh_vector_to_local(wi);
            // Combine these two is in general faster.
            auto [f, pdf_bsdf] = bsdf.eval_and_pdf(wo_local, wi_local, hit);
            if (!f.isZero() && pdf_bsdf > 0.0f) {
                Ray shadow_ray = spawn_ray<OffsetType::NextBounce>(hit.p, wi, hit.frame.n, 0.0f, wi_dist);
                if (!geom.occlude1(shadow_ray)) {
                    float mis = 1.0f;
                    if (!delta_light) {
                        // float pdf_bsdf = bsdf.pdf(wo_local, wi_local, hit);
                        mis = power_heur(pdf_light, pdf_bsdf);
                    }
                    Ld += f * L_beta * mis;
                    ASSERT(Ld.allFinite() && (Ld >= 0.0f).all());
                }
            }
        }
    }
    if (!delta_light) {
        vec3 wi_local;
        float pdf_bsdf;
        color3 f_beta = bsdf.sample(wo_local, wi_local, hit, u_bsdf_lobe, u_bsdf_wi, pdf_bsdf);
        if (pdf_bsdf > 0.0f && !f_beta.isZero()) {
            vec3 wi = hit.sh_vector_to_world(wi_local);
            float wi_dist;
            color3 L = light.eval(hit.p, wi, wi_dist);
            if (!L.isZero()) {
                Ray shadow_ray = spawn_ray<OffsetType::NextBounce>(hit.p, wi, hit.frame.n, 0.0f, wi_dist);
                if (!geom.occlude1(shadow_ray)) {
                    // float mis = delta_bsdf ? 1.0f : power_heur(pdf_bsdf, pdf_light);
                    float mis = 1.0f;
                    if (!delta_bsdf) {
                        float pdf_light = light.pdf(hit.p, wi, wi_dist);
                        mis = power_heur(pdf_bsdf, pdf_light);
                    }
                    Ld += f_beta * L * mis;
                    ASSERT(Ld.allFinite() && (Ld >= 0.0f).all());
                }
            }
        }
    }

    return Ld;
}

color3 next_event_estimate(const Scene &scene, const LightSampler &light_sampler, const BSDF &bsdf,
                           const Intersection &hit, const vec3 &wo, PTRenderSampler &sampler)
{
    // color3 Ld = color3::Zero();
    // for (int i = 0; i < lights.size(); ++i) {
    //     vec2 u_light = rng.next2d();
    //     vec2 u_bsdf = rng.next2d();
    //     Ld += sample_direct(*lights[i], bsdf, hit, scene, wo, u_light, u_bsdf);
    // }
    // return Ld;

    bool delta_bsdf = bsdf.delta();
    if (delta_bsdf) {
        return color3::Zero();
    }

    float pr_light;
    auto [_, light] = light_sampler.sample(sampler.sobol.next(), pr_light);
    bool delta_light = light->delta();

    vec3 wi;
    float wi_dist;
    float pdf_wi_light;
    vec2 u_light = sampler.sobol.next2d();
    color3 Le_beta = light->sample(hit.p, u_light, wi, wi_dist, pdf_wi_light);

    color3 beta_nee = color3::Zero();
    if (!Le_beta.isZero() && pdf_wi_light > 0.0f) {
        vec3 wo_local = hit.sh_vector_to_local(wo);
        vec3 wi_local = hit.sh_vector_to_local(wi);
        // Combine these two is in general faster.
        auto [f, pdf_wi_bsdf] = bsdf.eval_and_pdf(wo_local, wi_local, hit);
        if (!f.isZero() && pdf_wi_bsdf > 0.0f) {
            Ray shadow_ray = spawn_ray<OffsetType::NextBounce>(hit.p, wi, hit.frame.n, 0.0f, wi_dist);
            if (!scene.occlude1(shadow_ray)) {
                float mis_weight = 1.0f;
                if (!delta_light) {
                    // float pdf_bsdf = bsdf.pdf(wo_local, wi_local, hit);
                    mis_weight = power_heur(pdf_wi_light, pdf_wi_bsdf);
                }
                beta_nee += Le_beta * f * mis_weight / pr_light;
                ASSERT(beta_nee.allFinite() && (beta_nee >= 0.0f).all());
            }
        }
    }

    return beta_nee;
}

} // namespace ks