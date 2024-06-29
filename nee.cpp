#include "nee.h"
#include "bsdf.h"
#include "light.h"
#include "ray.h"
#include "scene.h"
#include "sobol.h"

namespace ks
{

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