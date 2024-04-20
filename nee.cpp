#include "nee.h"
#include "bsdf.h"
#include "light.h"
#include "ray.h"
#include "rng.h"
#include "scene.h"

namespace ks
{

static color3 sample_direct(const Light &light, const BSDF &bsdf, const Intersection &hit, const Scene &geom,
                            const vec3 &wo, const vec2 &u_light, const vec2 &u_bsdf)
{
    color3 Ld = color3::Zero();
    vec3 wo_local = hit.sh_vector_to_local(wo);

    bool delta_bsdf = bsdf.delta();
    bool delta_light = light.delta();

    if (!delta_bsdf) {
        vec3 wi;
        float pdf_light;
        color3 L_beta = light.sample(hit.p, u_light, wi, pdf_light);
        if (pdf_light > 0.0f && !L_beta.isZero()) {
            vec3 wi_local = hit.sh_vector_to_local(wi);
            // Combine these two is in general faster.
            auto [f, pdf_bsdf] = bsdf.eval_and_pdf(wo_local, wi_local, hit);
            if (!f.isZero() && pdf_bsdf > 0.0f) {
                Ray shadow_ray = spawn_ray<OffsetType::NextBounce>(hit.p, wi, hit.frame.n, 0.0f, inf);
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
        color3 f_beta = bsdf.sample(wo_local, wi_local, hit, u_bsdf, pdf_bsdf);
        if (pdf_bsdf > 0.0f && !f_beta.isZero()) {
            vec3 wi = hit.sh_vector_to_world(wi_local);
            color3 L = light.eval(hit.p, wi);
            if (!L.isZero()) {
                Ray shadow_ray = spawn_ray<OffsetType::NextBounce>(hit.p, wi, hit.frame.n, 0.0f, inf);
                if (!geom.occlude1(shadow_ray)) {
                    // float mis = delta_bsdf ? 1.0f : power_heur(pdf_bsdf, pdf_light);
                    float mis = 1.0f;
                    if (!delta_bsdf) {
                        float pdf_light = light.pdf(hit.p, wi);
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

color3 sample_direct(const Scene &scene, std::span<const Light *const> lights, const BSDF &bsdf,
                     const Intersection &hit, const vec3 &wo, RNG &rng)
{
    color3 Ld = color3::Zero();
    for (int i = 0; i < lights.size(); ++i) {
        vec2 u_light = rng.next2d();
        vec2 u_bsdf = rng.next2d();
        Ld += sample_direct(*lights[i], bsdf, hit, scene, wo, u_light, u_bsdf);
    }
    return Ld;
}

} // namespace ks