#include "material.h"
#include "nee.h"
#include "normal_map.h"
#include "rng.h"
#include "scene.h"
#include "subsurface.h"

namespace ks
{

inline float fresnel_dielectric_cos(float cosi, float eta)
{
    // compute fresnel reflectance without explicitly computing
    // the refracted direction
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f; // TIR(no refracted component)
}

/* Calculate the fresnel color which is a blend between white and the F0 color (cspec0) */
inline color3 interpolate_fresnel_color(const vec3 &L, const vec3 &H, float ior, float F0, const color3 &cspec0)
{
    /* Calculate the fresnel interpolation factor
     * The value from fresnel_dielectric_cos(...) has to be normalized because
     * the cspec0 keeps the F0 color
     */
    float F0_norm = 1.0f / (1.0f - F0);
    float FH = (fresnel_dielectric_cos(L.dot(H), ior) - F0) * F0_norm;

    /* Blend between white and a specular color with respect to the fresnel */
    return cspec0 * (1.0f - FH) + color3::Constant(FH);
}

BlendedMaterial::BlendedMaterial() = default;

BlendedMaterial::~BlendedMaterial() = default;

color3 BlendedMaterial::sample(vec3 wo, const Intersection &entry, const Scene &scene, const LocalGeometry &local_geom,
                               RNG &rng, vec3 &wi, Intersection &exit) const
{
    color3 beta = color3::Ones();

    // TODO: currently select between BSDF and BSSRDF uniformly...
    float bssrdf_sample_weight = 1.0f;
    float bsdf_sample_weight = 1.0f;
    float sum = bssrdf_sample_weight + bsdf_sample_weight;
    //

    bool sample_subsurface = false;
    if (subsurface) {
        if (rng.next() < (bsdf_sample_weight / sum)) {
            beta *= (sum / bsdf_sample_weight);
        } else {
            sample_subsurface = true;
            beta *= (sum / bssrdf_sample_weight);
            // NOTE: need to adjust radiance scaling since the ray gets "refracted" in
            // The exit bsdf must also revert this when the ray exiting.
            beta /= sqr(subsurface->ior);
        }
    }

    const BSDF *exit_bsdf = nullptr;
    if (sample_subsurface) {
        vec3 ss_wi;
        SceneHit exit_hit;
        if (!subsurface->sample(local_geom, entry, vec3::Zero(), rng, beta, exit_hit, ss_wi)) {
            return color3::Zero();
        }
        wo = -ss_wi;
        exit = exit_hit.it;
        if (lambert_exit) {
            exit_bsdf = lambert_exit.get();
        } else
            exit_bsdf = exit_hit.material->bsdf;
    } else {
        exit = entry;
        exit_bsdf = bsdf;
    }

    vec3 wo_local = exit.sh_vector_to_local(wo);
    vec3 wi_local;
    float pdf;
    beta *= exit_bsdf->sample(wo_local, wi_local, exit, rng.next2d(), pdf);
    if (beta.maxCoeff() == 0.0f || pdf == 0.0f) {
        return color3::Zero();
    }
    ASSERT(beta.allFinite() && (beta >= 0.0f).all());
    wi = exit.sh_vector_to_world(wi_local);

    return beta;
}

MaterialSample BlendedMaterial::sample_with_direct(vec3 wo, const Intersection &entry, const Scene &scene,
                                                   const LocalGeometry &local_geom,
                                                   std::span<const Light *const> lights, RNG &rng, vec3 &wi,
                                                   Intersection &exit) const
{
    MaterialSample s;

    // Heuristic loosely from Blender (Monaco)
    // float bsdf_sample_weight = 1.0f;
    // color3 fc = interpolate_fresnel_color(wo, entry.sh_frame.n, 1.5f, 0.04f, color3::Constant(0.04f));
    // bsdf_sample_weight *= fc.x();
    // float bssrdf_sample_weight = 3.0f;
    // float sum = bssrdf_sample_weight + bsdf_sample_weight;
    // bsdf_sample_weight = std::max(bsdf_sample_weight, 0.125f * sum);
    // sum = bssrdf_sample_weight + bsdf_sample_weight;

    // TODO: currently select between BSDF and BSSRDF uniformly...
    float bssrdf_sample_weight = 1.0f;
    float bsdf_sample_weight = 1.0f;
    float sum = bssrdf_sample_weight + bsdf_sample_weight;
    //

    bool sample_subsurface = false;
    if (subsurface) {
        if (rng.next() < (bsdf_sample_weight / sum)) {
            s.beta *= (sum / bsdf_sample_weight);
        } else {
            sample_subsurface = true;
            s.beta *= (sum / bssrdf_sample_weight);
            // NOTE: need to adjust radiance scaling since the ray gets "refracted" in
            // The exit bsdf must also revert this when the ray exiting.
            s.beta /= sqr(subsurface->ior);
        }
    }

    const BSDF *exit_bsdf = nullptr;
    if (sample_subsurface) {
        vec3 ss_wi;
        SceneHit exit_hit;
        if (!subsurface->sample(local_geom, entry, vec3::Zero(), rng, s.beta, exit_hit, ss_wi)) {
            return {color3::Zero(), color3::Zero()};
        }
        wo = -ss_wi;
        exit = exit_hit.it;
        if (lambert_exit) {
            exit_bsdf = lambert_exit.get();
        } else
            exit_bsdf = exit_hit.material->bsdf;
    } else {
        exit = entry;
        exit_bsdf = bsdf;
    }

    s.Ld = s.beta * sample_direct(scene, lights, *exit_bsdf, exit, wo, rng);
    vec3 wo_local = exit.sh_vector_to_local(wo);
    vec3 wi_local;
    float pdf;
    s.beta *= exit_bsdf->sample(wo_local, wi_local, exit, rng.next2d(), pdf);
    ASSERT(s.beta.allFinite() && (s.beta >= 0.0f).all());
    if (s.beta.maxCoeff() == 0.0f || pdf == 0.0f) {
        return s;
    }
    wi = exit.sh_vector_to_world(wi_local);

    return s;
}

std::unique_ptr<BlendedMaterial> create_blended_material(const ConfigArgs &args)
{
    std::unique_ptr<BlendedMaterial> material = std::make_unique<BlendedMaterial>();
    material->bsdf = args.asset_table().get<BSDF>(args.load_string("bsdf"));
    material->subsurface = args.asset_table().get<BSSRDF>(args.load_string("subsurface", ""));
    material->normal_map = args.asset_table().get<NormalMap>(args.load_string("normal_map", ""));
    if (material->subsurface)
        material->lambert_exit = std::make_unique<LambertianSubsurfaceExitAdapter>(material->subsurface->ior);
    return material;
}

color3 StackedMaterial::sample(vec3 wo, const Intersection &entry, const Scene &scene, const LocalGeometry &local_geom,
                               RNG &rng, vec3 &wi, Intersection &exit) const
{
    color3 beta = color3::Ones();
    // (assume radiance scaling due to refractive index handled in bsdf)
    // 1. sample surface at entry
    vec3 entry_wo_local = entry.sh_vector_to_local(wo);
    vec3 entry_wi_local;
    float pdf;
    beta *= bsdf->sample(entry_wo_local, entry_wi_local, entry, rng.next2d(), pdf);
    if (beta.maxCoeff() == 0.0f || pdf == 0.0f) {
        return color3::Zero();
    }
    ASSERT(beta.allFinite() && (beta >= 0.0f).all());
    wi = entry.sh_vector_to_world(entry_wi_local);

    // 2. if refracting in, sample BSSRDF
    if (entry_wo_local.z() > 0.0f && entry_wi_local.z() < 0.0f) {
        vec3 D = entry.sh_vector_to_world(entry_wi_local);
        vec3 ss_wi;
        SceneHit exit_hit;
        if (!subsurface->sample(local_geom, entry, D, rng, beta, exit_hit, ss_wi)) {
            return color3::Zero();
        }
        exit = exit_hit.it;
        const BSDF *exit_bsdf = exit_hit.material->bsdf;

        // 3. sample surface at exit
        vec3 exit_wo_local = exit.sh_vector_to_local(-ss_wi);
        vec3 exit_wi_local;
        beta *= exit_bsdf->sample(exit_wo_local, exit_wi_local, exit, rng.next2d(), pdf);
        if (beta.maxCoeff() == 0.0f || pdf == 0.0f) {
            return color3::Zero();
        }
        ASSERT(beta.allFinite() && (beta >= 0.0f).all());
        wi = exit.sh_vector_to_world(exit_wi_local);
    }
    return beta;
}

MaterialSample StackedMaterial::sample_with_direct(vec3 wo, const Intersection &entry, const Scene &scene,
                                                   const LocalGeometry &local_geom,
                                                   std::span<const Light *const> lights, RNG &rng, vec3 &wi,
                                                   Intersection &exit) const
{
    MaterialSample s;

    // (assume radiance scaling due to refractive index handled in bsdf)
    // 1.1. nee at entry
    s.Ld += sample_direct(scene, lights, *bsdf, entry, wo, rng);
    // 1.2. sample surface at entry
    vec3 entry_wo_local = entry.sh_vector_to_local(wo);
    vec3 entry_wi_local;
    float pdf;
    s.beta *= bsdf->sample(entry_wo_local, entry_wi_local, entry, rng.next2d(), pdf);
    if (s.beta.maxCoeff() == 0.0f || pdf == 0.0f) {
        return {color3::Zero(), color3::Zero()};
    }
    ASSERT(s.beta.allFinite() && (s.beta >= 0.0f).all());
    wi = entry.sh_vector_to_world(entry_wi_local);

    // 2. if refracting in, sample BSSRDF
    if (entry_wo_local.z() > 0.0f && entry_wi_local.z() < 0.0f) {
        vec3 D = wi;
        vec3 ss_wi;
        SceneHit exit_hit;
        if (!subsurface->sample(local_geom, entry, D, rng, s.beta, exit_hit, ss_wi)) {
            return {color3::Zero(), color3::Zero()};
        }
        exit = exit_hit.it;
        const BSDF *exit_bsdf = exit_hit.material->bsdf;

        // 3.1 nee at exit
        vec3 exit_wo = -ss_wi;
        s.Ld += s.beta * sample_direct(scene, lights, *exit_bsdf, exit, exit_wo, rng);
        // 3.2 sample surface at exit
        vec3 exit_wo_local = exit.sh_vector_to_local(exit_wo);
        vec3 exit_wi_local;
        s.beta *= exit_bsdf->sample(exit_wo_local, exit_wi_local, exit, rng.next2d(), pdf);
        if (s.beta.maxCoeff() == 0.0f || pdf == 0.0f) {
            return {color3::Zero(), color3::Zero()};
        }
        ASSERT(s.beta.allFinite() && (s.beta >= 0.0f).all());
        wi = exit.sh_vector_to_world(exit_wi_local);
    }
    return s;
}

std::unique_ptr<StackedMaterial> create_stacked_material(const ConfigArgs &args)
{
    std::unique_ptr<StackedMaterial> material = std::make_unique<StackedMaterial>();
    material->bsdf = args.asset_table().get<BSDF>(args.load_string("bsdf"));
    material->subsurface = args.asset_table().get<BSSRDF>(args.load_string("subsurface", ""));
    material->normal_map = args.asset_table().get<NormalMap>(args.load_string("normal_map", ""));
    return material;
}

std::unique_ptr<Material> create_material(const ConfigArgs &args)
{
    std::string type = args.load_string("type", "blended");
    std::unique_ptr<Material> material;
    if (type == "blended") {
        material = create_blended_material(args);
    } else if (type == "stacked") {
        material = create_stacked_material(args);
    }

    return material;
}

} // namespace ks