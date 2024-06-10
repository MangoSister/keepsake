// Based on Blender random walk SSS implementation
// And Monaco internal implementation (refraction entry)
// Blender repo: intern/cycles/kernel/integrator/subsurface_random_walk.h

#pragma once

#include "bsdf.h"
#include "config.h"
#include "maths.h"

namespace ks
{

struct Scene;
struct LocalGeometry;
struct SceneHit;
struct RNG;

struct LambertianSubsurfaceExitAdapter : public BSDF
{
    LambertianSubsurfaceExitAdapter() = default;
    LambertianSubsurfaceExitAdapter(float eta) : eta(eta){};

    bool delta() const { return false; }
    // NOTE: return cosine-weighted bsdf: f*cos(theta_i)
    color3 eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const;
    // NOTE: return cosine-weighted throughput weight: (f*cos(theta_i) / pdf)
    color3 sample(const vec3 &wo, vec3 &wi, const Intersection &it, float u_lobe, const vec2 &u_wi, float &pdf) const;
    float pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const;

    float eta;
};

struct BSSRDF : Configurable
{
    bool sample(const LocalGeometry &local_geometry, const Intersection &entry, vec3 D, RNG &rng, color3 &throughput,
                SceneHit &exit, vec3 &wi) const;

    std::unique_ptr<ShaderField<color3>> albedo;
    std::unique_ptr<ShaderField<color3>> radius;
    float anisotropy;
    float ior; // Should keep consistent with the surface BSDF?
    float rfr_entry_prob;
};

std::unique_ptr<BSSRDF> create_bssrdf(const ConfigArgs &args);

} // namespace ks