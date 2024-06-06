#pragma once
#include "maths.h"
#include <span>

namespace ks
{

struct BSDF;
struct Scene;
struct LightSampler;
struct Intersection;
struct PTRenderSampler;

color3 next_event_estimate(const Scene &scene, const LightSampler &light_sampler, const BSDF &bsdf,
                           const Intersection &hit, const vec3 &wo, PTRenderSampler &sampler);

template <int n = 2>
inline float power_heur(float pf, float pg)
{
    float pf_n = pow<n>(pf);
    float pg_n = pow<n>(pg);
    return pf_n / (pf_n + pg_n);
}

} // namespace ks