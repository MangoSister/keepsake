#pragma once
#include "maths.h"
#include <span>

namespace ks
{

struct Light;
struct Intersection;
struct BSDF;
struct RNG;
struct Scene;

color3 sample_direct(const Scene &scene, std::span<const Light *const> lights, const BSDF &bsdf,
                     const Intersection &hit, const vec3 &wo, RNG &rng);

template <int n = 2>
inline float power_heur(float pf, float pg)
{
    float pf_n = pow<n>(pf);
    float pg_n = pow<n>(pg);
    return pf_n / (pf_n + pg_n);
}

} // namespace ks