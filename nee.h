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

} // namespace ks