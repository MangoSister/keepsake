#pragma once
#include "maths.h"
#include <span>

KS_NAMESPACE_BEGIN

struct Light;
struct Intersection;
struct BSDF;
struct RNG;
struct Scene;

color3 sample_direct(const Scene &scene, std::span<const Light *const> lights, const BSDF &bsdf,
                     const Intersection &hit, const vec3 &wo, RNG &rng);

KS_NAMESPACE_END