#pragma once
#include "config.h"
#include <span>

namespace ks
{

struct BSDF;
struct BSSRDF;
struct NormalMap;
struct RNG;
struct Intersection;
struct Scene;
struct LocalGeometry;
struct Light;
struct LambertianSubsurfaceExitAdapter;

struct MaterialSample
{
    color3 Ld = color3::Zero();
    color3 beta = color3::Ones();
};

struct Material : public Configurable
{
    virtual color3 sample(vec3 wo, const Intersection &entry, const Scene &scene, const LocalGeometry &local_geom,
                          RNG &rng, vec3 &wi, Intersection &exit) const = 0;

    virtual MaterialSample sample_with_direct(vec3 wo, const Intersection &entry, const Scene &scene,
                                              const LocalGeometry &local_geom, std::span<const Light *const> lights,
                                              RNG &rng, vec3 &wi, Intersection &exit) const = 0;

    const BSDF *bsdf = nullptr;
    const BSSRDF *subsurface = nullptr;
    const NormalMap *normal_map = nullptr;
};

struct BlendedMaterial : public Material
{
    BlendedMaterial();
    ~BlendedMaterial();

    color3 sample(vec3 wo, const Intersection &entry, const Scene &scene, const LocalGeometry &local_geom, RNG &rng,
                  vec3 &wi, Intersection &exit) const;

    MaterialSample sample_with_direct(vec3 wo, const Intersection &entry, const Scene &scene,
                                      const LocalGeometry &local_geom, std::span<const Light *const> lights, RNG &rng,
                                      vec3 &wi, Intersection &exit) const;

    std::unique_ptr<LambertianSubsurfaceExitAdapter> lambert_exit;
};

struct StackedMaterial : public Material
{
    color3 sample(vec3 wo, const Intersection &entry, const Scene &scene, const LocalGeometry &local_geom, RNG &rng,
                  vec3 &wi, Intersection &exit) const;

    MaterialSample sample_with_direct(vec3 wo, const Intersection &entry, const Scene &scene,
                                      const LocalGeometry &local_geom, std::span<const Light *const> lights, RNG &rng,
                                      vec3 &wi, Intersection &exit) const;
};

std::unique_ptr<Material> create_material(const ConfigArgs &args);

} // namespace ks