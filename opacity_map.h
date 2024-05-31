#pragma once
#include "config.h"
#include "shader_field.h"

namespace ks
{

struct Texture;
struct TextureSampler;
struct Intersection;

struct OpacityMap : Configurable
{
    // TODO: should opacity map support mipmapping (sub-pixel correlation)?
    float eval(const vec2 &uv) const;
    bool stochastic_test(const vec2 &uv, float rnd) const;
    // pbrt trick: just hash ray origin and dir to get a random number.
    bool stochastic_test(const vec2 &uv, const vec3 &ro, const vec3 &rd) const;

    std::unique_ptr<ShaderField1> map;
};

std::unique_ptr<OpacityMap> create_opacity_map(const ConfigArgs &args);

} // namespace ks