#pragma once
#include "config.h"
#include "shader_field.h"

namespace ks
{

struct Texture;
struct TextureSampler;
struct Intersection;

struct NormalMap : Configurable
{
    enum class Space
    {
        TangentSpace,
        WorldSpace
    };

    NormalMap() = default;

    // Normal map should not be simply mipmapped...
    vec3 sample(const vec2 &uv) const;
    void apply(Intersection &it) const;

    Space space = Space::TangentSpace;
    bool range01 = true;
    vec3i swizzle = vec3i(0, 1, 2);
    float strength = 1.0f;
    Transform to_world;
    std::unique_ptr<ShaderField<color<3>>> map;
};

std::unique_ptr<NormalMap> create_normal_map(const ConfigArgs &args);

} // namespace ks