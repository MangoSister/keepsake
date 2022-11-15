#pragma once
#include "config.h"
#include "shader_field.h"

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

    vec3 sample(const vec2 &uv) const;
    void apply(Intersection &it) const;

    Space space = Space::TangentSpace;
    bool range01 = true;
    vec3i swizzle = vec3i(0, 1, 2);
    float strength = 1.0f;
    std::unique_ptr<ShaderField<color<3>>> map;
};

std::unique_ptr<NormalMap> create_normal_map(const ConfigArgs &args);
