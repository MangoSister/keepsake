#pragma once
#include "barray.h"
#include "config.h"
#include "distrib.h"
#include "maths.h"
#include <filesystem>
namespace fs = std::filesystem;

struct Light
{
    virtual ~Light() = default;
    virtual bool delta_position() const = 0;
    virtual bool delta_direction() const = 0;
    bool delta() const { return delta_direction() || delta_position(); };

    // NOTE: the point passed in is the shading point
    virtual color3 eval(const vec3 &p_shade, const vec3 &wi) const = 0;
    // NOTE: return throughput weight: (L / pdf)
    virtual color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &pdf) const = 0;
    virtual float pdf(const vec3 &p_shade, const vec3 &wi) const = 0;
};

struct SkyLight : public Light
{
    SkyLight(const fs::path &path, const Transform &l2w, float strength = 1.0f);
    bool delta_position() const { return false; };
    bool delta_direction() const { return false; };

    // NOTE: the shading point is ignored for skylight
    color3 eval(const vec3 &p_shade, const vec3 &wi) const;
    // NOTE: return throughput weight: (L / pdf)
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi) const;

    DistribTable2D distrib;
    BlockedArray<color3> map;
    Transform l2w;
    float strength = 1.0f;
};

struct DirectionalLight : public Light
{
    DirectionalLight() = default;
    DirectionalLight(const color3 &L, const vec3 &dir) : L(L), dir(dir){};

    bool delta_position() const { return false; };
    bool delta_direction() const { return true; };
    color3 eval(const vec3 &p_shade, const vec3 &wi) const;
    // NOTE: return throughput weight: (L / pdf)
    color3 sample(const vec3 &p_shade, const vec2 &u, vec3 &wi, float &pdf) const;
    float pdf(const vec3 &p_shade, const vec3 &wi) const;

    color3 L;
    vec3 dir;
};

std::unique_ptr<Light> create_light(const ConfigArgs &args);
std::unique_ptr<SkyLight> create_sky_light(const ConfigArgs &args);
std::unique_ptr<DirectionalLight> create_directional_light(const ConfigArgs &args);