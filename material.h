#pragma once
#include "bsdf.h"
#include "config.h"

// TODO: This is pretty much a placeholder for now

struct SubsurfaceProfile : public Configurable
{
};

struct NormalMap : public Configurable
{
};

struct Material : public Configurable
{
    const BSDF *bsdf = nullptr;
    const SubsurfaceProfile *subsurface = nullptr;
    const NormalMap *normal_map = nullptr;
};

std::unique_ptr<Material> create_material(const ConfigArgs &args);