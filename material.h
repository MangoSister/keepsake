#pragma once
#include "config.h"

struct BSDF;
struct BSSRDF;
struct NormalMap;

struct Material : public Configurable
{
    const BSDF *bsdf = nullptr;
    const BSSRDF *subsurface = nullptr;
    const NormalMap *normal_map = nullptr;
};

std::unique_ptr<Material> create_material(const ConfigArgs &args);