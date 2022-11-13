#include "material.h"

std::unique_ptr<Material> create_material(const ConfigArgs &args)
{
    std::unique_ptr<Material> material = std::make_unique<Material>();
    material->bsdf = args.asset_table().get<BSDF>(args.load_string("bsdf"));
    material->subsurface = args.asset_table().get<SubsurfaceProfile>(args.load_string("subsurface", ""));
    material->normal_map = args.asset_table().get<NormalMap>(args.load_string("normal_map", ""));

    return material;
}