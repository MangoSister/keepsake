#include "opacity_map.h"

namespace ks
{

float OpacityMap::eval(const vec2 &uv) const { return saturate((*map)(uv, mat2::Zero())[0]); }

bool OpacityMap::stochastic_test(const vec2 &uv, float rnd) const
{
    float opacity = eval(uv);
    if (opacity <= 0.0f) {
        return false;
    } else if (opacity >= 1.0f) {
        return true;
    } else {
        return rnd < opacity;
    }
}

bool OpacityMap::stochastic_test(const vec2 &uv, const vec3 &ro, const vec3 &rd) const
{
    float opacity = eval(uv);
    if (opacity <= 0.0f) {
        return false;
    } else if (opacity >= 1.0f) {
        return true;
    } else {
        float u = hash_float(ro, rd);
        return u < opacity;
    }
}

std::unique_ptr<OpacityMap> create_opacity_map(const ConfigArgs &args)
{
    auto opacity_map = std::make_unique<OpacityMap>();
    opacity_map->map = args.asset_table().create_in_place<ShaderField1>("shader_field_1", args["map"]);
    return opacity_map;
}

} // namespace ks