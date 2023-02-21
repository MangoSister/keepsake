#include "normal_map.h"
#include "assertion.h"
#include "ray.h"
#include "texture.h"

namespace ks
{

vec3 NormalMap::sample(const vec2 &uv) const
{
    color3 sample = (*map)(uv, mat2::Zero()).head(3);
    if (range01) {
        sample = sample * 2.0f - 1.0f;
    }
    vec3 normal = sample.matrix().normalized();
    if (space == Space::TangentSpace && strength < 1.0f) {
        normal = lerp(vec3(0.0f, 0.0f, 1.0f), normal, strength);
        normal.normalize();
    }

    return normal;
}

void NormalMap::apply(Intersection &it) const
{
    vec3 normal = sample(it.uv);
    vec3 n_perturb;
    if (space == Space::TangentSpace) {
        n_perturb = it.sh_frame.to_world(normal);
    } else if (space == Space::WorldSpace) {
        n_perturb = to_world.normal(normal);
    }
    vec3 bump_deriv = -normal / normal.z();
    vec3 t_perturb = (it.sh_frame.t + bump_deriv.x() * it.sh_frame.n).normalized();
    // re-orthogonalize (n_perturb is respected).
    vec3 b_perturb = n_perturb.cross(t_perturb).normalized();
    t_perturb = b_perturb.cross(n_perturb).normalized();
    it.sh_frame = Frame(t_perturb, b_perturb, n_perturb);
}

std::unique_ptr<NormalMap> create_normal_map(const ConfigArgs &args)
{
    auto normal_map = std::make_unique<NormalMap>();
    normal_map->map = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["map"]);
    std::string space = args.load_string("space", "tangent");
    if (space == "tangent") {
        normal_map->space = NormalMap::Space::TangentSpace;
    } else if (space == "world") {
        normal_map->space = NormalMap::Space::WorldSpace;
    }
    normal_map->range01 = args.load_bool("range01", true);
    normal_map->strength = args.load_float("strength", 1.0f);
    normal_map->to_world = args.load_transform("to_world", Transform());
    return normal_map;
}

} // namespace ks