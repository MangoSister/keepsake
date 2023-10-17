#include "camera.cuh"

namespace ksc
{

void Camera::load_from_config(const ks::ConfigArgs &args)
{
    ks::vec3 camera_pos = args.load_vec3("position");
    ks::vec3 camera_target = args.load_vec3("target");
    ks::vec3 camera_up = args.load_vec3("up", true);
    float camera_vfov = ksc::to_radian(args.load_float("vfov"));
    float camera_aspect = args.load_float("aspect");
    *this = Camera(ksc::vec3(camera_pos.x(), camera_pos.y(), camera_pos.z()),
                   ksc::vec3(camera_target.x(), camera_target.y(), camera_target.z()),
                   ksc::vec3(camera_up.x(), camera_up.y(), camera_up.z()), camera_vfov, camera_aspect);
}

} // namespace ksc
