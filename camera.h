#pragma once
#include "config.h"
#include "maths.h"
#include "ray.h"
#include <memory>

namespace ks
{

struct Camera
{
    Camera() = default;
    Camera(const Transform &to_world, float vfov, float aspect);
    Camera(const vec3 &position, const vec3 &target, const vec3 &up, float vfov, float aspect);
    Camera(const Transform &to_world, float left, float right, float bottom, float top, float near_clip,
           float far_clip);
    Camera(const vec3 &position, const vec3 &target, const vec3 &up, float left, float right, float bottom, float top,
           float near_clip, float far_clip);

    Ray spawn_ray(const vec2 &film_pos, const vec2i &film_res, int spp) const;

    vec3 position() const { return vec3(camera_to_world.m(0, 3), camera_to_world.m(1, 3), camera_to_world.m(2, 3)); }

    vec3 direction() const
    {
        return -(vec3(camera_to_world.m(0, 2), camera_to_world.m(1, 2), camera_to_world.m(2, 2))).normalized();
    }

    Transform world_to_proj;
    Transform proj_to_world;
    Transform proj_to_camera;
    Transform camera_to_world;
    bool orthographic = false;
};

ks::mat4 look_at(const ks::vec3 &position, const ks::vec3 &target, ks::vec3 up);
ks::mat4 look_at_view(const ks::vec3 &position, const ks::vec3 &target, ks::vec3 up);
ks::mat4 rev_inf_projection(float vfov, float aspect, float near_clip = 0.1f);
ks::mat4 rev_orthographic(float left, float right, float bottom, float top, float near_clip, float far_clip);

struct ConfigArgs;
std::unique_ptr<Camera> create_camera(const ConfigArgs &args);

struct CameraAnimation : public Configurable
{
    uint32_t n_frames() const { return translation_keys.size(); }

    float duration() const { return translation_keys.back(); }

    Camera eval(float time) const;

    // TODO: also support animating FOV? GLTF doesn't support it...
    bool perspective;
    float vfov, aspect;
    float left, right, bottom, top, near_clip, far_clip;

    std::vector<float> translation_keys;
    std::vector<vec3> translation_values;
    std::vector<float> rotation_keys;
    std::vector<quat> rotation_values;
};

std::unique_ptr<CameraAnimation> create_camera_animation(const ConfigArgs &args);

} // namespace ks