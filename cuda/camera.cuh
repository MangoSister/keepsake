#pragma once

#include "ray.cuh"
#ifdef CPP_CODE_ONLY
#include "../config.h"
#endif

namespace ksc
{

struct Camera
{
    Camera() = default;
    CUDA_HOST_DEVICE
    Camera(const ksc::Transform &to_world, float vfov, float aspect);
    CUDA_HOST_DEVICE
    Camera(const ksc::vec3 &position, const ksc::vec3 &target, const ksc::vec3 &up, float vfov, float aspect);
    CUDA_HOST_DEVICE
    Camera(const ksc::Transform &to_world, float left, float right, float bottom, float top, float near, float far);
    CUDA_HOST_DEVICE
    Camera(const ksc::vec3 &position, const ksc::vec3 &target, const ksc::vec3 &up, float left, float right,
           float bottom, float top, float near, float far);

    CUDA_HOST_DEVICE
    Ray spawn_ray(const ksc::vec2 &film_pos, const ksc::vec2i &film_res, int scale_spp) const;

    CUDA_HOST_DEVICE
    ksc::vec3 position() const
    {
        return ksc::vec3(camera_to_world.m[0][3], camera_to_world.m[1][3], camera_to_world.m[2][3]);
    }
    CUDA_HOST_DEVICE
    ksc::vec3 direction() const
    {
        return -normalize(ksc::vec3(camera_to_world.m[0][2], camera_to_world.m[1][2], camera_to_world.m[2][2]));
    }

    ksc::Transform proj_to_world;
    ksc::Transform proj_to_camera;
    ksc::Transform camera_to_world;
    bool orthographic = false;

#ifdef CPP_CODE_ONLY
    void load_from_config(const ks::ConfigArgs &args);
#endif
};

CUDA_HOST_DEVICE
ksc::mat4 look_at(const ksc::vec3 &position, const ksc::vec3 &target, ksc::vec3 up);
CUDA_HOST_DEVICE
ksc::mat4 look_at_view(const ksc::vec3 &position, const ksc::vec3 &target, ksc::vec3 up);
CUDA_HOST_DEVICE
ksc::mat4 rev_inf_projection(float vfov, float aspect, float near_clip = 0.01f);
CUDA_HOST_DEVICE
ksc::mat4 rev_orthographic(float left, float right, float bottom, float top, float near, float far);

} // namespace ksc