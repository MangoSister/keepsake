#pragma once
#include "ks/maths.h"
#include "ks/ray.h"

struct Camera
{
    Camera() = default;
    Camera(const vec3 &position, const vec3 &target, const vec3 &up, float vfov, float aspect);
    Camera(const vec3 &position, const vec3 &target, const vec3 &up, float left, float right, float bottom, float top,
           float near, float far);

    Ray spawn_ray(const vec2 &film_pos, const vec2i &film_res, int spp) const;

    vec3 camera_position;
    vec3 ortho_dir;

    Transform proj_to_world;
    Transform proj_to_camera;
    Transform camera_to_world;
};

struct ConfigArgs;
std::unique_ptr<Camera> create_camera(const ConfigArgs &args);
