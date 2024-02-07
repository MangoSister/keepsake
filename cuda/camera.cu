#include "camera.cuh"
#include <algorithm>

namespace ksc
{
mat4 look_at(const vec3 &position, const vec3 &target, vec3 up)
{
    vec3 back = normalize(position - target);
    vec3 right = cross(up, back);

    if (length_squared(right) < 1e-8f) {
        orthonormal_basis(back, right, up);
    } else {
        right = normalize(right);
        up = cross(back, right);
    }
    mat4 mat;
    mat[0][0] = right.x, mat[0][1] = up.x, mat[0][2] = back.x, mat[0][3] = position.x;
    mat[1][0] = right.y, mat[1][1] = up.y, mat[1][2] = back.y, mat[1][3] = position.y;
    mat[2][0] = right.z, mat[2][1] = up.z, mat[2][2] = back.z, mat[2][3] = position.z;
    mat[3][0] = 0.0f, mat[3][1] = 0.0f, mat[3][2] = 0.0f, mat[3][3] = 1.0f;

    return mat;
}

mat4 look_at_view(const vec3 &position, const vec3 &target, vec3 up)
{
    vec3 back = normalize(position - target);
    vec3 right = cross(up, back);

    if (length_squared(right) < 1e-8f) {
        orthonormal_basis(back, right, up);
    } else {
        right = normalize(right);
        up = cross(back, right);
    }
    mat4 mat;
    mat[0][0] = right.x, mat[0][1] = right.y, mat[0][2] = right.z, mat[0][3] = dot(-right, position);
    mat[1][0] = up.x, mat[1][1] = up.y, mat[1][2] = up.z, mat[1][3] = dot(-up, position);
    mat[2][0] = back.x, mat[2][1] = back.y, mat[2][2] = back.z, mat[2][3] = dot(-back, position);
    mat[3][0] = 0.0f, mat[3][1] = 0.0f, mat[3][2] = 0.0f, mat[3][3] = 1.0f;

    return mat;
}

mat4 rev_inf_projection(float vfov, float aspect, float near_clip)
{
    // Reverse inf projection.
    float cot_half_vfov = 1.0f / std::tan(vfov * 0.5f);
    // Vulkan NDC space y axis pointing downward (same as screen space).
    mat4 proj;
    proj[0][0] = cot_half_vfov / aspect, proj[0][1] = 0.0f, proj[0][2] = 0.0f, proj[0][3] = 0.0f;
    proj[1][0] = 0.0f, proj[1][1] = -cot_half_vfov, proj[1][2] = 0.0f, proj[1][3] = 0.0f;
    proj[2][0] = 0.0f, proj[2][1] = 0.0f, proj[2][2] = 0.0f, proj[2][3] = near_clip;
    proj[3][0] = 0.0f, proj[3][1] = 0.0f, proj[3][2] = -1.0f, proj[3][3] = 0.0f;

    return proj;
}

mat4 rev_orthographic(float left, float right, float bottom, float top, float near, float far)
{
    mat4 proj;
    // clang-format off
    proj[0][0] = 2.0f / (right - left),proj[0][1] =  0.0f, proj[0][2] = 0.0f, proj[0][3] = - (right + left) / (right - left);
    proj[1][0] = 0.0f, proj[1][1] = 2.0f / (bottom - top),proj[1][2] =  0.0f,  proj[1][3] = - (top + bottom) / (bottom - top);
    proj[2][0] = 0.0f, proj[2][1] = 0.0f,proj[2][2] =  1.0f / (far - near), proj[2][3] = far / (far - near);
    proj[3][0] = 0.0f, proj[3][1] = 0.0f, proj[3][2] = 0.0f, proj[3][3] = 1.0f;
    // clang-format on
    return proj;
}

Camera::Camera(const Transform &to_world, float vfov, float aspect) : camera_to_world(to_world), orthographic(false)
{
    mat4 world_to_camera_m = camera_to_world.inv;
    mat4 camera_to_proj_m = rev_inf_projection(vfov, aspect);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
}

Camera::Camera(const vec3 &position, const vec3 &target, const vec3 &up, float vfov, float aspect) : orthographic(false)
{
    mat4 camera_to_world_m = look_at(position, target, up);
    mat4 world_to_camera_m = inverse(camera_to_world_m);
    mat4 camera_to_proj_m = rev_inf_projection(vfov, aspect);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
}

Camera::Camera(const Transform &to_world, float left, float right, float bottom, float top, float near, float far)
    : camera_to_world(to_world), orthographic(true)
{
    mat4 world_to_camera_m = camera_to_world.inv;
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near, far);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
}

Camera::Camera(const vec3 &position, const vec3 &target, const vec3 &up, float left, float right, float bottom,
               float top, float near, float far)
    : orthographic(true)
{
    mat4 camera_to_world_m = look_at(position, target, up);
    mat4 world_to_camera_m = inverse(camera_to_world_m);
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near, far);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
}

Ray Camera::spawn_ray(const vec2 &film_pos, const vec2i &film_res, int spp) const
{
    Ray ray;
    vec3 ndc_pos(film_pos.x * 2.0f - 1.0f, film_pos.y * 2.0f - 1.0f, 1.0f);

    vec3 mid = proj_to_camera.point_hdiv(vec3(0.0f, 0.0f, 1.0f));
    vec3 right = proj_to_camera.point_hdiv(vec3(2.0f / film_res.x, 0.0f, 1.0f));
    vec3 up = proj_to_camera.point_hdiv(vec3(0.0f, 2.0f / film_res.y, 1.0f));
    vec3 camera_dx = right - mid;
    camera_dx = camera_to_world.direction(camera_dx);
    vec3 camera_dy = up - mid;
    camera_dy = camera_to_world.direction(camera_dy);

    if (!orthographic) {
        vec3 camera_position = position();
        vec3 world_pos = proj_to_world.point_hdiv(ndc_pos);
        vec3 ray_dir = normalize(world_pos - camera_position);
        ray = Ray(camera_position, ray_dir, 0.0f, inf);

        ray.rx_origin = ray.ry_origin = ray.origin;
        ray.rx_dir = normalize(world_pos + camera_dx - camera_position);
        ray.ry_dir = normalize(world_pos + camera_dy - camera_position);
    } else {
        vec3 ortho_dir = direction();
        vec3 origin = proj_to_world.point_hdiv(ndc_pos);
        // TODO: ray diff in this case?
        ray = Ray(origin, ortho_dir, 0.0f, inf);

        ray.rx_origin = ray.origin + camera_dx;
        ray.ry_origin = ray.origin + camera_dy;
        ray.rx_dir = ray.ry_dir = ray.dir;
    }

    if (ray.has_ray_diffs()) {
        float scale = 1.0f / std::sqrt((float)spp);
        ray.rx_origin = ray.origin + (ray.rx_origin - ray.origin) * scale;
        ray.ry_origin = ray.origin + (ray.ry_origin - ray.origin) * scale;
        ray.rx_dir = ray.dir + (ray.rx_dir - ray.dir) * scale;
        ray.ry_dir = ray.dir + (ray.ry_dir - ray.dir) * scale;
    }
    return ray;
}

Camera CameraAnimation::eval(float time) const
{
    int left, right;
    if (translation_keys.size() == 1) {
        left = right = 0;
    }
    auto it = std::upper_bound(translation_keys.begin(), translation_keys.end(), time);
    if (it != translation_keys.end()) {
        right = std::distance(translation_keys.begin(), it);
        left = right - 1;
    } else {
        right = (int)translation_keys.size() - 1;
        left = right - 1;
    }

    float time_left = translation_keys[left];
    float time_right = translation_keys[right];
    float delta = (time - time_left) / (time_right - time_left);

    vec3 translation = lerp(delta, translation_values[left], translation_values[right]);

    if (rotation_keys.size() == 1) {
        left = right = 0;
    }
    it = std::upper_bound(rotation_keys.begin(), rotation_keys.end(), time);
    if (it != rotation_keys.end()) {
        right = std::distance(rotation_keys.begin(), it);
        left = right - 1;
    } else {
        right = (int)rotation_keys.size() - 1;
        left = right - 1;
    }

    time_left = rotation_keys[left];
    time_right = rotation_keys[right];
    delta = (time - time_left) / (time_right - time_left);

    quat rotation = slerp(delta, rotation_values[left], rotation_values[right]);
    mat4 m = make_affine(rotation.to_matrix(), translation);
    Transform to_world(m, affine_inverse(m));
    if (perspective) {
        return Camera(to_world, vfov, aspect);
    } else {
        return Camera(to_world, left, right, bottom, top, near, far);
    }
}

} // namespace ksc