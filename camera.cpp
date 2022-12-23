#include "camera.h"
#include "config.h"

KS_NAMESPACE_BEGIN

static inline mat4 look_at(const vec3 &position, const vec3 &target, vec3 up)
{
    vec3 back = (position - target).normalized();
    vec3 right = up.cross(back);

    if (right.squaredNorm() < 1e-8f) {
        orthonormal_basis(back, right, up);
    } else {
        right.normalize();
        up = back.cross(right);
    }
    mat4 mat;
    // clang-format off
    mat <<
        right.x(), up.x(), back.x(), position.x(),
        right.y(), up.y(), back.y(), position.y(),
        right.z(), up.z(), back.z(), position.z(),
        0.0f, 0.0f, 0.0f, 1.0f;
    // clang-format on
    return mat;
}

static inline mat4 look_at_view(const vec3 &position, const vec3 &target, vec3 up)
{
    vec3 back = (position - target).normalized();
    vec3 right = up.cross(back);

    if (right.squaredNorm() < 1e-8f) {
        orthonormal_basis(back, right, up);
    } else {
        right.normalize();
        up = back.cross(right);
    }
    mat4 mat;
    // clang-format off
    mat <<
        right.x(), right.y(), right.z(), -right.dot(position),
        up.x(), up.y(), up.z(), -up.dot(position),
        back.x(), back.y(), back.z(), -back.dot(position),
        0.0f, 0.0f, 0.0f, 1.0f;
    // clang-format on
    return mat;
}

static inline mat4 rev_inf_projection(float vfov, float aspect, float near_clip = 0.01f)
{
    // Reverse inf projection.
    float cot_half_vfov = 1.0f / std::tan(vfov * 0.5f);
    // Vulkan NDC space y axis pointing downward (same as screen space).
    mat4 proj;
    // clang-format off
    proj <<
        cot_half_vfov / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, -cot_half_vfov, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, near_clip,
        0.0f, 0.0f, -1.0f, 0.0f;
    // clang-format on
    return proj;
}

static inline mat4 rev_orthographic(float left, float right, float bottom, float top, float near, float far)
{
    mat4 proj;
    // clang-format off
    proj <<
        2.0f / (right - left), 0.0f, 0.0f, - (right + left) / (right - left),
        0.0f, 2.0f / (bottom - top), 0.0f,  - (top + bottom) / (bottom - top),
        0.0f, 0.0f, 1.0f / (far - near), far / (far - near),
        0.0f, 0.0f, 0.0f, 1.0f;
    // clang-format on
    return proj;
}

Camera::Camera(const Transform &to_world, float vfov, float aspect) : camera_to_world(to_world)
{
    camera_position = to_world.m.col(3).head(3);
    mat4 world_to_camera_m = camera_to_world.inverse().m;
    mat4 camera_to_proj_m = rev_inf_projection(vfov, aspect);
    proj_to_world = Transform(camera_to_proj_m * world_to_camera_m).inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    ortho_dir = vec3::Zero();
}

Camera::Camera(const vec3 &position, const vec3 &target, const vec3 &up, float vfov, float aspect)
    : camera_position(position)
{
    mat4 camera_to_world_m = look_at(camera_position, target, up);
    mat4 world_to_camera_m = camera_to_world_m.inverse();
    mat4 camera_to_proj_m = rev_inf_projection(vfov, aspect);
    proj_to_world = Transform(camera_to_proj_m * world_to_camera_m).inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
    ortho_dir = vec3::Zero();
}

Camera::Camera(const Transform &to_world, float left, float right, float bottom, float top, float near, float far)
    : camera_to_world(to_world)
{
    camera_position = to_world.m.col(3).head(3);
    mat4 world_to_camera_m = camera_to_world.inverse().m;
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near, far);
    proj_to_world = Transform(camera_to_proj_m * world_to_camera_m).inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    ortho_dir = -to_world.m.col(2).head(3).normalized();
}

Camera::Camera(const vec3 &position, const vec3 &target, const vec3 &up, float left, float right, float bottom,
               float top, float near, float far)
    : camera_position(position)
{
    mat4 camera_to_world_m = look_at(camera_position, target, up);
    mat4 world_to_camera_m = camera_to_world_m.inverse();
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near, far);
    proj_to_world = Transform(camera_to_proj_m * world_to_camera_m).inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
    ortho_dir = (target - position).normalized();
}

Ray Camera::spawn_ray(const vec2 &film_pos, const vec2i &film_res, int spp) const
{
    Ray ray;
    vec3 ndc_pos(film_pos.x() * 2.0f - 1.0f, film_pos.y() * 2.0f - 1.0f, 1.0f);

    vec3 mid = proj_to_camera.point_hdiv(vec3(0.0f, 0.0f, 1.0f));
    vec3 right = proj_to_camera.point_hdiv(vec3(2.0f / film_res.x(), 0.0f, 1.0f));
    vec3 up = proj_to_camera.point_hdiv(vec3(0.0f, 2.0f / film_res.y(), 1.0f));
    vec3 camera_dx = right - mid;
    camera_dx = camera_to_world.direction(camera_dx);
    vec3 camera_dy = up - mid;
    camera_dy = camera_to_world.direction(camera_dy);

    if (ortho_dir == vec3::Zero()) {
        vec3 world_pos = proj_to_world.point_hdiv(ndc_pos);
        vec3 ray_dir = (world_pos - camera_position).normalized();
        ray = Ray(camera_position, ray_dir, 0.0f, inf);

        ray.rx_origin = ray.ry_origin = ray.origin;
        ray.rx_dir = (world_pos + camera_dx - camera_position).normalized();
        ray.ry_dir = (world_pos + camera_dy - camera_position).normalized();
    } else {
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

std::unique_ptr<Camera> create_camera(const ConfigArgs &args)
{
    Transform to_world;
    if (args.contains("to_world")) {
        to_world = args.load_transform("to_world");
    } else {
        vec3 camera_pos = args.load_vec3("pos");
        vec3 camera_target = args.load_vec3("target");
        vec3 camera_up = args.load_vec3("up", true);
        to_world = Transform(look_at(camera_pos, camera_target, camera_up));
    }

    std::string type = args.load_string("type");
    if (type == "perspective") {
        float vfov = to_radian(args.load_float("vfov"));
        float aspect = args.load_float("aspect");
        return std::make_unique<Camera>(to_world, vfov, aspect);
    } else if (type == "orthographic") {
        float left = args.load_float("left");
        float right = args.load_float("right");
        float bottom = args.load_float("bottom");
        float top = args.load_float("top");
        float near = args.load_float("near");
        float far = args.load_float("far");
        return std::make_unique<Camera>(to_world, left, right, bottom, top, near, far);
    }
    return nullptr;
}

KS_NAMESPACE_END