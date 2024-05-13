#include "camera.h"
#include "assertion.h"
#include "config.h"

#include "tiny_gltf.h"
#include <stb_image.h>
#include <stb_image_write.h>

namespace ks
{

mat4 look_at(const vec3 &position, const vec3 &target, vec3 up)
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

mat4 look_at_view(const vec3 &position, const vec3 &target, vec3 up)
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

mat4 rev_inf_projection(float vfov, float aspect, float near_clip)
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

mat4 rev_orthographic(float left, float right, float bottom, float top, float near_clip, float far_clip)
{
    mat4 proj;
    // clang-format off
    proj <<
        2.0f / (right - left), 0.0f, 0.0f, - (right + left) / (right - left),
        0.0f, 2.0f / (bottom - top), 0.0f,  - (top + bottom) / (bottom - top),
        0.0f, 0.0f, 1.0f / (far_clip - near_clip), far_clip / (far_clip - near_clip),
        0.0f, 0.0f, 0.0f, 1.0f;
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
    mat4 world_to_camera_m = camera_to_world_m.inverse();
    mat4 camera_to_proj_m = rev_inf_projection(vfov, aspect);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
}

Camera::Camera(const Transform &to_world, float left, float right, float bottom, float top, float near_clip,
               float far_clip)
    : camera_to_world(to_world), orthographic(true)
{
    mat4 world_to_camera_m = camera_to_world.inv;
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near_clip, far_clip);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
}

Camera::Camera(const vec3 &position, const vec3 &target, const vec3 &up, float left, float right, float bottom,
               float top, float near_clip, float far_clip)
    : orthographic(true)
{
    mat4 camera_to_world_m = look_at(position, target, up);
    mat4 world_to_camera_m = camera_to_world_m.inverse();
    mat4 camera_to_proj_m = rev_orthographic(left, right, bottom, top, near_clip, far_clip);
    world_to_proj = Transform(camera_to_proj_m * world_to_camera_m);
    proj_to_world = world_to_proj.inverse();
    proj_to_camera = Transform(camera_to_proj_m).inverse();
    camera_to_world = Transform(camera_to_world_m);
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

    if (!orthographic) {
        vec3 camera_position = position();
        vec3 world_pos = proj_to_world.point_hdiv(ndc_pos);
        vec3 ray_dir = (world_pos - camera_position).normalized();
        ray = Ray(camera_position, ray_dir, 0.0f, inf);

        ray.rx_origin = ray.ry_origin = ray.origin;
        ray.rx_dir = (world_pos + camera_dx - camera_position).normalized();
        ray.ry_dir = (world_pos + camera_dy - camera_position).normalized();
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
        float aspect = args.load_float("aspect");
        float vfov;
        if (args.contains("vfov")) {
            vfov = to_radian(args.load_float("vfov"));
        } else {
            float hfov = to_radian(args.load_float("hfov"));
            vfov = 2.0f * std::atan((1.0f / aspect) * std::tan(0.5f * hfov));
        }
        return std::make_unique<Camera>(to_world, vfov, aspect);
    } else if (type == "orthographic") {
        float left = args.load_float("left");
        float right = args.load_float("right");
        float bottom = args.load_float("bottom");
        float top = args.load_float("top");
        float near_clip = args.load_float("near_clip");
        float far_clip = args.load_float("far_clip");
        return std::make_unique<Camera>(to_world, left, right, bottom, top, near_clip, far_clip);
    }
    return nullptr;
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
        left = std::max(right - 1, 0);
    } else {
        right = (int)translation_keys.size() - 1;
        left = std::max(right - 1, 0);
    }

    float time_left = translation_keys[left];
    float time_right = translation_keys[right];
    float delta = saturate((time - time_left) / (time_right - time_left));

    vec3 translation = lerp(translation_values[left], translation_values[right], delta);

    if (rotation_keys.size() == 1) {
        left = right = 0;
    }
    it = std::upper_bound(rotation_keys.begin(), rotation_keys.end(), time);
    if (it != rotation_keys.end()) {
        right = std::distance(rotation_keys.begin(), it);
        left = std::max(right - 1, 0);
    } else {
        right = (int)rotation_keys.size() - 1;
        left = std::max(right - 1, 0);
    }

    time_left = rotation_keys[left];
    time_right = rotation_keys[right];
    delta = saturate((time - time_left) / (time_right - time_left));

    quat rotation = rotation_values[left].slerp(delta, rotation_values[right]);
    mat4 m = mat4::Identity();
    m.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    m.col(3).head(3) = translation;
    Transform to_world(m, affine_inverse(m));

    if (perspective) {
        return Camera(to_world, vfov, aspect);
    } else {
        return Camera(to_world, left, right, bottom, top, near_clip, far_clip);
    }
}

static void copy_accessor_to_linear(const std::vector<tinygltf::Buffer> &buffers,
                                    const std::vector<tinygltf::BufferView> &bufferviews, const tinygltf::Accessor &acc,
                                    uint8_t *dest)
{
    const auto &view = bufferviews[acc.bufferView];
    const auto &buf = buffers[view.buffer];

    ASSERT(!buf.data.empty());
    const uint8_t *buf_data = buf.data.data();
    const uint8_t *src = buf_data + view.byteOffset + acc.byteOffset;

    int comp_size_in_bytes = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(acc.componentType));
    int num_comp = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(acc.type));
    int element_size_in_bytes = comp_size_in_bytes * num_comp;

    int stride = acc.ByteStride(view);
    for (int i = 0; i < acc.count; ++i) {
        std::copy(src, src + element_size_in_bytes, dest);
        dest += element_size_in_bytes;
        src += stride;
    }
}

// TODO: merge this with GLTF compound mesh asset?
std::unique_ptr<CameraAnimation> create_camera_animation(const ks::ConfigArgs &args)
{
    fs::path path = args.load_path("path");
    std::string camera_name = args.load_string("camera_name");
    bool convert_y_up_to_z_up = args.load_bool("convert_y_up_to_z_up");
    if (!convert_y_up_to_z_up) {
        fprintf(stderr, "Is this data from Blender? Currently Blender GLTF exporter is buggy with non y-up! May want "
                        "to export as y-up and convert back to z-up.\n");
    }
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    if (!ret || !err.empty()) {
        fprintf(stderr, "GLTF Loader failed to load [%s] with error: [%s]\n", path.string().c_str(), err.c_str());
        std::abort();
    }
    if (!warn.empty()) {
        fprintf(stderr, "GLTF Loader Warning: [%s]\n", warn.c_str());
    }

    const std::vector<tinygltf::Camera> &cameras = model.cameras;
    const std::vector<tinygltf::Animation> &animations = model.animations;
    const std::vector<tinygltf::Buffer> &buffers = model.buffers;
    const std::vector<tinygltf::BufferView> &bufferviews = model.bufferViews;
    const std::vector<tinygltf::Accessor> &accessors = model.accessors;

    auto dfs_find_camera = [&](int node_idx, auto &self) -> int {
        const auto &node = model.nodes[node_idx];
        if (node.camera >= 0 && cameras[node.camera].name == camera_name) {
            return node_idx;
        }

        for (int child_idx : node.children) {
            int ret = self(child_idx, self);
            if (ret >= 0)
                return ret;
        }
        return -1;
    };

    int camera_node = -1;
    for (int root : model.scenes[model.defaultScene].nodes) {
        camera_node = dfs_find_camera(root, dfs_find_camera);
        if (camera_node >= 0) {
            break;
        }
    }
    if (camera_node == -1) {
        fprintf(stderr, "No camera node named [%s] in the gltf file [%s]\n", camera_name.c_str(),
                path.string().c_str());
        std::abort();
    }

    std::unique_ptr<CameraAnimation> camera_anim = std::make_unique<CameraAnimation>();

    const tinygltf::Camera &camera = cameras[model.nodes[camera_node].camera];
    if (camera.type == "perspective") {
        camera_anim->perspective = true;
        camera_anim->vfov = camera.perspective.yfov;
        camera_anim->aspect = camera.perspective.aspectRatio;
    } else if (camera.type == "orthographic") {
        camera_anim->perspective = false;
        camera_anim->left = -0.5f * camera.orthographic.xmag;
        camera_anim->right = 0.5f * camera.orthographic.xmag;
        camera_anim->bottom = -0.5f * camera.orthographic.ymag;
        camera_anim->top = 0.5f * camera.orthographic.xmag;
        camera_anim->near_clip = camera.orthographic.znear;
        camera_anim->far_clip = camera.orthographic.zfar;
    } else {
        fprintf(stderr, "Invalida camera type [%s] in the gltf file [%s]\n", camera.type.c_str(),
                path.string().c_str());
        std::abort();
    }

    const tinygltf::Accessor *translation_input = nullptr;
    const tinygltf::Accessor *translation_output = nullptr;

    const tinygltf::Accessor *rotation_input = nullptr;
    const tinygltf::Accessor *rotation_output = nullptr;

    for (const tinygltf::Animation &anim : animations) {
        int translation_sampler = -1;
        int rotation_sampler = -1;
        for (int ch = 0; ch < (int)anim.channels.size(); ++ch) {
            if (anim.channels[ch].target_node == camera_node) {
                if (anim.channels[ch].target_path == "translation") {
                    translation_sampler = anim.channels[ch].sampler;
                } else if (anim.channels[ch].target_path == "rotation") {
                    rotation_sampler = anim.channels[ch].sampler;
                }
            }
        }

        // Just do linear/slerp for now...
        if (translation_sampler >= 0) {
            translation_input = &accessors[anim.samplers[translation_sampler].input];
            translation_output = &accessors[anim.samplers[translation_sampler].output];
        }
        if (rotation_sampler >= 0) {
            rotation_input = &accessors[anim.samplers[rotation_sampler].input];
            rotation_output = &accessors[anim.samplers[rotation_sampler].output];
        }
        if (translation_sampler >= 0 && rotation_sampler >= 0) {
            break;
        }
    }
    if (!translation_input && !rotation_input) {
        fprintf(stderr,
                "Did not find any animation for the camera [%s] in the gltf file [%s]. Wrong camera or animation "
                "applied to its parent?\n",
                camera_name.c_str(), path.string().c_str());
        std::abort();
    }

    if (translation_input) {
        ASSERT(translation_input->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               translation_input->type == TINYGLTF_TYPE_SCALAR);
        ASSERT(translation_output->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               translation_output->type == TINYGLTF_TYPE_VEC3);
        camera_anim->translation_keys.resize(translation_input->count);
        copy_accessor_to_linear(buffers, bufferviews, *translation_input,
                                reinterpret_cast<uint8_t *>(camera_anim->translation_keys.data()));

        camera_anim->translation_values.resize(translation_output->count);
        copy_accessor_to_linear(buffers, bufferviews, *translation_output,
                                reinterpret_cast<uint8_t *>(camera_anim->translation_values.data()));

        if (!std::is_sorted(camera_anim->translation_keys.begin(), camera_anim->translation_keys.end())) {
            fprintf(stderr, "Translation keys are not in ascending order.\n");
            std::abort();
        }
    }
    if (rotation_input) {
        ASSERT(rotation_input->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               rotation_input->type == TINYGLTF_TYPE_SCALAR);
        ASSERT(rotation_output->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               rotation_output->type == TINYGLTF_TYPE_VEC4);

        camera_anim->rotation_keys.resize(rotation_input->count);
        copy_accessor_to_linear(buffers, bufferviews, *rotation_input,
                                reinterpret_cast<uint8_t *>(camera_anim->rotation_keys.data()));

        // NOTE: GLTF rotations are stored as XYZW quaternions
        std::vector<float> xyzw_values(rotation_output->count * 4);
        copy_accessor_to_linear(buffers, bufferviews, *rotation_output,
                                reinterpret_cast<uint8_t *>(xyzw_values.data()));
        camera_anim->rotation_values.resize(rotation_output->count);
        for (uint32_t i = 0; i < (uint32_t)rotation_output->count; ++i) {
            camera_anim->rotation_values[i].w() = xyzw_values[4 * i + 3];
            camera_anim->rotation_values[i].x() = xyzw_values[4 * i + 0];
            camera_anim->rotation_values[i].y() = xyzw_values[4 * i + 1];
            camera_anim->rotation_values[i].z() = xyzw_values[4 * i + 2];
        }

        if (!std::is_sorted(camera_anim->rotation_keys.begin(), camera_anim->rotation_keys.end())) {
            fprintf(stderr, "Rotation keys are not in ascending order.\n");
            std::abort();
        }
    }

    if (convert_y_up_to_z_up) {
        for (auto &v : camera_anim->translation_values) {
            float x = v.x();
            float y = v.y();
            float z = v.z();

            v.y() = -z;
            v.z() = y;
        }

        quat r(convert_yup_to_zup().block<3, 3>(0, 0));
        for (auto &q : camera_anim->rotation_values) {
            q = r * q;
        }
    }

    return camera_anim;
}

} // namespace ks