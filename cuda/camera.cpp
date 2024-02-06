#include "camera.cuh"
#include "../assertion.h"

#include <algorithm>
#include <stb_image.h>
#include <stb_image_write.h>

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"
// tiny_gltf includes windows.h...
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif

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

void copy_accessor_to_linear(const std::vector<tinygltf::Buffer> &buffers,
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

void CameraAnimation::load_from_config(const ks::ConfigArgs &args)
{
    fs::path path = args.load_path("path");
    std::string camera_name = args.load_string("camera_name");

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());

    if (!err.empty()) {
        fprintf(stderr, "GLTF Loader Error: [%s]\n", err.c_str());
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

    const tinygltf::Camera &camera = cameras[model.nodes[camera_node].camera];
    if (camera.type == "perspective") {
        perspective = true;
        vfov = camera.perspective.yfov;
        aspect = camera.perspective.aspectRatio;
    } else if (camera.type == "orthographic") {
        perspective = false;
        left = -0.5f * camera.orthographic.xmag;
        right = 0.5f * camera.orthographic.xmag;
        bottom = -0.5f * camera.orthographic.ymag;
        top = 0.5f * camera.orthographic.xmag;
        near = camera.orthographic.znear;
        far = camera.orthographic.zfar;
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
        translation_input = &accessors[anim.samplers[translation_sampler].input];
        translation_output = &accessors[anim.samplers[translation_sampler].output];

        rotation_input = &accessors[anim.samplers[rotation_sampler].input];
        rotation_output = &accessors[anim.samplers[rotation_sampler].output];

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
        translation_keys.resize(translation_input->count);
        copy_accessor_to_linear(buffers, bufferviews, *translation_input,
                                reinterpret_cast<uint8_t *>(translation_keys.data()));

        translation_values.resize(translation_output->count);
        copy_accessor_to_linear(buffers, bufferviews, *translation_output,
                                reinterpret_cast<uint8_t *>(translation_values.data()));

        if (!std::is_sorted(translation_keys.begin(), translation_keys.end())) {
            fprintf(stderr, "Translation keys are not in ascending order.\n");
            std::abort();
        }
    }
    if (rotation_input) {
        ASSERT(rotation_input->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               rotation_input->type == TINYGLTF_TYPE_SCALAR);
        ASSERT(rotation_output->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
               rotation_output->type == TINYGLTF_TYPE_VEC4);

        rotation_keys.resize(rotation_input->count);
        copy_accessor_to_linear(buffers, bufferviews, *rotation_input,
                                reinterpret_cast<uint8_t *>(rotation_keys.data()));

        // NOTE: GLTF rotations are stored as XYZW quaternions
        rotation_values.resize(rotation_output->count);
        copy_accessor_to_linear(buffers, bufferviews, *rotation_output,
                                reinterpret_cast<uint8_t *>(translation_values.data()));

        if (!std::is_sorted(rotation_keys.begin(), rotation_keys.end())) {
            fprintf(stderr, "Rotation keys are not in ascending order.\n");
            std::abort();
        }
    }
}

} // namespace ksc
