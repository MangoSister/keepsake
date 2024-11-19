#pragma once

#include "../camera.h"
#include "ksvk.h"
#include "scene.h"

#include <slang.h>

namespace ks
{

struct GPUSmallPTInput
{
    const ks::Camera *camera = nullptr;
    vk::ImageWithView render_target;

    int bounces = 1;
    int render_width = 256;
    int render_height = 256;
    int crop_start_x = 0;
    int crop_start_y = 0;
    int crop_width = 0;
    int crop_height = 0;
    int spp = 1;
    int rng_seed = 0;
    bool scale_ray_diff = true;
    float clamp_indirect = 10.0f;

    int spp_prog_interval = 32;
    std::function<void(const vk::ImageWithView &rt, VkCommandBuffer cb, int spp_finished)> prog_interval_callback_gpu;
    std::function<void(int spp_finished)> prog_interval_callback_cpu;
};

struct GPUSmallPT
{
    // TODO decouple this? only descriptor set layout is needed.
    GPUSmallPT(const vk::Context &ctx, slang::ISession &slang_session, const GPUScene &gpu_scene);
    ~GPUSmallPT();

    void run(const GPUSmallPTInput &in);

    const GPUScene *scene = nullptr;

    vk::ParameterBlockMeta data_generator_param_block_meta;
    vk::ParameterBlock data_generator_param_block;

    VkPipelineLayout rt_pipeline_layout;
    VkPipeline rt_pipeline;
    vk::SBTWrapper sbt_wrapper;

    vk::AutoRelease<vk::FrequentUploadBuffer> global_params_buf;

    vk::CommandPool cp;

    vk::Allocator *allocator = nullptr;
    VkDevice device;
};

} // namespace ks