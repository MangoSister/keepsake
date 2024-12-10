#pragma once
#include "ksvk.h"

namespace ks
{

struct AdamOptimizerHyperParams
{
    float lr = 1e-3f;
    float beta1 = 0.9f, beta2 = 0.999f;
    float eps = 1e-8f;
    int iter = 1; // NOTE: iter starts from 1.
};

struct AdamOptimizerShared
{
    AdamOptimizerShared(GPUContext &gpu, uint32_t max_insts);

    ~AdamOptimizerShared();

    vk::ParameterBlockMeta param_block_meta;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct AdamOptimizer
{
    AdamOptimizer(AdamOptimizerHyperParams hparams, uint32_t num_parameters, vk::Buffer parameters_buf,
                  vk::Buffer gradients_buf, AdamOptimizerShared &shared, vk::Context &ctx,
                  VkCommandBuffer ext_cb = VK_NULL_HANDLE);

    void step(VkCommandBuffer cb);

    // TODO: reset, etc
    // TODO: weight decay?

    AdamOptimizerHyperParams hparams;
    uint32_t num_parameters;
    vk::AutoRelease<vk::Buffer> first_moments;
    vk::AutoRelease<vk::Buffer> second_moments;
    vk::ParameterBlock param_block;
    AdamOptimizerShared *shared;
};

} // namespace ks