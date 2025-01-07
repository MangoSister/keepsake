#pragma once

#include "../aabb.h"
#include "../light.h"
#include "../maths.h"
#include "ksvk.h"

namespace ks
{

// TODO:
// support mesh lights
// support more formats
// support editing
// support LTC??
struct GPULightSystem
{
  public:
    GPULightSystem(const AABB3 &scene_bound, LightPointers light_ptrs, const vk::Context &ctx);
    ~GPULightSystem();

    std::array<VkDescriptorSetLayout, 3> get_desc_set_layout() const
    {
        return {
            global_param_block_meta.desc_set_layout,
            texture_array_param_block_meta.desc_set_layout,
            pmf_array_param_block_meta.desc_set_layout,
        };
    }
    std::array<VkDescriptorSet, 3> get_desc_set() const
    {
        return {global_param_block.desc_set, texture_array_param_block.desc_set, pmf_array_param_block.desc_set};
    }

  private:
    LightPointers light_ptrs;

    vk::AutoRelease<vk::Buffer> uniforms_buf;
    vk::AutoRelease<vk::Buffer> headers_buf;
    vk::AutoRelease<vk::Buffer> pmf_buf;
    std::vector<vk::AutoRelease<vk::Buffer>> per_light_pmf_buf;
    std::vector<vk::AutoRelease<vk::ImageWithView>> textures_2d;
    VkSampler equal_area_sampler; // nearest clamp because we do bilerp manually

    vk::ParameterBlockMeta global_param_block_meta;
    vk::ParameterBlock global_param_block;

    vk::ParameterBlockMeta texture_array_param_block_meta;
    vk::ParameterBlock texture_array_param_block;

    vk::ParameterBlockMeta pmf_array_param_block_meta;
    vk::ParameterBlock pmf_array_param_block;

    VkDevice device;
};

} // namespace ks
