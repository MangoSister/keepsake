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

enum LightFlagBits : uint32_t
{
    Directional = (1 << 0),
    Point = (1 << 1),
    Sky = (1 << 2),
};
using LightFlag = uint32_t;

struct LightHeader
{
    vec3 field1;
    LightFlag flag;
    vec3 field2;
    int ext;
};

struct LightSystemUniforms
{
    uint32_t num_skylights;
};

struct GPULightSystem
{
  public:
    // TODO: for now we assume that all skylights are at the beginning.
    GPULightSystem(LightPointers light_ptrs, const vk::Context &ctx, uint32_t max_frames_in_flight);
    ~GPULightSystem();

    std::array<VkDescriptorSetLayout, 3> get_desc_set_layout() const
    {
        return {
            global_param_block_meta.desc_set_layout,
            texture_array_param_block_meta.desc_set_layout,
            pmf_array_param_block_meta.desc_set_layout,
        };
    }
    std::array<VkDescriptorSet, 3> get_desc_set(uint32_t curr_frame_in_flight_index) const
    {
        return {global_param_block[curr_frame_in_flight_index].desc_set, texture_array_param_block.desc_set,
                pmf_array_param_block.desc_set};
    }

    void update(const AABB3 &scene_bound);
    void upload(uint32_t curr_frame_in_flight_index, VkCommandBuffer cb);

  private:
    LightPointers light_ptrs;

    std::vector<float> skylights_unit_powers;
    std::vector<uint32_t> skylight_texture_indices;
    LightSystemUniforms uniforms;
    std::vector<LightHeader> headers;
    AliasTable pmf;

    std::vector<vk::AutoRelease<vk::FrequentUploadBuffer>> uniforms_buf;
    std::vector<vk::AutoRelease<vk::FrequentUploadBuffer>> headers_buf;
    std::vector<vk::AutoRelease<vk::FrequentUploadBuffer>> pmf_buf;

    // Don't support texture update yet...
    std::vector<vk::AutoRelease<vk::Buffer>> per_light_pmf_buf;

    std::vector<vk::AutoRelease<vk::ImageWithView>> textures_2d;
    VkSampler equal_area_sampler; // nearest clamp because we do bilerp manually

    vk::ParameterBlockMeta global_param_block_meta;
    std::vector<vk::ParameterBlock> global_param_block;

    vk::ParameterBlockMeta texture_array_param_block_meta;
    vk::ParameterBlock texture_array_param_block;

    vk::ParameterBlockMeta pmf_array_param_block_meta;
    vk::ParameterBlock pmf_array_param_block;

    VkDevice device = VK_NULL_HANDLE;
    std::shared_ptr<vk::Allocator> allocator;
};

} // namespace ks
