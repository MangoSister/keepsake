#include "light.h"

namespace ks
{

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

inline LightHeader make_skylight_header(const EqualAreaSkyLight &l, uint32_t tex_idx)
{
    LightHeader h;
    h.flag = LightFlagBits::Sky;
    h.field1 = vec3(l.to_world.w(), l.to_world.x(), l.to_world.y());
    h.field2 = vec3(l.to_world.z(), l.strength, 0.0f);
    h.ext = tex_idx;
    return h;
}

inline LightHeader make_dirlight_header(const DirectionalLight &l)
{
    LightHeader h;
    h.flag = LightFlagBits::Directional;
    h.field1 = l.L;
    h.field2 = l.dir;
    return h;
}

inline LightHeader make_pointlight_header(const PointLight &l)
{
    LightHeader h;
    h.flag = LightFlagBits::Directional;
    h.field1 = l.I;
    h.field2 = l.pos;
    return h;
}

struct LightSystemUniforms
{
    uint32_t num_skylights;
};

enum class LightSystemGlobalBindings : uint32_t
{
    Uniforms = 0,
    Headers = 1,
    PMF = 2,
};

GPULightSystem::GPULightSystem(const AABB3 &scene_bound, LightPointers light_ptrs_, const vk::Context &ctx)
    : light_ptrs(std::move(light_ptrs_)), device(ctx.device)
{
    std::vector<const EqualAreaSkyLight *> skylights;
    std::vector<const Light *> other_lights;
    for (uint32_t i = 0; i < (uint32_t)light_ptrs.lights.size(); ++i) {
        const EqualAreaSkyLight *sky = dynamic_cast<const EqualAreaSkyLight *>(light_ptrs.lights[i]);
        if (sky) {
            skylights.push_back(sky);
        } else {
            other_lights.push_back(light_ptrs.lights[i]);
        }
    }

    std::vector<LightHeader> headers(light_ptrs.lights.size());
    std::vector<float> powers(light_ptrs.lights.size());
    uint32_t tex_count = 0;
    for (uint32_t i = 0; i < (uint32_t)light_ptrs.lights.size(); ++i) {
        if (i < skylights.size()) {
            headers[i] = make_skylight_header(*skylights[i], tex_count);
            powers[i] = skylights[i]->power(scene_bound).mean();

            ctx.allocator->stage_session([&](vk::Allocator &self) {
                per_light_pmf_buf.emplace_back(ctx.allocator,
                                               VkBufferCreateInfo{
                                                   .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               },
                                               VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0),
                                               skylights[i]->pmf.bins);
            });

            VkImageCreateInfo image_ci{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VkImageType::VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                .extent = VkExtent3D(skylights[i]->res, skylights[i]->res, 1),
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            };

            vk::Image image;
            ctx.allocator->stage_session([&](vk::Allocator &self) {
                image = ctx.allocator->create_and_upload_image(
                    image_ci, VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0),
                    [&](std::byte *dest) {
                        short *fp16_ptr = reinterpret_cast<short *>(dest);
                        uint32_t n_texels = sqr(skylights[i]->res);
                        for (uint32_t t = 0; t < n_texels; ++t) {
                            for (uint32_t c = 0; c < 3; ++c) {
                                *(fp16_ptr++) = convert_float_to_half(skylights[i]->texels[t][c]);
                            }
                            *(fp16_ptr++) = convert_float_to_half(1.0f);
                        }
                    },
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, vk::MipmapOption::AutoGenerate, false);
            });
            textures_2d.emplace_back(ctx.allocator, image,
                                     VkImageViewCreateInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                                           .image = image.image,
                                                           .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                                           .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                                                           .subresourceRange = VkImageSubresourceRange{
                                                               .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                               .baseMipLevel = 0,
                                                               .levelCount = VK_REMAINING_MIP_LEVELS,
                                                               .baseArrayLayer = 0,
                                                               .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                                           }});

            ++tex_count;
        } else {
            int idx = i - (uint32_t)skylights.size();
            if (const DirectionalLight *l = dynamic_cast<const DirectionalLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_dirlight_header(*l);
            } else if (const PointLight *l = dynamic_cast<const PointLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_pointlight_header(*l);
            }
            powers[i] = other_lights[idx]->power(scene_bound).mean();
        }
    }
    AliasTable pmf(powers);

    LightSystemUniforms uniforms;
    uniforms.num_skylights = (uint32_t)skylights.size();

    ctx.allocator->stage_session([&](vk::Allocator &self) {
        uniforms_buf = vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                                   VkBufferCreateInfo{
                                                       .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                       .size = sizeof(LightSystemUniforms),
                                                       .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                   },
                                                   VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0),
                                                   reinterpret_cast<const std::byte *>(&uniforms));

        headers_buf = vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                                  VkBufferCreateInfo{
                                                      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  },
                                                  VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), headers);

        pmf_buf = vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                              VkBufferCreateInfo{
                                                  .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              },
                                              VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), pmf.bins);
    });

    {
        VkSamplerCreateInfo sampler_ci{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };
        vk::vk_check(vkCreateSampler(device, &sampler_ci, nullptr, &equal_area_sampler));
    }

    // vk::ParameterBlockMeta param_block_meta;
    // vk::ParameterBlock param_block;
    constexpr VkShaderStageFlags ray_trace_shader_stages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    constexpr VkShaderStageFlags lighting_stages =
        ray_trace_shader_stages | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    {
        vk::DescriptorSetHelper helper;
        helper.add_binding("uniforms", {.binding = (uint32_t)LightSystemGlobalBindings::Uniforms,
                                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                        .descriptorCount = 1,
                                        .stageFlags = lighting_stages});
        helper.add_binding("headers", {.binding = (uint32_t)LightSystemGlobalBindings::Headers,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = lighting_stages});
        helper.add_binding("pmf", {.binding = (uint32_t)LightSystemGlobalBindings::PMF,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   .descriptorCount = 1,
                                   .stageFlags = lighting_stages});
        global_param_block_meta = vk::ParameterBlockMeta(ctx.device, 1, std::move(helper));
        global_param_block = global_param_block_meta.allocate_block();

        vk::ParameterWriteArray write_array;
        global_param_block.write_buffer(
            "uniforms", VkDescriptorBufferInfo{.buffer = uniforms_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            write_array);
        global_param_block.write_buffer(
            "headers", VkDescriptorBufferInfo{.buffer = headers_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            write_array);
        global_param_block.write_buffer(
            "pmf", VkDescriptorBufferInfo{.buffer = pmf_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE}, write_array);
        vkUpdateDescriptorSets(device, write_array.writes.size(), write_array.writes.data(), 0, nullptr);
    }

    {
        vk::DescriptorSetHelper helper;
        // We only need to specify a maximum array length, but here the max is same as the actual allocated number
        // anyway. Also cap to 1 to avoid 0 descriptorCount validation error.
        uint32_t num_textures_2d = std::max((uint32_t)textures_2d.size(), 1u);
        helper.add_binding("t",
                           {.binding = 0,
                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = num_textures_2d,
                            .stageFlags = lighting_stages},
                           true);
        texture_array_param_block_meta = vk::ParameterBlockMeta(ctx.device, 1, std::move(helper));
        texture_array_param_block = texture_array_param_block_meta.allocate_block(num_textures_2d);

        vk::ParameterWriteArray write_array;
        if (!textures_2d.empty()) {
            std::vector<VkDescriptorImageInfo> image_infos(textures_2d.size());
            for (uint32_t i = 0; i < headers.size(); ++i) {
                if (headers[i].flag & LightFlagBits::Sky) {
                    uint32_t tex_id = headers[i].ext;
                    image_infos[tex_id].imageView = textures_2d[tex_id]->view;
                    image_infos[tex_id].sampler = equal_area_sampler;
                    image_infos[tex_id].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
                //
            }
            texture_array_param_block.write_images("t", std::move(image_infos), 0, write_array);
        }
        vkUpdateDescriptorSets(device, write_array.writes.size(), write_array.writes.data(), 0, nullptr);
    }
    {
        vk::DescriptorSetHelper helper;
        // We only need to specify a maximum array length, but here the max is same as the actual allocated number
        // anyway. Also cap to 1 to avoid 0 descriptorCount validation error.
        uint32_t num_per_light_pmfs = std::max((uint32_t)per_light_pmf_buf.size(), 1u);
        helper.add_binding("a",
                           {.binding = 0,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = num_per_light_pmfs,
                            .stageFlags = lighting_stages},
                           true);
        pmf_array_param_block_meta = vk::ParameterBlockMeta(ctx.device, 1, std::move(helper));
        pmf_array_param_block = pmf_array_param_block_meta.allocate_block(num_per_light_pmfs);

        vk::ParameterWriteArray write_array;
        if (!per_light_pmf_buf.empty()) {
            std::vector<VkDescriptorBufferInfo> buffer_infos(per_light_pmf_buf.size());
            for (uint32_t i = 0; i < headers.size(); ++i) {
                if (headers[i].flag & LightFlagBits::Sky) {
                    uint32_t pmf_id = headers[i].ext;
                    buffer_infos[pmf_id].buffer = per_light_pmf_buf[pmf_id]->buffer;
                    buffer_infos[pmf_id].offset = 0;
                    buffer_infos[pmf_id].range = VK_WHOLE_SIZE;
                }
                //
            }
            pmf_array_param_block.write_buffers("a", std::move(buffer_infos), 0, write_array);
        }
        vkUpdateDescriptorSets(device, write_array.writes.size(), write_array.writes.data(), 0, nullptr);
    }
}

GPULightSystem ::~GPULightSystem()
{
    //
    vkDestroySampler(device, equal_area_sampler, nullptr);
}

} // namespace ks
