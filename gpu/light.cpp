#include "light.h"

namespace ks
{

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

enum class LightSystemGlobalBindings : uint32_t
{
    Uniforms = 0,
    Headers = 1,
    PMF = 2,
};

GPULightSystem::GPULightSystem(LightPointers light_ptrs_, const vk::Context &ctx, uint32_t max_frames_in_flight)
    : light_ptrs(std::move(light_ptrs_)), device(ctx.device), allocator(ctx.allocator)
{
    // TODO: for now we assume that all skylights are at the beginning.
    uniforms.num_skylights = 0;
    for (uint32_t i = 0; i < (uint32_t)light_ptrs.lights.size(); ++i) {
        const EqualAreaSkyLight *sky = dynamic_cast<const EqualAreaSkyLight *>(light_ptrs.lights[i]);
        if (sky) {
            ++uniforms.num_skylights;
        } else {
            break;
        }
    }

    headers.resize(light_ptrs.lights.size());
    skylights_unit_powers.resize(uniforms.num_skylights);
    skylight_texture_indices.resize(uniforms.num_skylights);
    uint32_t tex_count = 0;
    for (uint32_t i = 0; i < (uint32_t)light_ptrs.lights.size(); ++i) {
        if (i < uniforms.num_skylights) {
            skylight_texture_indices[i] = tex_count;
            const EqualAreaSkyLight *sky = dynamic_cast<const EqualAreaSkyLight *>(light_ptrs.lights[i]);
            skylights_unit_powers[i] = sky->unit_power().mean();

            headers[i] = make_skylight_header(*sky, tex_count);

            ctx.allocator->stage_session([&](vk::Allocator &self) {
                per_light_pmf_buf.emplace_back(ctx.allocator,
                                               VkBufferCreateInfo{
                                                   .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               },
                                               VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), sky->pmf.bins);
            });

            VkImageCreateInfo image_ci{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VkImageType::VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R16G16B16A16_SFLOAT,
                .extent = VkExtent3D(sky->res, sky->res, 1),
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
                        uint32_t n_texels = sqr(sky->res);
                        for (uint32_t t = 0; t < n_texels; ++t) {
                            for (uint32_t c = 0; c < 3; ++c) {
                                *(fp16_ptr++) = convert_float_to_half(sky->texels[t][c]);
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
            if (const DirectionalLight *l = dynamic_cast<const DirectionalLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_dirlight_header(*l);
            } else if (const PointLight *l = dynamic_cast<const PointLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_pointlight_header(*l);
            }
        }
    }

    uniforms_buf.resize(max_frames_in_flight);
    headers_buf.resize(max_frames_in_flight);
    pmf_buf.resize(max_frames_in_flight);
    for (uint32_t i = 0; i < max_frames_in_flight; ++i) {
        uniforms_buf[i] = vk::AutoRelease<vk::FrequentUploadBuffer>(
            ctx.allocator, VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = sizeof(LightSystemUniforms),
                                              .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT});

        headers_buf[i] = vk::AutoRelease<vk::FrequentUploadBuffer>(
            ctx.allocator, VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = sizeof(LightHeader) * light_ptrs.lights.size(),
                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

        pmf_buf[i] = vk::AutoRelease<vk::FrequentUploadBuffer>(
            ctx.allocator, VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = sizeof(AliasTable::Bin) * light_ptrs.lights.size(),
                                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    }

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
        global_param_block_meta = vk::ParameterBlockMeta(ctx.device, max_frames_in_flight, std::move(helper));
        global_param_block.resize(max_frames_in_flight);
        global_param_block_meta.allocate_blocks(max_frames_in_flight, global_param_block);
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

void GPULightSystem::update(const AABB3 &scene_bound)
{
    uint32_t n_skylights = uniforms.num_skylights;

    std::vector<float> powers(light_ptrs.lights.size());
    float scene_radius = 0.5f * scene_bound.extents().norm();
    for (uint32_t i = 0; i < (uint32_t)light_ptrs.lights.size(); ++i) {
        if (i < n_skylights) {
            headers[i] = make_skylight_header(*dynamic_cast<const EqualAreaSkyLight *>(light_ptrs.lights[i]),
                                              skylight_texture_indices[i]);

            powers[i] = skylights_unit_powers[i] * pi * sqr(scene_radius);
        } else {
            if (const DirectionalLight *l = dynamic_cast<const DirectionalLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_dirlight_header(*l);
            } else if (const PointLight *l = dynamic_cast<const PointLight *>(light_ptrs.lights[i]); l) {
                headers[i] = make_pointlight_header(*l);
            }
            powers[i] = light_ptrs.lights[i]->power(scene_bound).mean();
        }
    }

    pmf = AliasTable(powers);
}

void GPULightSystem::upload(uint32_t curr_frame_in_flight_index, VkCommandBuffer cb)
{
    {
        vk::ParameterWriteArray write_array;
        global_param_block[curr_frame_in_flight_index].write_buffer(
            "uniforms",
            VkDescriptorBufferInfo{
                .buffer = uniforms_buf[curr_frame_in_flight_index]->dest.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            write_array);
        global_param_block[curr_frame_in_flight_index].write_buffer(
            "headers",
            VkDescriptorBufferInfo{
                .buffer = headers_buf[curr_frame_in_flight_index]->dest.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            write_array);
        global_param_block[curr_frame_in_flight_index].write_buffer(
            "pmf",
            VkDescriptorBufferInfo{
                .buffer = pmf_buf[curr_frame_in_flight_index]->dest.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            write_array);
        vkUpdateDescriptorSets(device, write_array.writes.size(), write_array.writes.data(), 0, nullptr);
    }

    allocator->map(*uniforms_buf[curr_frame_in_flight_index], true, [&](std::byte *ptr) {
        auto &dst = *reinterpret_cast<LightSystemUniforms *>(ptr);
        dst = uniforms;
    });
    allocator->map(*headers_buf[curr_frame_in_flight_index], true, [&](std::byte *ptr) {
        auto *dst = reinterpret_cast<LightHeader *>(ptr);
        std::copy(headers.begin(), headers.end(), dst);
    });
    allocator->map(*pmf_buf[curr_frame_in_flight_index], true, [&](std::byte *ptr) {
        auto *dst = reinterpret_cast<AliasTable::Bin *>(ptr);
        std::copy(pmf.bins.begin(), pmf.bins.end(), dst);
    });

    uniforms_buf[curr_frame_in_flight_index]->upload(cb);
    headers_buf[curr_frame_in_flight_index]->upload(cb);
    pmf_buf[curr_frame_in_flight_index]->upload(cb);
}

} // namespace ks
