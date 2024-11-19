#include "small_pt.h"

#include "../log_util.h"

namespace ks
{

enum class GlobalBindings : uint32_t
{
    Camera = 0,
    OutImage = 1 // Ray tracer output image
};

struct GPUSmallPTGlobalUniforms
{
    mat4 world_to_proj;
    mat4 camera_to_world;
    mat4 proj_to_camera;
    uint32_t bounces;
    uint32_t full_render_width;
    uint32_t full_render_height;
    uint32_t crop_start_x;
    uint32_t crop_start_y;
    uint32_t spp;
    uint32_t spp_interval_start;
    uint32_t spp_interval_end;
    uint32_t rng_seed;
    uint32_t padding[7]; // pad to vec4 boundary for now...
    //  NOTE: dont forget to check alignment
};

GPUSmallPT::GPUSmallPT(const vk::Context &ctx, slang::ISession &slang_session, const GPUScene &gpu_scene)
    : scene(&gpu_scene), allocator(ctx.allocator.get()), device(ctx.device)
{
    constexpr VkShaderStageFlags ray_trace_shader_stages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    vk::DescriptorSetHelper helper;
    helper.add_binding("camera", {.binding = (uint32_t)GlobalBindings::Camera,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                  .descriptorCount = 1,
                                  .stageFlags = ray_trace_shader_stages});
    helper.add_binding("out_image", {.binding = (uint32_t)GlobalBindings::OutImage,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                     .descriptorCount = 1,
                                     .stageFlags = ray_trace_shader_stages});
    data_generator_param_block_meta = vk::ParameterBlockMeta(ctx.device, 1, std::move(helper));
    data_generator_param_block = data_generator_param_block_meta.allocate_block();

    VkDescriptorSetLayout data_generator_global_binding_desc_set_layout =
        data_generator_param_block_meta.desc_set_layout;

    enum class StageIndices : uint32_t
    {
        Raygen,
        Miss,
        ShadowMiss,
        ClosestHit,
        AnyHit,
        ShaderGroupCount
    };

    std::array<std::string, (uint32_t)StageIndices::ShaderGroupCount> entry_point_names;
    entry_point_names[(uint32_t)StageIndices::Raygen] = "ray_gen_shader";
    entry_point_names[(uint32_t)StageIndices::Miss] = "miss_shader";
    entry_point_names[(uint32_t)StageIndices::ShadowMiss] = "shadow_miss_shader";
    entry_point_names[(uint32_t)StageIndices::ClosestHit] = "closest_hit_shader";
    entry_point_names[(uint32_t)StageIndices::AnyHit] = "any_hit_shader";

    CompiledSlangShader compiled_slang_shader(slang_session, ctx.device, "gpu_small_pt.slang", entry_point_names);

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> rt_shader_groups;

    // All stages
    std::array<VkPipelineShaderStageCreateInfo, (uint32_t)StageIndices::ShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};

    // Raygen
    stage.pName = entry_point_names[(uint32_t)StageIndices::Raygen].c_str();
    stage.module = compiled_slang_shader.shader_modules[(uint32_t)StageIndices::Raygen];
    stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[(uint32_t)StageIndices::Raygen] = stage;
    // Miss
    stage.pName = entry_point_names[(uint32_t)StageIndices::Miss].c_str();
    stage.module = compiled_slang_shader.shader_modules[(uint32_t)StageIndices::Miss];
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[(uint32_t)StageIndices::Miss] = stage;

    // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion
    // has been found
    stage.pName = entry_point_names[(uint32_t)StageIndices::ShadowMiss].c_str();
    stage.module = compiled_slang_shader.shader_modules[(uint32_t)StageIndices::ShadowMiss];
    stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[(uint32_t)StageIndices::ShadowMiss] = stage;

    // Hit Group - Closest Hit
    stage.pName = entry_point_names[(uint32_t)StageIndices::ClosestHit].c_str();
    stage.module = compiled_slang_shader.shader_modules[(uint32_t)StageIndices::ClosestHit];
    stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[(uint32_t)StageIndices::ClosestHit] = stage;

    // Hit Group - Any Hit
    stage.pName = entry_point_names[(uint32_t)StageIndices::AnyHit].c_str();
    stage.module = compiled_slang_shader.shader_modules[(uint32_t)StageIndices::AnyHit];
    stage.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[(uint32_t)StageIndices::AnyHit] = stage;

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    // Raygen
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = (uint32_t)StageIndices::Raygen;
    rt_shader_groups.push_back(group);

    // Miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = (uint32_t)StageIndices::Miss;
    rt_shader_groups.push_back(group);

    // Shadow hit Miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = (uint32_t)StageIndices::ShadowMiss;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    rt_shader_groups.push_back(group);

    // closest/any hit shader
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = (uint32_t)StageIndices::ClosestHit;
    group.anyHitShader = (uint32_t)StageIndices::AnyHit;
    rt_shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    // VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    //                                     VK_SHADER_STAGE_MISS_BIT_KHR,
    //                                 0, sizeof(PushConstantRay)};

    VkPipelineLayoutCreateInfo pipeline_layout_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    // pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    // pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::array<VkDescriptorSetLayout, 2> rt_desc_layouts = {gpu_scene.get_ray_tracing_desc_set_layout(),
                                                            data_generator_global_binding_desc_set_layout};
    pipeline_layout_ci.setLayoutCount = static_cast<uint32_t>(rt_desc_layouts.size());
    pipeline_layout_ci.pSetLayouts = rt_desc_layouts.data();

    vkCreatePipelineLayout(ctx.device, &pipeline_layout_ci, nullptr, &rt_pipeline_layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rt_pipeline_ci{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rt_pipeline_ci.stageCount = static_cast<uint32_t>(stages.size()); // Stages are shaders
    rt_pipeline_ci.pStages = stages.data();

    // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
    // two miss shader groups, and one hit group.
    rt_pipeline_ci.groupCount = static_cast<uint32_t>(rt_shader_groups.size());
    rt_pipeline_ci.pGroups = rt_shader_groups.data();

    // We only need a recursion level of 2 for PT (normal rays and shadow rays).
    // Bounces are implemented as loops in the raygen shader.
    rt_pipeline_ci.maxPipelineRayRecursionDepth = 2; // Ray depth
    rt_pipeline_ci.layout = rt_pipeline_layout;

    vkCreateRayTracingPipelinesKHR(ctx.device, {}, {}, 1, &rt_pipeline_ci, nullptr, &rt_pipeline);

    // Requesting ray tracing properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rt_properties};
    prop2.pNext = &rt_properties;
    vkGetPhysicalDeviceProperties2(ctx.physical_device, &prop2);

    // Spec only guarantees 1 level of "recursion". Check for that sad possibility here.
    if (rt_properties.maxRayRecursionDepth <= 1) {
        throw std::runtime_error("Device fails to support ray recursion (m_rtProperties.maxRayRecursionDepth <= 1)");
    }

    // for (auto &s : stages) {
    //     vkDestroyShaderModule(ctx.device, s.module, nullptr);
    // }

    // Create SBT
    sbt_wrapper.init(ctx.device, ctx.main_queue_family_index, ctx.allocator.get(), rt_properties);
    sbt_wrapper.create(rt_pipeline, rt_pipeline_ci);

    global_params_buf = vk::AutoRelease<vk::FrequentUniformBuffer>(
        ctx.allocator,
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(GPUSmallPTGlobalUniforms)});

    cp = vk::CommandPool(device, ctx.main_queue_family_index);
}

GPUSmallPT::~GPUSmallPT()
{
    vkDestroyPipeline(device, rt_pipeline, nullptr);
    vkDestroyPipelineLayout(device, rt_pipeline_layout, nullptr);
}

void GPUSmallPT::run(const GPUSmallPTInput &in)
{
    vk::ParameterWriteArray write_array;
    data_generator_param_block.write_buffer(
        "camera", VkDescriptorBufferInfo{.buffer = global_params_buf->dest.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);
    data_generator_param_block.write_image("out_image",
                                           VkDescriptorImageInfo{
                                               .imageView = in.render_target.view,
                                               .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                           },
                                           write_array);
    write_array.update_writes(device);

    int full_render_width = in.render_width;
    int full_render_height = in.render_height;
    int crop_render_start_x = 0;
    int crop_render_start_y = 0;
    int crop_render_width = full_render_width;
    int crop_render_height = full_render_height;
    if (in.crop_width > 0 && in.crop_height > 0) {
        crop_render_start_x = std::max(in.crop_start_x, 0);
        crop_render_start_y = std::max(in.crop_start_y, 0);
        crop_render_width = std::min(in.crop_width, full_render_width - crop_render_start_x);
        crop_render_height = std::min(in.crop_height, full_render_height - crop_render_start_y);
    }

    for (int s_interval_start = 0; s_interval_start < in.spp; s_interval_start += in.spp_prog_interval) {
        int spp_batch = std::min(in.spp_prog_interval, in.spp - s_interval_start);
        int s_interval_end = s_interval_start + spp_batch;

        // Get a cb and start work.
        VkCommandBuffer cb = cp.create_command_buffer();

        // Upload uniforms
        allocator->map(*global_params_buf, true, [&](std::byte *ptr) {
            GPUSmallPTGlobalUniforms &global = *reinterpret_cast<GPUSmallPTGlobalUniforms *>(ptr);
            // Note: slang by default uses row-major.
            global.world_to_proj = in.camera->world_to_proj.m.transpose();
            global.proj_to_camera = in.camera->proj_to_camera.m.transpose();
            global.camera_to_world = in.camera->camera_to_world.m.transpose();
            global.bounces = in.bounces;
            global.full_render_width = full_render_width;
            global.full_render_height = full_render_height;
            global.spp = in.spp;
            global.rng_seed = in.rng_seed;
            global.crop_start_x = crop_render_start_x;
            global.crop_start_y = crop_render_start_y;
            global.spp_interval_start = s_interval_start;
            global.spp_interval_end = s_interval_end;
        });
        global_params_buf->upload(cb, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Bind pipeline, descriptors, push constants
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline);

        std::array<VkDescriptorSet, 2> desc_sets = {scene->get_ray_tracing_desc_set(),
                                                    data_generator_param_block.desc_set};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline_layout, 0,
                                (uint32_t)desc_sets.size(), desc_sets.data(), 0, nullptr);
        // vkCmdPushConstants

        // Trace
        auto &regions = sbt_wrapper.get_regions();
        vkCmdTraceRaysKHR(cb, &regions[0], &regions[1], &regions[2], &regions[3], crop_render_width, crop_render_height,
                          1);

        // Call back (gpu)
        if (in.prog_interval_callback_gpu) {
            in.prog_interval_callback_gpu(in.render_target, cb, s_interval_end);
        }

        // Submit
        cp.submit_and_wait(cb);

        // Call back (cpu)
        if (in.prog_interval_callback_cpu) {
            in.prog_interval_callback_cpu(s_interval_end);
        }
    }
}

void gpu_small_pt(const ks::ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    GPUContext &gpu = get_gpu_context();

    const CompoundMeshAsset *compound_mesh_asset =
        args.asset_table().get<CompoundMeshAsset>(args.load_string("compound_object"));

    GPUScene gpu_scene(*compound_mesh_asset, gpu.vkctx);
    gpu_scene.prepare_for_ray_tracing();

    GPUSmallPT gpu_small_pt(gpu.vkctx, *gpu.slang_session, gpu_scene);
    // gpu_small_pt.scene = &gpu_scene;

    GPUSmallPTInput gpu_small_pt_in;
    gpu_small_pt_in.bounces = args.load_integer("bounces");
    gpu_small_pt_in.clamp_indirect = args.load_float("clamp_indirect", 10.0f);
    gpu_small_pt_in.render_width = args.load_integer("render_width");
    gpu_small_pt_in.render_height = args.load_integer("render_height");
    gpu_small_pt_in.crop_start_x = args.load_integer("crop_start_x", 0);
    gpu_small_pt_in.crop_start_y = args.load_integer("crop_start_y", 0);
    gpu_small_pt_in.crop_width = args.load_integer("crop_width", 0);
    gpu_small_pt_in.crop_height = args.load_integer("crop_height", 0);
    gpu_small_pt_in.spp = args.load_integer("spp");
    gpu_small_pt_in.scale_ray_diff = args.load_bool("scale_ray_diff", true);
    gpu_small_pt_in.rng_seed = args.load_integer("rng_seed", 0);
    gpu_small_pt_in.spp_prog_interval = args.load_integer("spp_prog_interval", 32);

    vk::AutoRelease<vk::ImageWithView> render_target;
    gpu.vkctx.allocator->stage_session([&](vk::Allocator &self) {
        VkImageCreateInfo render_target_img_ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .extent = VkExtent3D(gpu_small_pt_in.render_width, gpu_small_pt_in.render_height, 1),
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        };
        vk::Image render_target_img = self.create_and_transit_image(
            render_target_img_ci, VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), VK_IMAGE_LAYOUT_GENERAL);

        render_target = vk::AutoRelease<vk::ImageWithView>(
            self.create_image_with_view(render_target_img, vk::simple_view_info_from_image_info(
                                                               render_target_img_ci, render_target_img, false)),
            gpu.vkctx.allocator);
    });
    gpu_small_pt_in.render_target = *render_target;

    vk::AutoRelease<vk::Image> render_target_readback(
        gpu.vkctx.allocator->create_image(
            VkImageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                .extent = VkExtent3D(gpu_small_pt_in.render_width, gpu_small_pt_in.render_height, 1),
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_LINEAR,
                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            },
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT),
        gpu.vkctx.allocator);

    const CameraAnimation *camera_anim = nullptr;
    std::unique_ptr<Camera> camera_static;
    uint32_t n_frames = 0;
    uint32_t frame_offset = 0;
    uint32_t frame_count = 0;
    if (args.contains("camera_animation")) {
        camera_anim = args.asset_table().get<CameraAnimation>(args.load_string("camera_animation"));
        n_frames = camera_anim->n_frames();
        frame_offset = (uint32_t)args.load_integer("camera_animation_offset", 0);
        frame_count = (uint32_t)args.load_integer("camera_animation_count", 0);
    } else {
        camera_static = create_camera(args["camera"]);
        n_frames = 1;
    }

    uint32_t frame_start = frame_offset;
    uint32_t frame_end = frame_count == 0 ? n_frames : frame_offset + frame_count;
    for (uint32_t frame_idx = frame_start; frame_idx < frame_end; ++frame_idx) {
        get_default_logger().info("Frame [{}/{}] | Start", frame_idx + 1, n_frames);
        get_default_logger().flush();

        Camera camera_frame;
        if (camera_anim) {
            float anim_time = camera_anim->duration() * (frame_idx + 0.5f) / (float)n_frames;
            camera_frame = camera_anim ? camera_anim->eval(anim_time) : *camera_static;
        } else {
            camera_frame = *camera_static;
        }
        gpu_small_pt_in.camera = &camera_frame;

        auto render_start = std::chrono::steady_clock::now();
        auto interval_render_start = render_start;

        gpu_small_pt_in.prog_interval_callback_gpu = [&](const vk::ImageWithView &rt, VkCommandBuffer cb,
                                                         int spp_finished) {
            // Read back and save here.
            {
                std::array<VkImageMemoryBarrier, 2> img_barriers;
                // rt_to_transfer_src;
                img_barriers[0] = VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,

                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = render_target->image,
                    .subresourceRange =
                        VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                        },
                };
                // readback_image_to_transfer_dst
                img_barriers[1] = VkImageMemoryBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_NONE,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,

                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = render_target_readback->image,
                    .subresourceRange =
                        VkImageSubresourceRange{
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1,
                        },
                };
                vk::pipeline_barrier(
                    cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VkDependencyFlags(0), {}, {}, img_barriers);
            }

            VkImageCopy copy;
            copy.srcOffset = {0, 0, 0};
            copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.srcSubresource.mipLevel = 0;
            copy.srcSubresource.baseArrayLayer = 0;
            copy.srcSubresource.layerCount = 1;
            copy.dstOffset = {0, 0, 0};
            copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.dstSubresource.mipLevel = 0;
            copy.dstSubresource.baseArrayLayer = 0;
            copy.dstSubresource.layerCount = 1;
            copy.extent = {(uint32_t)gpu_small_pt_in.render_width, (uint32_t)gpu_small_pt_in.render_height, 1};
            vkCmdCopyImage(cb, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, render_target_readback->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            VkImageMemoryBarrier readback_image_to_general{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,

                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = render_target_readback->image,
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, VkDependencyFlags(0), {},
                                 {}, {&readback_image_to_general, 1});

            VkImageMemoryBarrier rt_to_general{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,

                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = render_target->image,
                .subresourceRange =
                    VkImageSubresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                 VkDependencyFlags(0), {}, {}, {&rt_to_general, 1});
        };

        gpu_small_pt_in.prog_interval_callback_cpu = [&](int spp_finished) {
            auto interval_render_end = std::chrono::steady_clock::now();
            float interval_render_time =
                std::chrono::duration<float>(interval_render_end - interval_render_start).count();

            get_default_logger().info("Frame [{}/{}] | [{}/{}] spp interval took: {:.2f} sec", frame_idx + 1, n_frames,
                                      spp_finished, gpu_small_pt_in.spp, interval_render_time);
            get_default_logger().flush();

            fs::path save_path_prefix = task_dir / string_format("small_pt_spp%d", spp_finished);
            std::string save_path_postfix = n_frames > 1 ? string_format("%06u", frame_idx) : std::string();
            fs::path save_path = save_path_prefix;
            save_path += save_path_postfix.empty() ? ".exr" : string_format("_%s.exr", save_path_postfix.c_str());

            // Get layout of the image (including row pitch)
            VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
            VkSubresourceLayout subResourceLayout;
            vkGetImageSubresourceLayout(gpu.vkctx.device, render_target_readback->image, &subResource,
                                        &subResourceLayout);

            gpu.vkctx.allocator->map(render_target_readback->allocation, false, [&](std::byte *ptr) {
                uint32_t w = gpu_small_pt_in.render_width;
                uint32_t h = gpu_small_pt_in.render_height;
                if (subResourceLayout.offset == 0 && subResourceLayout.rowPitch == sizeof(float[4]) * w) {
                    ks::save_to_exr(ptr, false, w, h, 4, save_path);
                } else {
                    std::vector<std::byte> cpu_buf(w * h * sizeof(float[4]));
                    ptr += subResourceLayout.offset;
                    for (uint32_t y = 0; y < h; ++y) {
                        std::copy_n(ptr, sizeof(float[4]) * w, cpu_buf.begin() + (sizeof(float[4]) * (y * w)));
                        ptr += subResourceLayout.rowPitch;
                    }
                    ks::save_to_exr(cpu_buf.data(), false, w, h, 4, save_path);
                }
            });

            interval_render_start = interval_render_end;
        };

        gpu_small_pt.run(gpu_small_pt_in);
        auto render_end = std::chrono::steady_clock::now();
        std::chrono::duration<float> render_time_sec = render_end - render_start;
        get_default_logger().info("Frame [{}/{}] | Rendering time: {:.2f} sec", frame_idx + 1, n_frames,
                                  render_time_sec.count());
        get_default_logger().flush();
    }
}

} // namespace ks