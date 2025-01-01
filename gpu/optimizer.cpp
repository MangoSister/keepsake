#include "optimizer.h"

namespace ks
{

AdamOptimizerShared::AdamOptimizerShared(GPUContext &gpu, uint32_t max_insts)
{
    vk::DescriptorSetHelper desc_set_helper;
    desc_set_helper.add_binding("parameters",
                                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    desc_set_helper.add_binding("gradients",
                                {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    desc_set_helper.add_binding("first_moments",
                                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    desc_set_helper.add_binding("second_moments",
                                {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});

    param_block_meta = vk::ParameterBlockMeta(gpu.vk.device, max_insts, std::move(desc_set_helper));

    std::array<VkPushConstantRange, 1> pc_ranges{VkPushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(AdamOptimizerHyperParams)}};

    const std::string entry_point_name = "adam_optimizer_step";
    CompiledSlangShader compiled_slang_shader(*gpu.slang_session, gpu.vk.device, "ks", {&entry_point_name, 1});

    VkPipelineLayoutCreateInfo pipeline_layout_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                    .setLayoutCount = 1,
                                                    .pSetLayouts = &param_block_meta.desc_set_layout,
                                                    .pushConstantRangeCount = (uint32_t)pc_ranges.size(),
                                                    .pPushConstantRanges = pc_ranges.data()};

    vk::vk_check(vkCreatePipelineLayout(gpu.vk.device, &pipeline_layout_info, nullptr, &pipeline_layout));

    VkComputePipelineCreateInfo pipelineCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.module = compiled_slang_shader.shader_modules[0];
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.pName = entry_point_name.c_str();
    pipelineCreateInfo.layout = pipeline_layout;

    vk::vk_check(
        vkCreateComputePipelines(gpu.vk.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));
}

AdamOptimizerShared::~AdamOptimizerShared()
{
    vkDestroyPipeline(param_block_meta.device, pipeline, nullptr);
    vkDestroyPipelineLayout(param_block_meta.device, pipeline_layout, nullptr);
}

AdamOptimizer::AdamOptimizer(AdamOptimizerHyperParams hparams, uint32_t num_parameters, vk::Buffer parameters_buf,
                             vk::Buffer gradients_buf, AdamOptimizerShared &shared, vk::Context &ctx,
                             VkCommandBuffer ext_cb)
    : hparams(std::move(hparams)), num_parameters(num_parameters), shared(&shared)
{
    hparams.iter = 1;

    first_moments =
        vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                    VkBufferCreateInfo{
                                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                        .size = sizeof(float) * num_parameters,
                                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    },
                                    VMA_MEMORY_USAGE_AUTO, 0);

    second_moments =
        vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                    VkBufferCreateInfo{
                                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                        .size = sizeof(float) * num_parameters,
                                        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    },
                                    VMA_MEMORY_USAGE_AUTO, 0);

    param_block = shared.param_block_meta.allocate_block();

    vk::ParameterWriteArray write_array;
    param_block.write_buffer(
        "parameters", VkDescriptorBufferInfo{.buffer = parameters_buf.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);
    param_block.write_buffer(
        "gradients", VkDescriptorBufferInfo{.buffer = gradients_buf.buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);
    param_block.write_buffer(
        "first_moments", VkDescriptorBufferInfo{.buffer = first_moments->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);
    param_block.write_buffer(
        "second_moments", VkDescriptorBufferInfo{.buffer = second_moments->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);
    write_array.update_writes(ctx.device);

    // Zero out the moment buffers.
    if (ext_cb != VK_NULL_HANDLE) {
        vkCmdFillBuffer(ext_cb, first_moments->buffer, 0, VK_WHOLE_SIZE, (uint32_t)0);
        vkCmdFillBuffer(ext_cb, second_moments->buffer, 0, VK_WHOLE_SIZE, (uint32_t)0);
    } else {
        ctx.submit_once([&](VkCommandBuffer cb) {
            vkCmdFillBuffer(cb, first_moments->buffer, 0, VK_WHOLE_SIZE, (uint32_t)0);
            vkCmdFillBuffer(cb, second_moments->buffer, 0, VK_WHOLE_SIZE, (uint32_t)0);
        });
    }
}

void AdamOptimizer::step(VkCommandBuffer cb)
{
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, shared->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, shared->pipeline_layout, 0, 1, &param_block.desc_set, 0,
                            nullptr);
    vkCmdPushConstants(cb, shared->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdamOptimizerHyperParams),
                       &hparams);
    vk::dispatch_compute<1024, 1, 1>(cb, {num_parameters, 1, 1});
}

} // namespace ks