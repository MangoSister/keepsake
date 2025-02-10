#include "ksvk.h"

#include "../log_util.h"

#include <array>
#include <filesystem>
namespace fs = std::filesystem;

// #define IMGUI_ENABLE_FREETYPE
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>
#include <vulkan/utility/vk_format_utils.h>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#define VOLK_IMPLEMENTATION
#include <volk.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Don't reorder these headers...
// clang-format off
#include <windows.h>
#include <vulkan/vulkan_win32.h>
// clang-format on
#elif defined(__linux__)

#endif

namespace ks
{

GPUContext g_gpu_context{};
std::mutex mutex_init_gpu;
bool gpu_initialized = false;

void init_gpu(std::span<const char *> shader_search_paths, int vk_device, const vk::ContextArgs &vkctx_args)
{
    // Make sure no stupid things happen...
    std::scoped_lock lock(mutex_init_gpu);

    if (gpu_initialized) {
        return;
    }
    gpu_initialized = true;

    g_gpu_context.vk.create_instance(vkctx_args);

    // TODO: improve this...
    // If enabled swapchain, we test surface support by creating a temp window and surface and destroy right after
    // this...
    GLFWwindow *test_window = nullptr;
    VkSurfaceKHR test_surface = VK_NULL_HANDLE;
    if (vkctx_args.swapchain) {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        test_window = glfwCreateWindow(640, 480, "test", nullptr, nullptr);
        vk::vk_check(glfwCreateWindowSurface(g_gpu_context.vk.instance, test_window, nullptr, &test_surface));
    }
    auto compatibles = g_gpu_context.vk.query_compatible_devices(vkctx_args, test_surface);

    if (vkctx_args.swapchain) {
        glfwDestroyWindow(test_window);
        vkDestroySurfaceKHR(g_gpu_context.vk.instance, test_surface, nullptr);
    }

    if (compatibles.empty()) {
        get_default_logger().critical("No compatible vulkan devices.");
        std::abort();
    }

    ASSERT(vk_device >= 0 && vk_device < compatibles.size());
    g_gpu_context.vk.create_device(vkctx_args, compatibles[vk_device]);

    // First we need to create slang global session with work with the Slang API.
    slang_check(slang::createGlobalSession(g_gpu_context.slang_global_session.writeRef()));

    // Next we create a compilation session to generate SPIRV code from Slang source.
    std::vector<const char *> search_paths;
    search_paths.insert(search_paths.end(), shader_search_paths.begin(), shader_search_paths.end());
    search_paths.push_back(KS_SHADER_DIR);

    slang::SessionDesc sessionDesc = {};
    sessionDesc.searchPathCount = search_paths.size();
    sessionDesc.searchPaths = search_paths.data();

    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = g_gpu_context.slang_global_session->findProfile("spirv_1_6");
    targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    //
    std::array<slang::CompilerOptionEntry, 1> compiler_option_entries;
    compiler_option_entries[0].name = slang::CompilerOptionName::VulkanUseEntryPointName;
    compiler_option_entries[0].value.kind = slang::CompilerOptionValueKind::Int;
    compiler_option_entries[0].value.intValue0 = 1;
    compiler_option_entries[0].value.intValue1 = 1;

    sessionDesc.compilerOptionEntries = compiler_option_entries.data();
    sessionDesc.compilerOptionEntryCount = (uint32_t)compiler_option_entries.size();

    // Note: on CPU side Eigen uses column-major, but we will follow the default row-major matrix layout for Slang.
    slang_check(g_gpu_context.slang_global_session->createSession(sessionDesc, g_gpu_context.slang_session.writeRef()));
}

// Convenient function to create argument with support for usual features used by ks and applications such as ray
// tracing, bindless, atomics, etc.
// Validation can have performance overhead, but usually we want it for testing both in debug and release build until we
// are very confident...
vk::ContextArgs get_default_context_args(bool validation, bool swapchain)
{
    vk::ContextArgs ctx_args{};
    ctx_args.api_version_major = 1;
    ctx_args.api_version_minor = 3;
    if (validation) {
        ctx_args.enable_validation();
    }
    if (swapchain) {
        ctx_args.enable_swapchain();
    }
    //
    ctx_args.device_features.features.samplerAnisotropy = VK_TRUE;
    ctx_args.device_features.features.shaderInt64 = VK_TRUE;
    ctx_args.device_features.features.shaderFloat64 = VK_TRUE;

    ctx_args.add_device_feature<VkPhysicalDeviceDynamicRenderingFeatures>() = VkPhysicalDeviceDynamicRenderingFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, .dynamicRendering = VK_TRUE};

    ctx_args.add_device_feature<VkPhysicalDeviceVulkan11Features>() = VkPhysicalDeviceVulkan11Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .variablePointersStorageBuffer = VK_TRUE,
        .variablePointers = VK_TRUE,
    };

    ctx_args.add_device_feature<VkPhysicalDeviceVulkan12Features>() = VkPhysicalDeviceVulkan12Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
        .shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
        .shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
        .shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
        .shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
        .shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
        .shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
        .descriptorBindingPartiallyBound = VK_TRUE,
        .descriptorBindingVariableDescriptorCount = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
        .scalarBlockLayout = VK_TRUE,
        .timelineSemaphore = VK_TRUE, // used by debug printf
        .bufferDeviceAddress = VK_TRUE,
    };
    ctx_args.device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

    ctx_args.add_device_feature<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>() =
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
            .shaderBufferFloat32Atomics = VK_TRUE,
            .shaderBufferFloat32AtomicAdd = VK_TRUE,
            .shaderSharedFloat32Atomics = VK_TRUE,
            .shaderSharedFloat32AtomicAdd = VK_TRUE,
            .shaderImageFloat32Atomics = VK_TRUE,
            .shaderImageFloat32AtomicAdd = VK_TRUE,
        };
    ctx_args.device_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    ctx_args.add_device_feature<VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT>() =
        VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
            .shaderImageInt64Atomics = VK_TRUE};
    ctx_args.device_extensions.push_back(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME);

    ctx_args.add_device_feature<VkPhysicalDeviceAccelerationStructureFeaturesKHR>() =
        VkPhysicalDeviceAccelerationStructureFeaturesKHR{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .accelerationStructure = VK_TRUE,
        };
    ctx_args.device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

    ctx_args.add_device_feature<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>() =
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .rayTracingPipeline = VK_TRUE,
            .rayTracingPipelineTraceRaysIndirect = VK_TRUE,
        };
    ctx_args.device_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);

    ctx_args.device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    return ctx_args;
}

void init_gpu(std::span<const char *> shader_search_paths, int vk_device, bool vk_validation, bool vk_swapchain)
{
    vk::ContextArgs vkctx_args = get_default_context_args(vk_validation, vk_swapchain);
    init_gpu(shader_search_paths, vk_device, vkctx_args);
}

GPUContext &get_gpu_context()
{
    ASSERT(gpu_initialized);
    return g_gpu_context;
}

namespace vk
{

//-----------------------------------------------------------------------------
// [Debug util (adapted from nvvk)]
//-----------------------------------------------------------------------------
bool DebugUtil::s_enabled = false;

// Local extension functions point
PFN_vkCmdBeginDebugUtilsLabelEXT DebugUtil::s_vkCmdBeginDebugUtilsLabelEXT = 0;
PFN_vkCmdEndDebugUtilsLabelEXT DebugUtil::s_vkCmdEndDebugUtilsLabelEXT = 0;
PFN_vkCmdInsertDebugUtilsLabelEXT DebugUtil::s_vkCmdInsertDebugUtilsLabelEXT = 0;
PFN_vkSetDebugUtilsObjectNameEXT DebugUtil::s_vkSetDebugUtilsObjectNameEXT = 0;

void DebugUtil::setup(VkDevice device)
{
    m_device = device;
    // Get the function pointers
    if (s_enabled == false) {
        s_vkCmdBeginDebugUtilsLabelEXT =
            (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdBeginDebugUtilsLabelEXT");
        s_vkCmdEndDebugUtilsLabelEXT =
            (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdEndDebugUtilsLabelEXT");
        s_vkCmdInsertDebugUtilsLabelEXT =
            (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetDeviceProcAddr(device, "vkCmdInsertDebugUtilsLabelEXT");
        s_vkSetDebugUtilsObjectNameEXT =
            (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");

        s_enabled = s_vkCmdBeginDebugUtilsLabelEXT != nullptr && s_vkCmdEndDebugUtilsLabelEXT != nullptr &&
                    s_vkCmdInsertDebugUtilsLabelEXT != nullptr && s_vkSetDebugUtilsObjectNameEXT != nullptr;
    }
}

void DebugUtil::set_object_name(const uint64_t object, const std::string &name, VkObjectType t) const
{
    if (s_enabled) {
        VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, t, object,
                                        name.c_str()};
        s_vkSetDebugUtilsObjectNameEXT(m_device, &s);
    }
}

void DebugUtil::begin_label(VkCommandBuffer cmdBuf, const std::string &label)
{
    if (s_enabled) {
        VkDebugUtilsLabelEXT s{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        s_vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &s);
    }
}

void DebugUtil::end_label(VkCommandBuffer cmdBuf)
{
    if (s_enabled) {
        s_vkCmdEndDebugUtilsLabelEXT(cmdBuf);
    }
}

void DebugUtil::insert_label(VkCommandBuffer cmdBuf, const std::string &label)
{
    if (s_enabled) {
        VkDebugUtilsLabelEXT s{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        s_vkCmdInsertDebugUtilsLabelEXT(cmdBuf, &s);
    }
}

//-----------------------------------------------------------------------------
// [Memory allocation]
//-----------------------------------------------------------------------------

Allocator::Allocator(const VmaAllocatorCreateInfo &vma_info, uint32_t upload_queue_family_index, VkQueue upload_queue)
{
    device = vma_info.device;
    vk_check(vmaCreateAllocator(&vma_info, &vma));
    this->upload_queue = upload_queue;
    VkCommandPoolCreateInfo cmdPoolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdPoolCI.queueFamilyIndex = upload_queue_family_index;
    cmdPoolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    vk_check(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &upload_cp));
    upload_cb = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties physicalDeviceProps;
    vkGetPhysicalDeviceProperties(vma_info.physicalDevice, &physicalDeviceProps);
    min_uniform_buffer_offset_alignment = physicalDeviceProps.limits.minUniformBufferOffsetAlignment;
    min_storage_buffer_offset_alignment = physicalDeviceProps.limits.minStorageBufferOffsetAlignment;
    min_texel_buffer_offset_alignment = physicalDeviceProps.limits.minTexelBufferOffsetAlignment;
}

Allocator::~Allocator()
{
    if (vma == VK_NULL_HANDLE) {
        return;
    }
    clear_staging_buffer();

    vkDestroyCommandPool(device, upload_cp, nullptr);
    vmaDestroyAllocator(vma);
}

Buffer Allocator::create_buffer(const VkBufferCreateInfo &info_, VmaMemoryUsage usage, VmaAllocationCreateFlags flags,
                                const std::byte *data)
{
    Buffer buf;
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;

    VkBufferCreateInfo info = info_;
    if (data) {
        info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    vk_check(vmaCreateBuffer(vma, &info, &allocCI, &buf.buffer, &buf.allocation, nullptr));

    // Get the device address if requested
    if (info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buf.buffer;
        if (vkGetBufferDeviceAddress) {
            buf.address = vkGetBufferDeviceAddress(device, &info);
        } else if (vkGetBufferDeviceAddressKHR) {
            buf.address = vkGetBufferDeviceAddressKHR(device, &info);
        } else {
            ASSERT(false);
        }
    }

    if (data) {
        ASSERT(upload_cb);

        Buffer staging = create_staging_buffer(info.size, data, info.size);
        VkBufferCopy region;
        region.srcOffset = 0;

        region.dstOffset = 0;
        region.size = info.size;
        vkCmdCopyBuffer(upload_cb, staging.buffer, buf.buffer, 1, &region);
    }

    return buf;
}

Buffer Allocator::create_buffer_with_alignment(const VkBufferCreateInfo &info_, VkDeviceSize min_alignment,
                                               VmaMemoryUsage usage, VmaAllocationCreateFlags flags,
                                               const std::byte *data)
{
    Buffer buf;
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;

    VkBufferCreateInfo info = info_;
    if (data) {
        info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    vk_check(vmaCreateBufferWithAlignment(vma, &info, &allocCI, min_alignment, &buf.buffer, &buf.allocation, nullptr));

    // Get the device address if requested
    if (info.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buf.buffer;
        if (vkGetBufferDeviceAddress) {
            buf.address = vkGetBufferDeviceAddress(device, &info);
        } else if (vkGetBufferDeviceAddressKHR) {
            buf.address = vkGetBufferDeviceAddressKHR(device, &info);
        } else {
            ASSERT(false);
        }
    }

    if (data) {
        ASSERT(upload_cb);

        Buffer staging = create_staging_buffer(info.size, data, info.size);
        VkBufferCopy region;
        region.srcOffset = 0;

        region.dstOffset = 0;
        region.size = info.size;
        vkCmdCopyBuffer(upload_cb, staging.buffer, buf.buffer, 1, &region);
    }

    return buf;
}

TexelBuffer Allocator::create_texel_buffer(const VkBufferCreateInfo &info, VkBufferViewCreateInfo &buffer_view_info,
                                           VmaMemoryUsage usage, VmaAllocationCreateFlags flags, const std::byte *data)
{
    TexelBuffer tb;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;
    vk_check(vmaCreateBuffer(vma, &info, &allocCI, &tb.buffer, &tb.allocation, nullptr));

    if (data) {
        ASSERT(upload_cb);

        Buffer staging = create_staging_buffer(info.size, data, info.size);
        VkBufferCopy region;
        region.srcOffset = 0;
        region.dstOffset = 0;
        region.size = info.size;
        vkCmdCopyBuffer(upload_cb, staging.buffer, tb.buffer, 1, &region);
    }

    buffer_view_info.buffer = tb.buffer;
    vk_check(vkCreateBufferView(device, &buffer_view_info, nullptr, &tb.buffer_view));

    return tb;
}

FrequentUploadBuffer Allocator::create_frequent_upload_buffer(const VkBufferCreateInfo &info_)
{
    VkBufferCreateInfo info = info_;
    info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateFlags alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                           VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
    Buffer dest = create_buffer(info, VMA_MEMORY_USAGE_AUTO, alloc_flags);
    VkMemoryPropertyFlags memPropFlags;
    vmaGetAllocationMemoryProperties(vma, dest.allocation, &memPropFlags);

    if (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        // Allocation ended up in a mappable memory and is already mapped - write to it directly.
        return FrequentUploadBuffer{dest, Buffer(), info_.size};
    } else {
        // Allocation ended up in a non-mappable memory - a transfer using a staging buffer is required.
        Buffer staging = create_buffer(
            VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = info.size,
                               .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT},
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, nullptr);
        return FrequentUploadBuffer{dest, staging, info_.size};
    }
}

void FrequentUploadBuffer::upload(VkCommandBuffer cb, VkPipelineStageFlags dst_stage_mask) const
{
    if (!require_staging()) {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dest.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;

        vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_HOST_BIT, dst_stage_mask, 0, {}, {&barrier, 1}, {});
    } else {
        VkBufferMemoryBarrier barrier_mapping = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier_mapping.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier_mapping.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier_mapping.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_mapping.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_mapping.buffer = staging.buffer;
        barrier_mapping.offset = 0;
        barrier_mapping.size = VK_WHOLE_SIZE;

        vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {},
                             {&barrier_mapping, 1}, {});

        VkBufferCopy copy = {
            0,    // srcOffset
            0,    // dstOffset,
            size, // size
        };

        vkCmdCopyBuffer(cb, staging.buffer, dest.buffer, 1, &copy);

        VkBufferMemoryBarrier barrier_copy = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier_copy.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_copy.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT; // We created a uniform buffer
        barrier_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_copy.buffer = dest.buffer;
        barrier_copy.offset = 0;
        barrier_copy.size = VK_WHOLE_SIZE;

        vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage_mask, 0, {}, {&barrier_copy, 1}, {});
    }
}

std::byte *Allocator::map(VmaAllocation allocation)
{
    void *ptr = nullptr;
    vk_check(vmaMapMemory(vma, allocation, &ptr));
    return reinterpret_cast<std::byte *>(ptr);
}

void Allocator::unmap(VmaAllocation allocation) { vmaUnmapMemory(vma, allocation); }

void Allocator::flush(VmaAllocation allocation) { vk_check(vmaFlushAllocation(vma, allocation, 0, VK_WHOLE_SIZE)); }

PerFrameBuffer Allocator::create_per_frame_buffer(const VkBufferCreateInfo &per_frame_info, VmaMemoryUsage usage,
                                                  uint32_t num_frames)
{
    PerFrameBuffer buf;
    buf.num_frames = num_frames;
    VkDeviceSize alignment = 0;
    if (per_frame_info.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) {
        alignment = std::max(alignment, min_uniform_buffer_offset_alignment);
    }
    if (per_frame_info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        alignment = std::max(alignment, min_storage_buffer_offset_alignment);
    }
    if (per_frame_info.usage & VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT ||
        per_frame_info.usage & VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) {
        alignment = std::max(alignment, min_texel_buffer_offset_alignment);
    }
    buf.per_frame_size = (per_frame_info.size + alignment - 1) / alignment * alignment;

    VkBufferCreateInfo info = per_frame_info;
    info.size = buf.per_frame_size * buf.num_frames;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    vk_check(vmaCreateBuffer(vma, &info, &allocCI, &buf.buffer, &buf.allocation, nullptr));
    return buf;
}

void Allocator::flush(const PerFrameBuffer &buffer, uint32_t frameIndex)
{
    ASSERT(frameIndex < buffer.num_frames);
    VkDeviceSize offset = frameIndex * buffer.per_frame_size;
    vk_check(vmaFlushAllocation(vma, buffer.allocation, offset, buffer.per_frame_size));
}

void Allocator::begin_staging_session()
{
    ASSERT(!upload_cb);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = upload_cp;
    allocInfo.commandBufferCount = 1;

    vk_check(vkAllocateCommandBuffers(device, &allocInfo, &upload_cb));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(upload_cb, &beginInfo));
}

void Allocator::end_staging_session()
{
    ASSERT(upload_cb);
    vk_check(vkEndCommandBuffer(upload_cb));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &upload_cb;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vk_check(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    vk_check(vkQueueSubmit(upload_queue, 1, &submitInfo, fence));
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, upload_cp, 1, &upload_cb);

    upload_cb = VK_NULL_HANDLE;
    clear_staging_buffer();
}

VkDeviceSize Allocator::image_size(const VkImageCreateInfo &info) const
{
    size_t num_pixels = 0;
    for (uint32_t i = 0, w = info.extent.width, h = info.extent.height, d = info.extent.depth; i < info.mipLevels;
         ++i) {
        num_pixels += w * h * d;
        w = std::max(w / 2, 1u);
        h = std::max(h / 2, 1u);
        d = std::max(d / 2, 1u);
    }
    return num_pixels * vkuFormatElementSize(info.format) * info.arrayLayers;
}

#include <vulkan/utility/vk_format_utils.h>

Image Allocator::create_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, VmaAllocationCreateFlags flags)
{
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;
    Image image;
    vk_check(vmaCreateImage(vma, &info, &allocCI, &image.image, &image.allocation, nullptr));

    return image;
}

Image Allocator::create_and_transit_image(const VkImageCreateInfo &info, VmaMemoryUsage usage,
                                          VmaAllocationCreateFlags flags, VkImageLayout layout)
{
    Image image = create_image(info, usage, flags);

    ASSERT(upload_cb);

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = info.initialLayout;
    barrier.newLayout = layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = info.arrayLayers;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    return image;
}

Image Allocator::create_and_upload_image(const VkImageCreateInfo &info_, VmaMemoryUsage usage,
                                         VmaAllocationCreateFlags flags,
                                         const std::function<void(std::byte *)> &copy_fn, VkImageLayout layout,
                                         MipmapOption mipmap_option, bool cube_map)
{

    ASSERT(upload_cb);

    VkImageCreateInfo info = info_;
    info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (mipmap_option == MipmapOption::AutoGenerate) {
        info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    Image image = create_image(info, usage, flags);

    Buffer staging = create_staging_buffer(image_size(info), copy_fn);

    VkImageMemoryBarrier allToDst = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    allToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    allToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    allToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    allToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    allToDst.image = image.image;
    allToDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    allToDst.subresourceRange.baseMipLevel = 0;
    allToDst.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    allToDst.subresourceRange.baseArrayLayer = 0;
    allToDst.subresourceRange.layerCount = info.arrayLayers;
    allToDst.srcAccessMask = 0;
    allToDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &allToDst);

    switch (mipmap_option) {
    //////////////////////////////////////////////////////////////////////////
    case MipmapOption::OnlyAllocate: {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = info.arrayLayers;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = info.extent;
        region.bufferOffset = 0;
        vkCmdCopyBufferToImage(upload_cb, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        VkImageMemoryBarrier allToReady{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        allToReady.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allToReady.newLayout = layout;
        allToReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        allToReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        allToReady.image = image.image;
        allToReady.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        allToReady.subresourceRange.baseMipLevel = 0;
        allToReady.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        allToReady.subresourceRange.baseArrayLayer = 0;
        allToReady.subresourceRange.layerCount = info.arrayLayers;
        allToReady.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        allToReady.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &allToReady);
        break;
    }
    //////////////////////////////////////////////////////////////////////////
    case MipmapOption::PreGenerated: {
        std::vector<VkBufferImageCopy> regions(info.mipLevels);
        for (uint32_t i = 0, w = info.extent.width, h = info.extent.height, offset = 0; i < info.mipLevels; ++i) {
            regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[i].imageSubresource.mipLevel = i;
            regions[i].imageSubresource.baseArrayLayer = 0;
            regions[i].imageSubresource.layerCount = info.arrayLayers;
            regions[i].imageOffset = {0, 0, 0};
            regions[i].imageExtent = {w, h, 1};
            regions[i].bufferOffset = offset;
            offset += (w * h) * info.arrayLayers * vkuFormatElementSize(info.format);
            w = std::max(w / 2, 1u);
            h = std::max(h / 2, 1u);
        }
        vkCmdCopyBufferToImage(upload_cb, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               (uint32_t)regions.size(), regions.data());

        VkImageMemoryBarrier allToReady{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        allToReady.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        allToReady.newLayout = layout;
        allToReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        allToReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        allToReady.image = image.image;
        allToReady.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        allToReady.subresourceRange.baseMipLevel = 0;
        allToReady.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        allToReady.subresourceRange.baseArrayLayer = 0;
        allToReady.subresourceRange.layerCount = info.arrayLayers;
        allToReady.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        allToReady.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &allToReady);
        break;
    }
    //////////////////////////////////////////////////////////////////////////
    case MipmapOption::AutoGenerate: {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = info.arrayLayers;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = info.extent;
        region.bufferOffset = 0;
        vkCmdCopyBufferToImage(upload_cb, staging.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &region);

        int mipWidth = (int)info.extent.width;
        int mipHeight = (int)info.extent.height;

        for (uint32_t i = 1; i < info.mipLevels; ++i) {
            VkImageMemoryBarrier lastToSrc = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            lastToSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            lastToSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            lastToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            lastToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            lastToSrc.image = image.image;
            lastToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            lastToSrc.subresourceRange.baseMipLevel = i - 1;
            lastToSrc.subresourceRange.levelCount = 1;
            lastToSrc.subresourceRange.baseArrayLayer = 0;
            lastToSrc.subresourceRange.layerCount = info.arrayLayers;
            lastToSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            lastToSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &lastToSrc);

            VkImageBlit blit = {};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = info.arrayLayers;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = info.arrayLayers;

            vkCmdBlitImage(upload_cb, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

            VkImageMemoryBarrier lastToReady = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            lastToReady.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            lastToReady.newLayout = layout;
            lastToReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            lastToReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            lastToReady.image = image.image;
            lastToReady.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            lastToReady.subresourceRange.baseMipLevel = i - 1;
            lastToReady.subresourceRange.levelCount = 1;
            lastToReady.subresourceRange.baseArrayLayer = 0;
            lastToReady.subresourceRange.layerCount = info.arrayLayers;
            lastToReady.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            lastToReady.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

            vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &lastToReady);

            if (mipWidth > 1)
                mipWidth /= 2;
            if (mipHeight > 1)
                mipHeight /= 2;
        }

        VkImageMemoryBarrier topToReady = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        topToReady.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        topToReady.newLayout = layout;
        topToReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        topToReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        topToReady.image = image.image;
        topToReady.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        topToReady.subresourceRange.baseMipLevel = info.mipLevels - 1;
        topToReady.subresourceRange.levelCount = 1;
        topToReady.subresourceRange.baseArrayLayer = 0;
        topToReady.subresourceRange.layerCount = info.arrayLayers;
        topToReady.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        topToReady.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

        vkCmdPipelineBarrier(upload_cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &topToReady);
        break;
    }
    default:
        ASSERT(false);
    }

    return image;
}

ImageWithView Allocator::create_image_with_view(const VkImageCreateInfo &info, VmaMemoryUsage usage,
                                                VmaAllocationCreateFlags flags, VkImageViewCreateInfo view_info)
{
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;
    ImageWithView iv;
    vk_check(vmaCreateImage(vma, &info, &allocCI, &iv.image, &iv.allocation, nullptr));
    view_info.image = iv.image;
    vk_check(vkCreateImageView(device, &view_info, nullptr, &iv.view));

    return iv;
}

ImageWithView Allocator::create_image_with_view(const Image &image, const VkImageViewCreateInfo &view_info)
{
    ImageWithView iv;
    iv.image = image.image;
    iv.allocation = image.allocation;
    ASSERT(view_info.image == image.image);
    vk_check(vkCreateImageView(device, &view_info, nullptr, &iv.view));
    return iv;
}

// ImageWithView Allocator::create_color_buffer(uint32_t width, uint32_t height, VkFormat format, bool sample,
//                                              bool storage)
//{
//     VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
//     imageInfo.imageType = VK_IMAGE_TYPE_2D;
//     imageInfo.extent = {width, height, 1};
//     imageInfo.format = format;
//     imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//     imageInfo.mipLevels = 1;
//     imageInfo.arrayLayers = 1;
//     imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
//     imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//     imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
//     imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
//     if (sample) {
//         imageInfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
//     }
//     if (storage) {
//         imageInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
//     }
//
//     VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
//     viewInfo.format = format;
//     viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
//     viewInfo.components = VkComponentMapping{};
//     viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//     viewInfo.subresourceRange.baseMipLevel = 0;
//     viewInfo.subresourceRange.levelCount = 1;
//     viewInfo.subresourceRange.baseArrayLayer = 0;
//     viewInfo.subresourceRange.layerCount = 1;
//
//     return create_image_with_view(imageInfo, viewInfo, VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags(0));
// }
//
// ImageWithView Allocator::create_depth_buffer(uint32_t width, uint32_t height, bool sample, bool storage)
//{
//     VkImageCreateInfo depthInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
//     depthInfo.imageType = VK_IMAGE_TYPE_2D;
//     depthInfo.extent = {width, height, 1};
//     depthInfo.format = VK_FORMAT_D32_SFLOAT;
//     depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//     depthInfo.mipLevels = 1;
//     depthInfo.arrayLayers = 1;
//     depthInfo.samples = VK_SAMPLE_COUNT_1_BIT;
//     depthInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
//     depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
//     depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
//     if (sample) {
//         depthInfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
//     }
//     if (storage) {
//         depthInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
//     }
//
//     VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
//     depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
//     depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
//     depthViewInfo.components = VkComponentMapping{};
//     depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
//     depthViewInfo.subresourceRange.baseMipLevel = 0;
//     depthViewInfo.subresourceRange.levelCount = 1;
//     depthViewInfo.subresourceRange.baseArrayLayer = 0;
//     depthViewInfo.subresourceRange.layerCount = 1;
//
//     return create_image_with_view(depthInfo, depthViewInfo, VMA_MEMORY_USAGE_GPU_ONLY, VmaAllocationCreateFlags(0));
// }

Buffer Allocator::create_staging_buffer(VkDeviceSize buffer_size, const std::byte *data, VkDeviceSize data_size)
{
    ASSERT(buffer_size >= data_size);
    return create_staging_buffer(buffer_size, [&](std::byte *dest) { memcpy(dest, (const void *)data, data_size); });
}

Buffer Allocator::create_staging_buffer(VkDeviceSize buffer_size, const std::function<void(std::byte *)> &copy_fn)
{
    VkBufferCreateInfo bufferCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCI.size = buffer_size;
    bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    Buffer buffer;
    VmaAllocationInfo info;
    vk_check(vmaCreateBuffer(vma, &bufferCI, &allocCI, &buffer.buffer, &buffer.allocation, &info));

    //
    copy_fn(reinterpret_cast<std::byte *>(info.pMappedData));
    //

    vk_check(vmaFlushAllocation(vma, buffer.allocation, 0, VK_WHOLE_SIZE));

    staging_buffers.push_back(buffer);

    return buffer;
}

void Allocator::clear_staging_buffer()
{
    for (auto staging : staging_buffers) {
        vmaDestroyBuffer(vma, staging.buffer, staging.allocation);
    }
    staging_buffers.clear();
}

AccelKHR Allocator::create_accel(const VkAccelerationStructureCreateInfoKHR &accel_info_)
{
    AccelKHR resultAccel;
    VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = accel_info_.size,
                                   .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};

    VmaAllocationCreateInfo allocCI{.usage = VMA_MEMORY_USAGE_AUTO};

    vk_check(vmaCreateBuffer(vma, &buffer_info, &allocCI, &resultAccel.buffer, &resultAccel.allocation, nullptr));

    // Setting the buffer
    VkAccelerationStructureCreateInfoKHR accel = accel_info_;
    accel.buffer = resultAccel.buffer;
    // Create the acceleration structure
    vkCreateAccelerationStructureKHR(device, &accel, nullptr, &resultAccel.accel);

    if (vkGetAccelerationStructureDeviceAddressKHR != nullptr) {
        VkAccelerationStructureDeviceAddressInfoKHR info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        info.accelerationStructure = resultAccel.accel;
        resultAccel.address = vkGetAccelerationStructureDeviceAddressKHR(device, &info);
    }

    return resultAccel;
}

void Allocator::destroy(const Buffer &buffer) { vmaDestroyBuffer(vma, buffer.buffer, buffer.allocation); }

void Allocator::destroy(const TexelBuffer &texel_buffer)
{
    vkDestroyBufferView(device, texel_buffer.buffer_view, nullptr);
    vmaDestroyBuffer(vma, texel_buffer.buffer, texel_buffer.allocation);
}

void Allocator::destroy(const PerFrameBuffer &per_frame_buffer)
{
    vmaDestroyBuffer(vma, per_frame_buffer.buffer, per_frame_buffer.allocation);
}

void Allocator::destroy(const FrequentUploadBuffer &uniform_buffer)
{
    destroy(uniform_buffer.dest);
    destroy(uniform_buffer.staging);
}

void Allocator::destroy(const Image &image) { vmaDestroyImage(vma, image.image, image.allocation); }

void Allocator::destroy(const ImageWithView &image)
{
    vmaDestroyImage(vma, image.image, image.allocation);
    vkDestroyImageView(device, image.view, nullptr);
}

void Allocator::destroy(const AccelKHR &accel)
{
    vkDestroyAccelerationStructureKHR(device, accel.accel, nullptr);
    vmaDestroyBuffer(vma, accel.buffer, accel.allocation);
}

//-----------------------------------------------------------------------------
// [Basic vulkan object management]
//-----------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                        VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                                        void *user_data)
{
    if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        // Only allow debug printf info. Otherwise too noisy...
        if (std::strcmp(callback_data->pMessageIdName, "WARNING-DEBUG-PRINTF") != 0) {
            return VK_FALSE;
        }
        get_default_logger().info("vk_debug_callback: {}", callback_data->pMessage);
        return VK_FALSE;
    }
    get_default_logger().error("vk_debug_callback: {}", callback_data->pMessage);
    return VK_FALSE;
}

static void check_required_instance_extensions(const std::vector<const char *> &rexts)
{
    uint32_t availableExtCount = 0;
    vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &availableExtCount, nullptr));
    std::vector<VkExtensionProperties> exts(availableExtCount);
    vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &availableExtCount, exts.data()));

    for (const char *rext : rexts) {
        bool found = false;
        for (const auto &ext : exts) {
            if (strcmp(ext.extensionName, rext) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            get_default_logger().critical("Vulkan instance extension not available: [{}].", rext);
            std::abort();
        }
    }
}

static void check_required_instance_layers(const std::vector<const char *> &rlayers)
{
    uint32_t availableLayerCount;
    vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(availableLayerCount);
    vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());

    for (const char *rlayer : rlayers) {
        bool found = false;
        for (const auto &layer : availableLayers) {
            if (strcmp(layer.layerName, rlayer) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            get_default_logger().critical("Vulkan instance layer not available: [{}].", rlayer);
            std::abort();
        }
    }
}

void ContextArgs::enable_validation()
{
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");

    device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

#ifdef VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME
    // Don't forget to set environment variable NV_ALLOW_RAYTRACING_VALIDATION=1
    add_device_feature<VkPhysicalDeviceRayTracingValidationFeaturesNV>() =
        VkPhysicalDeviceRayTracingValidationFeaturesNV{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV,
            .rayTracingValidation = VK_TRUE};

    device_extensions.push_back(VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME);

    get_default_logger().info("Also enabled NV ray tracing validation.");
#endif

    validation = true;
}

void ContextArgs::enable_swapchain()
{
    if (!glfwInit()) {
        fprintf(stderr, "GLFW: Failed to initialize.\n");
        std::abort();
    }
    if (!glfwVulkanSupported()) {
        fprintf(stderr, "GLFW: vulkan not supported.\n");
        std::abort();
    }

    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    for (uint32_t i = 0; i < glfw_extension_count; ++i) {
        instance_extensions.push_back(glfw_extensions[i]);
    }
    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    swapchain = true;
}

Context::~Context()
{
    if (instance == VK_NULL_HANDLE) {
        return;
    }

    allocator = {};
    vkDestroyDevice(device, nullptr);

    if (validation) {
        auto vkDestroyDebugUtilsMessengerEXT =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}

void Context::create_instance(const ContextArgs &info)
{
    vk_check(volkInitialize());

    uint32_t instVersion = 0;
    vk_check(vkEnumerateInstanceVersion(&instVersion));
    uint32_t major = VK_VERSION_MAJOR(instVersion);
    uint32_t minor = VK_VERSION_MINOR(instVersion);
    uint32_t patch = VK_VERSION_PATCH(instVersion);
    get_default_logger().info("Vulkan instance version: {}.{}.{}.", major, minor, patch);

    std::vector<const char *> rexts;
    for (const auto &ext : info.instance_extensions) {
        rexts.push_back(ext.c_str());
    }
    check_required_instance_extensions(rexts);

    std::vector<const char *> rlayers;
    for (const auto &layer : info.instance_layers) {
        rlayers.push_back(layer.c_str());
    }
    check_required_instance_layers(rlayers);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_MAKE_VERSION(info.api_version_major, info.api_version_minor, 0);

    VkInstanceCreateInfo instanceCI{};
    instanceCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCI.pApplicationInfo = &appInfo;
    instanceCI.enabledExtensionCount = (uint32_t)rexts.size();
    instanceCI.ppEnabledExtensionNames = rexts.data();
    instanceCI.enabledLayerCount = (uint32_t)rlayers.size();
    instanceCI.ppEnabledLayerNames = rlayers.data();

    if (!info.validation) {
        vk_check(vkCreateInstance(&instanceCI, nullptr, &instance));
        validation = false;
    } else {
        VkDebugUtilsMessengerCreateInfoEXT debugMessagerCreateInfo{};
        debugMessagerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugMessagerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugMessagerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugMessagerCreateInfo.pfnUserCallback = vk_debug_callback;
        debugMessagerCreateInfo.pUserData = nullptr;

        VkValidationFeatureEnableEXT enables[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
        VkValidationFeaturesEXT features = {
            .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
            .enabledValidationFeatureCount = 1,
            .pEnabledValidationFeatures = enables,
        };

        VkLayerSettingsCreateInfoEXT layer_settings{
            .sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
        };
        features.pNext = &debugMessagerCreateInfo;
        instanceCI.pNext = &features;

        vk_check(vkCreateInstance(&instanceCI, nullptr, &instance));

        auto vkCreateDebugUtilsMessengerEXT =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        vk_check(vkCreateDebugUtilsMessengerEXT(instance, &debugMessagerCreateInfo, nullptr, &debug_messenger));
        validation = true;
    }

    volkLoadInstance(instance);
}

static bool has_required_device_extensions(VkPhysicalDevice physical_device, const std::vector<const char *> &rexts)
{
    uint32_t extCount;
    vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, nullptr));
    std::vector<VkExtensionProperties> availableExts(extCount);
    vk_check(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, availableExts.data()));

    for (const char *rext : rexts) {
        bool currFound = false;
        for (const auto &ext : availableExts) {
            if (strcmp(rext, ext.extensionName) == 0) {
                currFound = true;
                break;
            }
        }
        if (!currFound) {
            return false;
        }
    }
    return true;
}

// TODO: It's much more work to actually check all additional features (VkPhysicalDeviceFeatures2)...
static bool has_required_device_features(VkPhysicalDevice physical_device, const VkPhysicalDeviceFeatures &required)
{
    VkPhysicalDeviceFeatures available;
    vkGetPhysicalDeviceFeatures(physical_device, &available);

    constexpr uint32_t count = uint32_t(sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32));
    using FeatureArray = std::array<VkBool32, count>;

    FeatureArray requiredArr;
    memcpy(&requiredArr, &required, sizeof(FeatureArray));
    FeatureArray availableArr;
    memcpy(&availableArr, &available, sizeof(FeatureArray));
    for (uint32_t i = 0; i < count; ++i) {
        if (requiredArr[i] && !availableArr[i])
            return false;
    }
    return true;
}

static bool find_all_purpose_queue_family_index(VkPhysicalDevice physical_device, VkSurfaceKHR surface, uint32_t &index)
{
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProps;
    queueFamilyProps.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, queueFamilyProps.data());
    for (uint32_t q = 0; q < (uint32_t)queueFamilyProps.size(); ++q) {
        VkBool32 supported =
            queueFamilyProps[q].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
        if (surface != VK_NULL_HANDLE) {
            VkBool32 presentSupport = false;
            vk_check(vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, q, surface, &presentSupport));
            supported &= presentSupport;
        }
        if (supported) {
            index = q;
            return true;
        }
    }
    return false;
}

std::vector<CompatibleDevice> Context::query_compatible_devices(const ContextArgs &info, VkSurfaceKHR surface)
{
    std::vector<const char *> rexts;
    for (const std::string &rext : info.device_extensions) {
        rexts.push_back(rext.c_str());
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        get_default_logger().critical("Cannot find any vulkan physical device.");
        std::abort();
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    get_default_logger().info("Compatible devices:");
    std::vector<CompatibleDevice> compatibles;
    for (uint32_t i = 0; i < (uint32_t)devices.size(); ++i) {
        if (!has_required_device_extensions(devices[i], rexts)) {
            continue;
        }
        if (!has_required_device_features(devices[i], info.device_features.features)) {
            continue;
        }
        uint32_t queueFamilyIndex;
        if (!find_all_purpose_queue_family_index(devices[i], surface, queueFamilyIndex)) {
            continue;
        }

        VkPhysicalDeviceProperties prop;
        vkGetPhysicalDeviceProperties(devices[i], &prop);
        get_default_logger().info("GPU [{}]: {}.", i, prop.deviceName);
        compatibles.push_back({devices[i], i, queueFamilyIndex});
    }

    return compatibles;
}

void Context::create_device(const ContextArgs &info, CompatibleDevice compatible)
{
    get_default_logger().info("Selected GPU index: [{}].", compatible.physical_device_index);

    physical_device = compatible.physical_device;
    vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
    vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);
    main_queue_family_index = compatible.queue_family_index;

    std::vector<const char *> rexts;
    for (const std::string &rext : info.device_extensions) {
        rexts.push_back(rext.c_str());
    }

    VkDeviceQueueCreateInfo queueCI{};
    queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCI.queueFamilyIndex = main_queue_family_index;
    queueCI.queueCount = 1;
    constexpr float queuePriority = 1.0f;
    queueCI.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCI{};
    deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCI.pQueueCreateInfos = &queueCI;
    deviceCI.queueCreateInfoCount = 1;
    deviceCI.pEnabledFeatures = nullptr;
    deviceCI.pNext = &info.device_features;
    deviceCI.enabledExtensionCount = (uint32_t)rexts.size();
    deviceCI.ppEnabledExtensionNames = rexts.data();
    deviceCI.enabledLayerCount = 0;

    vk_check(vkCreateDevice(physical_device, &deviceCI, nullptr, &device));
    volkLoadDevice(device);

    vkGetDeviceQueue(device, main_queue_family_index, 0, &main_queue);

    VmaVulkanFunctions vmf_vk_fns{};
    vmf_vk_fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmf_vk_fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo vmaInfo{};
    // TODO: should set based on VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress
    vmaInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaInfo.pVulkanFunctions = &vmf_vk_fns;
    vmaInfo.physicalDevice = physical_device;
    vmaInfo.device = device;
    vmaInfo.instance = instance;
    allocator = std::make_shared<Allocator>(vmaInfo, main_queue_family_index, main_queue);
}

CommandPool::CommandPool(VkDevice device, uint32_t familyIndex, VkCommandPoolCreateFlags flags, VkQueue defaultQueue)
{
    assert(!m_device);
    m_device = device;
    VkCommandPoolCreateInfo info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.flags = flags;
    info.queueFamilyIndex = familyIndex;
    vkCreateCommandPool(m_device, &info, nullptr, &m_commandPool);
    if (defaultQueue) {
        m_queue = defaultQueue;
    } else {
        vkGetDeviceQueue(device, familyIndex, 0, &m_queue);
    }
}

CommandPool::~CommandPool()
{
    if (m_commandPool) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }
    m_device = VK_NULL_HANDLE;
}

VkCommandBuffer
CommandPool::create_command_buffer(VkCommandBufferLevel level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/, bool begin,
                                   VkCommandBufferUsageFlags flags /*= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT*/,
                                   const VkCommandBufferInheritanceInfo *pInheritanceInfo /*= nullptr*/)
{
    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.level = level;
    allocInfo.commandPool = m_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);

    if (begin) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = flags;
        beginInfo.pInheritanceInfo = pInheritanceInfo;

        vkBeginCommandBuffer(cmd, &beginInfo);
    }

    return cmd;
}

void CommandPool::destroy(size_t count, const VkCommandBuffer *cmds)
{
    vkFreeCommandBuffers(m_device, m_commandPool, (uint32_t)count, cmds);
}

void CommandPool::submit_and_wait(size_t count, const VkCommandBuffer *cmds, VkQueue queue)
{
    submit(count, cmds, queue);
    vk_check(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(m_device, m_commandPool, (uint32_t)count, cmds);
}

void CommandPool::submit(size_t count, const VkCommandBuffer *cmds, VkQueue queue, VkFence fence)
{
    for (size_t i = 0; i < count; i++) {
        vkEndCommandBuffer(cmds[i]);
    }

    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pCommandBuffers = cmds;
    submit.commandBufferCount = (uint32_t)count;
    vkQueueSubmit(queue, 1, &submit, fence);
}

void CommandPool::submit(size_t count, const VkCommandBuffer *cmds, VkFence fence)
{
    submit(count, cmds, m_queue, fence);
}

void CommandPool::submit(const std::vector<VkCommandBuffer> &cmds, VkFence fence)
{
    submit(cmds.size(), cmds.data(), m_queue, fence);
}

//-----------------------------------------------------------------------------
// [Convenience helper for setting up descriptor sets]
//-----------------------------------------------------------------------------

void DescriptorSetHelper::add_binding(std::string name, VkDescriptorSetLayoutBinding binding, bool unbounded_array)
{
    ASSERT(!last_unbounded_array, "There is already an unbounded array descriptor!");
    last_unbounded_array = unbounded_array;
    bindings.push_back(std::move(binding));
    auto res = name_map.insert({std::move(name), (uint32_t)bindings.size() - 1});
    ASSERT(res.second, "Duplicated binding name!");
}

VkDescriptorPool DescriptorSetHelper::create_pool(VkDevice device, uint32_t max_sets) const
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (const auto &b : bindings) {
        bool found = false;
        for (auto it = poolSizes.begin(); it != poolSizes.end(); ++it) {
            if (it->type == b.descriptorType) {
                it->descriptorCount += b.descriptorCount * max_sets;
                found = true;
                break;
            }
        }
        if (!found) {
            VkDescriptorPoolSize poolSize;
            poolSize.type = b.descriptorType;
            poolSize.descriptorCount = b.descriptorCount * max_sets;
            poolSizes.push_back(poolSize);
        }
    }

    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.poolSizeCount = (uint32_t)poolSizes.size();
    poolCI.pPoolSizes = poolSizes.data();
    poolCI.maxSets = max_sets;
    VkDescriptorPool pool;
    vk_check(vkCreateDescriptorPool(device, &poolCI, nullptr, &pool));
    return pool;
}

VkDescriptorSetLayout DescriptorSetHelper::create_set_layout(VkDevice device) const
{
    VkDescriptorSetLayoutCreateInfo setLayoutCI{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutCI.bindingCount = (uint32_t)bindings.size();
    setLayoutCI.pBindings = bindings.data();

    VkDescriptorSetLayout setLayout;
    if (last_unbounded_array) {
        std::vector<VkDescriptorBindingFlags> flags(bindings.size(), (VkFlags)0);
        flags.back() = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = (uint32_t)bindings.size(),
            .pBindingFlags = flags.data(),
        };
        setLayoutCI.pNext = &binding_flags;
        vkCreateDescriptorSetLayout(device, &setLayoutCI, nullptr, &setLayout);
    } else {
        vkCreateDescriptorSetLayout(device, &setLayoutCI, nullptr, &setLayout);
    }

    return setLayout;
}

VkWriteDescriptorSet DescriptorSetHelper::make_write(VkDescriptorSet dst_set, uint32_t dst_binding) const
{
    VkWriteDescriptorSet writeSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    for (size_t i = 0; i < bindings.size(); i++) {
        if (bindings[i].binding == dst_binding) {
            writeSet.descriptorCount = bindings[i].descriptorCount;
            writeSet.descriptorType = bindings[i].descriptorType;
            writeSet.dstBinding = dst_binding;
            writeSet.dstSet = dst_set;
            writeSet.dstArrayElement = 0;
            return writeSet;
        }
    }
    ASSERT(false, "binding not found");
    return writeSet;
}

VkWriteDescriptorSet DescriptorSetHelper::make_write(VkDescriptorSet dst_set, const std::string &binding_name) const
{
    auto it = name_map.find(binding_name);
    ASSERT(it != name_map.end(), "binding with name [%s] not found!", binding_name.c_str());
    auto &b = bindings[it->second];
    VkWriteDescriptorSet writeSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dst_set,
        .dstBinding = b.binding,
        .dstArrayElement = 0,
        .descriptorCount = b.descriptorCount,
        .descriptorType = b.descriptorType,
    };
    return writeSet;
}

VkWriteDescriptorSet DescriptorSetHelper::make_write_array(VkDescriptorSet dst_set, uint32_t dst_binding,
                                                           uint32_t start, uint32_t count) const
{
    VkWriteDescriptorSet writeSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    for (size_t i = 0; i < bindings.size(); i++) {
        if (bindings[i].binding == dst_binding) {
            ASSERT(start + count <= bindings[i].descriptorCount);
            writeSet.descriptorCount = count;
            writeSet.descriptorType = bindings[i].descriptorType;
            writeSet.dstBinding = dst_binding;
            writeSet.dstSet = dst_set;
            writeSet.dstArrayElement = start;
            return writeSet;
        }
    }
    ASSERT(false, "binding not found");
    return writeSet;
}

VkWriteDescriptorSet DescriptorSetHelper::make_write_array(VkDescriptorSet dst_set, const std::string &binding_name,
                                                           uint32_t start, uint32_t count) const
{
    auto it = name_map.find(binding_name);
    ASSERT(it != name_map.end(), "binding with name [%s] not found!", binding_name.c_str());
    auto &b = bindings[it->second];
    ASSERT(start + count <= b.descriptorCount);
    VkWriteDescriptorSet writeSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dst_set,
        .dstBinding = b.binding,
        .dstArrayElement = start,
        .descriptorCount = count,
        .descriptorType = b.descriptorType,
    };
    return writeSet;
}

void ParameterBlockMeta::init(VkDevice device, uint32_t max_sets, DescriptorSetHelper &&helper)
{
    if (is_init()) {
        return;
    }
    this->desc_set_helper = std::move(helper);
    this->device = device;
    this->max_sets = max_sets;

    last_unbounded_array = this->desc_set_helper.last_unbounded_array;

    desc_pool = desc_set_helper.create_pool(device, max_sets);
    desc_set_layout = desc_set_helper.create_set_layout(device);
    allocated_sets = 0;
}

void ParameterBlockMeta::deinit()
{
    if (!is_init()) {
        return;
    }
    vkDestroyDescriptorSetLayout(device, desc_set_layout, nullptr);
    vkDestroyDescriptorPool(device, desc_pool, nullptr);

    desc_set_helper = DescriptorSetHelper();
    desc_set_layout = VK_NULL_HANDLE;
    desc_pool = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;

    max_sets = 0;
    allocated_sets = 0;
    last_unbounded_array = false;
}

void ParameterBlockMeta::allocate_blocks(uint32_t num, std::span<ParameterBlock> out,
                                         std::optional<uint32_t> unbounded_array_max_size)
{
    ASSERT(allocated_sets + num <= max_sets, "Exceeds max sets!");
    allocated_sets += num;

    std::vector<VkDescriptorSetLayout> set_layouts(num, desc_set_layout);

    VkDescriptorSetVariableDescriptorCountAllocateInfo set_counts = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};

    std::vector<VkDescriptorSet> sets(num);
    if (last_unbounded_array) {
        ASSERT(unbounded_array_max_size);
        std::vector<uint32_t> counts(num, *unbounded_array_max_size);
        set_counts.descriptorSetCount = (uint32_t)counts.size();
        set_counts.pDescriptorCounts = counts.data();
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = &set_counts,
            .descriptorPool = desc_pool,
            .descriptorSetCount = num,
            .pSetLayouts = set_layouts.data(),
        };
        vk_check(vkAllocateDescriptorSets(device, &alloc_info, sets.data()));
    } else {
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = desc_pool,
            .descriptorSetCount = num,
            .pSetLayouts = set_layouts.data(),
        };
        vk_check(vkAllocateDescriptorSets(device, &alloc_info, sets.data()));
    }

    for (uint32_t i = 0; i < num; ++i) {
        out[i].desc_set = sets[i];
        out[i].meta = this;
    }
}

void ParameterBlock::write_buffers(const std::string &binding_name, std::vector<VkDescriptorBufferInfo> &&buffer_infos,
                                   uint32_t start, ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    if (buffer_infos.size() == 1) {
        write = meta->desc_set_helper.make_write(desc_set, binding_name);
    } else {
        write = meta->desc_set_helper.make_write_array(desc_set, binding_name, start, (uint32_t)buffer_infos.size());
    }

    write_array.buffer_infos.push_front(std::move(buffer_infos));
    write.pBufferInfo = write_array.buffer_infos.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_buffer(const std::string &binding_name, const VkDescriptorBufferInfo &buffer_info,
                                  ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    write = meta->desc_set_helper.make_write(desc_set, binding_name);
    write.pBufferInfo = &buffer_info;

    write_array.buffer_infos.push_front({buffer_info});
    write.pBufferInfo = write_array.buffer_infos.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_images(const std::string &binding_name, std::vector<VkDescriptorImageInfo> &&image_infos,
                                  uint32_t start, ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    if (image_infos.size() == 1) {
        write = meta->desc_set_helper.make_write(desc_set, binding_name);
    } else {
        write = meta->desc_set_helper.make_write_array(desc_set, binding_name, start, (uint32_t)image_infos.size());
    }

    write_array.image_infos.push_front(std::move(image_infos));
    write.pImageInfo = write_array.image_infos.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_image(const std::string &binding_name, const VkDescriptorImageInfo &image_info,
                                 ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    write = meta->desc_set_helper.make_write(desc_set, binding_name);

    write_array.image_infos.push_front({image_info});
    write.pImageInfo = write_array.image_infos.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_texel_buffers(const std::string &binding_name, std::vector<VkBufferView> &&buffer_views,
                                         uint32_t start, ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    if (buffer_views.size() == 1) {
        write = meta->desc_set_helper.make_write(desc_set, binding_name);
    } else {
        write = meta->desc_set_helper.make_write_array(desc_set, binding_name, start, (uint32_t)buffer_views.size());
    }

    write_array.texel_buffer_views.push_front(std::move(buffer_views));
    write.pTexelBufferView = write_array.texel_buffer_views.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_texel_buffer(const std::string &binding_name, const VkBufferView &buffer_view,
                                        ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    write = meta->desc_set_helper.make_write(desc_set, binding_name);

    write_array.texel_buffer_views.push_front({buffer_view});
    write.pTexelBufferView = write_array.texel_buffer_views.front().data();
    write_array.writes.push_back(write);
}

void ParameterBlock::write_accels(const std::string &binding_name,
                                  const VkWriteDescriptorSetAccelerationStructureKHR &accels, uint32_t start,
                                  ParameterWriteArray &write_array) const
{
    VkWriteDescriptorSet write;
    if (accels.accelerationStructureCount == 1) {
        write = meta->desc_set_helper.make_write(desc_set, binding_name);
    } else {
        write = meta->desc_set_helper.make_write_array(desc_set, binding_name, start,
                                                       (uint32_t)accels.accelerationStructureCount);
    }

    write_array.accels.push_front(accels);
    write.pNext = &write_array.accels.front();
    write_array.writes.push_back(write);
}

//-----------------------------------------------------------------------------
// [Ray tracing facilities (modified from nvvk)]
//-----------------------------------------------------------------------------

// Helper function to insert a memory barrier for acceleration structures
inline void accelerationStructureBarrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst)
{
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
}

void AccelerationStructureBuildData::add_geometry(const VkAccelerationStructureGeometryKHR &asGeom,
                                                  const VkAccelerationStructureBuildRangeInfoKHR &offset)
{
    geometry.push_back(asGeom);
    asBuildRangeInfo.push_back(offset);
}

void AccelerationStructureBuildData::add_geometry(const AccelerationStructureGeometryInfo &asGeom)
{
    geometry.push_back(asGeom.geometry);
    asBuildRangeInfo.push_back(asGeom.rangeInfo);
}

VkAccelerationStructureBuildSizesInfoKHR
AccelerationStructureBuildData::finalize_geometry(VkDevice device, VkBuildAccelerationStructureFlagsKHR flags)
{
    ASSERT(geometry.size() > 0 && "No geometry added to Build Structure");
    ASSERT(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");

    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = asType;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = static_cast<uint32_t>(geometry.size());
    buildInfo.pGeometries = geometry.data();
    buildInfo.ppGeometries = nullptr;
    buildInfo.scratchData.deviceAddress = 0;

    std::vector<uint32_t> maxPrimCount(asBuildRangeInfo.size());
    for (size_t i = 0; i < asBuildRangeInfo.size(); ++i) {
        maxPrimCount[i] = asBuildRangeInfo[i].primitiveCount;
    }

    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
                                            maxPrimCount.data(), &sizeInfo);

    return sizeInfo;
}

VkAccelerationStructureCreateInfoKHR AccelerationStructureBuildData::make_create_info() const
{
    ASSERT(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");
    ASSERT(sizeInfo.accelerationStructureSize > 0 && "Acceleration Structure Size not set");

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.type = asType;
    createInfo.size = sizeInfo.accelerationStructureSize;

    return createInfo;
}

AccelerationStructureGeometryInfo
AccelerationStructureBuildData::make_instanceGeometry(size_t numInstances, VkDeviceAddress instanceBufferAddr)
{
    ASSERT(asType == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && "Instance geometry can only be used with TLAS");

    // Describes instance data in the acceleration structure.
    VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    geometryInstances.data.deviceAddress = instanceBufferAddr;

    // Set up the geometry to use instance data.
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = geometryInstances;

    // Specifies the number of primitives (instances in this case).
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = static_cast<uint32_t>(numInstances);

    // Prepare and return geometry information.
    AccelerationStructureGeometryInfo result;
    result.geometry = geometry;
    result.rangeInfo = rangeInfo;

    return result;
}

void AccelerationStructureBuildData::cmd_build_acceleration_structure(VkCommandBuffer cmd,
                                                                      VkAccelerationStructureKHR accelerationStructure,
                                                                      VkDeviceAddress scratchAddress)
{
    ASSERT(geometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
    ASSERT(accelerationStructure != VK_NULL_HANDLE &&
           "Acceleration Structure not created, first call createAccelerationStructure");

    const VkAccelerationStructureBuildRangeInfoKHR *rangeInfo = asBuildRangeInfo.data();

    // Build the acceleration structure
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = accelerationStructure;
    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.pGeometries = geometry.data(); // In case the structure was copied, we need to update the pointer

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfo);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
    // is finished before starting the next one.
    accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

void AccelerationStructureBuildData::cmd_update_acceleration_structure(VkCommandBuffer cmd,
                                                                       VkAccelerationStructureKHR accelerationStructure,
                                                                       VkDeviceAddress scratchAddress)
{
    ASSERT(geometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
    ASSERT(accelerationStructure != VK_NULL_HANDLE &&
           "Acceleration Structure not created, first call createAccelerationStructure");

    const VkAccelerationStructureBuildRangeInfoKHR *rangeInfo = asBuildRangeInfo.data();

    // Build the acceleration structure
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = accelerationStructure;
    buildInfo.dstAccelerationStructure = accelerationStructure;
    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.pGeometries = geometry.data();
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfo);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
    // is finished before starting the next one.
    accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// Blas Builder : utility to create BLAS
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------

BlasBuilder::BlasBuilder(Allocator &allocator, VkDevice device) : m_device(device), m_alloc(&allocator) {}

BlasBuilder::~BlasBuilder()
{
    destroy_query_pool();
    destroy_non_compacted_blas();
}

void BlasBuilder::create_query_pool(uint32_t maxBlasCount)
{
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpci.queryCount = maxBlasCount;
    vkCreateQueryPool(m_device, &qpci, nullptr, &m_queryPool);
}

// This will build multiple BLAS serially, one after the other, ensuring that the process
// stays within the specified memory budget.
bool BlasBuilder::cmd_create_blas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                                  std::vector<AccelKHR> &blasAccel, VkDeviceAddress scratchAddress,
                                  VkDeviceSize hintMaxBudget)
{
    // It won't run in parallel, but will process all BLAS within the budget before returning
    return cmd_create_parallel_blas(cmd, blasBuildData, blasAccel, {scratchAddress}, hintMaxBudget);
}

// This function is responsible for building multiple Bottom-Level Acceleration Structures (BLAS) in parallel,
// ensuring that the process stays within the specified memory budget.
//
// Returns:
//   A boolean indicating whether all BLAS in the `blasBuildData` have been built by this function call.
//   Returns `true` if all BLAS were built, `false` otherwise.
bool BlasBuilder::cmd_create_parallel_blas(VkCommandBuffer cmd,
                                           std::vector<AccelerationStructureBuildData> &blasBuildData,
                                           std::vector<AccelKHR> &blasAccel,
                                           const std::vector<VkDeviceAddress> &scratchAddress,
                                           VkDeviceSize hintMaxBudget)
{
    // Initialize the query pool if necessary to handle queries for properties of built acceleration structures.
    initialize_query_pool_if_needed(blasBuildData);

    VkDeviceSize processBudget = 0;               // Tracks the total memory used in the construction process.
    uint32_t currentQueryIdx = m_currentQueryIdx; // Local copy of the current query index.

    // Process each BLAS in the data vector while staying under the memory budget.
    while (m_currentBlasIdx < blasBuildData.size() && processBudget < hintMaxBudget) {
        // Build acceleration structures and accumulate the total memory used.
        processBudget += build_acceleration_structures(cmd, blasBuildData, blasAccel, scratchAddress, hintMaxBudget,
                                                       processBudget, currentQueryIdx);
    }

    // Check if all BLAS have been built.
    return m_currentBlasIdx >= blasBuildData.size();
}

// Initializes a query pool for recording acceleration structure properties if necessary.
// This function ensures a query pool is available if any BLAS in the build data is flagged for compaction.
void BlasBuilder::initialize_query_pool_if_needed(const std::vector<AccelerationStructureBuildData> &blasBuildData)
{
    if (!m_queryPool) {
        // Iterate through each BLAS build data element to check if the compaction flag is set.
        for (const auto &blas : blasBuildData) {
            if (blas.has_compact_flag()) {
                create_query_pool(static_cast<uint32_t>(blasBuildData.size()));
                break;
            }
        }
    }

    // If a query pool is now available (either newly created or previously existing),
    // reset the query pool to clear any old data or states.
    if (m_queryPool) {
        vkResetQueryPool(m_device, m_queryPool, 0, static_cast<uint32_t>(blasBuildData.size()));
    }
}

// Builds multiple Bottom-Level Acceleration Structures (BLAS) for a Vulkan ray tracing pipeline.
// This function manages memory budgets and submits the necessary commands to the specified command buffer.
//
// Parameters:
//   cmd            - Command buffer where acceleration structure commands are recorded.
//   blasBuildData  - Vector of data structures containing the geometry and other build-related information for each
//   BLAS. blasAccel      - Vector where the function will store the created acceleration structures. scratchAddress -
//   Vector of device addresses pointing to scratch memory required for the build process. hintMaxBudget  - A hint for
//   the maximum budget allowed for building acceleration structures. currentBudget  - The current usage of the budget
//   prior to this call. currentQueryIdx - Reference to the current index for queries, updated during execution.
//
// Returns:
//   The total device size used for building the acceleration structures during this function call.
VkDeviceSize BlasBuilder::build_acceleration_structures(VkCommandBuffer cmd,
                                                        std::vector<AccelerationStructureBuildData> &blasBuildData,
                                                        std::vector<AccelKHR> &blasAccel,
                                                        const std::vector<VkDeviceAddress> &scratchAddress,
                                                        VkDeviceSize hintMaxBudget, VkDeviceSize currentBudget,
                                                        uint32_t &currentQueryIdx)
{
    // Temporary vectors for storing build-related data
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> collectedBuildInfo;
    std::vector<VkAccelerationStructureKHR> collectedAccel;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR *> collectedRangeInfo;

    // Pre-allocate memory based on the number of BLAS to be built
    collectedBuildInfo.reserve(blasBuildData.size());
    collectedAccel.reserve(blasBuildData.size());
    collectedRangeInfo.reserve(blasBuildData.size());

    // Initialize the total budget used in this function call
    VkDeviceSize budgetUsed = 0;

    // Loop through BLAS data while there is scratch address space and budget available
    while (collectedBuildInfo.size() < scratchAddress.size() && currentBudget + budgetUsed < hintMaxBudget &&
           m_currentBlasIdx < blasBuildData.size()) {
        auto &data = blasBuildData[m_currentBlasIdx];
        VkAccelerationStructureCreateInfoKHR createInfo = data.make_create_info();

        // Create and store acceleration structure
        blasAccel[m_currentBlasIdx] = m_alloc->create_accel(createInfo);
        collectedAccel.push_back(blasAccel[m_currentBlasIdx].accel);

        // Setup build information for the current BLAS
        data.buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        data.buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
        data.buildInfo.dstAccelerationStructure = blasAccel[m_currentBlasIdx].accel;
        data.buildInfo.scratchData.deviceAddress = scratchAddress[m_currentBlasIdx % scratchAddress.size()];
        data.buildInfo.pGeometries = data.geometry.data();
        collectedBuildInfo.push_back(data.buildInfo);
        collectedRangeInfo.push_back(data.asBuildRangeInfo.data());

        // Update the used budget with the size of the current structure
        budgetUsed += data.sizeInfo.accelerationStructureSize;
        m_currentBlasIdx++;
    }

    // Command to build the acceleration structures on the GPU
    vkCmdBuildAccelerationStructuresKHR(cmd, static_cast<uint32_t>(collectedBuildInfo.size()),
                                        collectedBuildInfo.data(), collectedRangeInfo.data());

    // Barrier to ensure proper synchronization after building
    accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    // If a query pool is available, record the properties of the built acceleration structures
    if (m_queryPool) {
        vkCmdWriteAccelerationStructuresPropertiesKHR(
            cmd, static_cast<uint32_t>(collectedAccel.size()), collectedAccel.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, m_queryPool, currentQueryIdx);
        currentQueryIdx += static_cast<uint32_t>(collectedAccel.size());
    }

    // Return the total budget used in this operation
    return budgetUsed;
}

// Compacts the Bottom-Level Acceleration Structures (BLAS) that have been built, reducing their memory footprint.
// This function uses the results from previously performed queries to determine the compacted sizes and then
// creates new, smaller acceleration structures. It also handles copying from the original to the compacted structures.
//
// Notes:
//   It assumes that a query has been performed earlier to determine the possible compacted sizes of the acceleration
//   structures.
//
void BlasBuilder::cmd_compact_blas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                                   std::vector<AccelKHR> &blasAccel)
{
    // Compute the number of queries that have been conducted between the current BLAS index and the query index.
    uint32_t queryCtn = m_currentBlasIdx - m_currentQueryIdx;
    // Ensure there is a valid query pool and BLAS to compact;
    if (m_queryPool == VK_NULL_HANDLE || queryCtn == 0) {
        return;
    }

    // Retrieve the compacted sizes from the query pool.
    std::vector<VkDeviceSize> compactSizes(queryCtn);
    vkGetQueryPoolResults(m_device, m_queryPool, m_currentQueryIdx, (uint32_t)compactSizes.size(),
                          compactSizes.size() * sizeof(VkDeviceSize), compactSizes.data(), sizeof(VkDeviceSize),
                          VK_QUERY_RESULT_WAIT_BIT);

    // Iterate through each BLAS index to process compaction.
    for (size_t i = m_currentQueryIdx; i < m_currentBlasIdx; i++) {
        size_t idx = i - m_currentQueryIdx; // Calculate local index for compactSizes vector.
        VkDeviceSize compactSize = compactSizes[idx];
        if (compactSize > 0) {
            // Update statistical tracking of sizes before and after compaction.
            m_stats.totalCompactSize += compactSize;
            m_stats.totalOriginalSize += blasBuildData[i].sizeInfo.accelerationStructureSize;
            blasBuildData[i].sizeInfo.accelerationStructureSize = compactSize;
            m_cleanupBlasAccel.push_back(blasAccel[i]); // Schedule old BLAS for cleanup.

            // Create a new acceleration structure for the compacted BLAS.
            VkAccelerationStructureCreateInfoKHR asCreateInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            asCreateInfo.size = compactSize;
            asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasAccel[i] = m_alloc->create_accel(asCreateInfo);

            // Command to copy the original BLAS to the newly created compacted version.
            VkCopyAccelerationStructureInfoKHR copyInfo{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
            copyInfo.src = blasBuildData[i].buildInfo.dstAccelerationStructure;
            copyInfo.dst = blasAccel[i].accel;
            copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
            vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);

            // Update the build data to reflect the new destination of the BLAS.
            blasBuildData[i].buildInfo.dstAccelerationStructure = blasAccel[i].accel;
        }
    }

    // Update the query index to the current BLAS index, marking the end of processing for these structures.
    m_currentQueryIdx = m_currentBlasIdx;
}

void BlasBuilder::destroy_non_compacted_blas()
{
    for (auto &blas : m_cleanupBlasAccel) {
        m_alloc->destroy(blas);
    }
    m_cleanupBlasAccel.clear();
}

void BlasBuilder::destroy_query_pool()
{
    if (m_queryPool) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
}

struct ScratchSizeInfo
{
    VkDeviceSize maxScratch;
    VkDeviceSize totalScratch;
};

ScratchSizeInfo calculateScratchAlignedSizes(const std::vector<AccelerationStructureBuildData> &buildData,
                                             uint32_t minAlignment)
{
    VkDeviceSize maxScratch{0};
    VkDeviceSize totalScratch{0};

    for (auto &buildInfo : buildData) {
        VkDeviceSize alignedSize = align_up(buildInfo.sizeInfo.buildScratchSize, minAlignment);
        // assert(alignedSize == buildInfo.sizeInfo.buildScratchSize);  // Make sure it was already aligned
        maxScratch = std::max(maxScratch, alignedSize);
        totalScratch += alignedSize;
    }

    return {maxScratch, totalScratch};
}

// Find if the total scratch size is within the budget, otherwise return n-time the max scratch size that fits in the
// budget
VkDeviceSize BlasBuilder::get_scratch_size(VkDeviceSize hintMaxBudget,
                                           const std::vector<AccelerationStructureBuildData> &buildData,
                                           uint32_t minAlignment /*= 128*/) const
{
    ScratchSizeInfo sizeInfo = calculateScratchAlignedSizes(buildData, minAlignment);
    VkDeviceSize maxScratch = sizeInfo.maxScratch;
    VkDeviceSize totalScratch = sizeInfo.totalScratch;

    if (totalScratch < hintMaxBudget) {
        return totalScratch;
    } else {
        uint64_t numScratch = std::max(uint64_t(1), hintMaxBudget / maxScratch);
        numScratch = std::min(numScratch, buildData.size());
        return numScratch * maxScratch;
    }
}

// Return the scratch addresses fitting the scrath strategy (see above)
void BlasBuilder::get_scratch_addresses(VkDeviceSize hintMaxBudget,
                                        const std::vector<AccelerationStructureBuildData> &buildData,
                                        VkDeviceAddress scratchBufferAddress,
                                        std::vector<VkDeviceAddress> &scratchAddresses, uint32_t minAlignment /*=128*/)
{
    ScratchSizeInfo sizeInfo = calculateScratchAlignedSizes(buildData, minAlignment);
    VkDeviceSize maxScratch = sizeInfo.maxScratch;
    VkDeviceSize totalScratch = sizeInfo.totalScratch;

    // Strategy 1: scratch was large enough for all BLAS, return the addresses in order
    if (totalScratch < hintMaxBudget) {
        VkDeviceAddress address = {};
        for (auto &buildInfo : buildData) {
            scratchAddresses.push_back(scratchBufferAddress + address);
            VkDeviceSize alignedSize = align_up(buildInfo.sizeInfo.buildScratchSize, minAlignment);
            address += alignedSize;
        }
    }
    // Strategy 2: there are n-times the max scratch fitting in the budget
    else {
        // Make sure there is at least one scratch buffer, and not more than the number of BLAS
        uint64_t numScratch = std::max(uint64_t(1), hintMaxBudget / maxScratch);
        numScratch = std::min(numScratch, buildData.size());

        VkDeviceAddress address = {};
        for (int i = 0; i < numScratch; i++) {
            scratchAddresses.push_back(scratchBufferAddress + address);
            address += maxScratch;
        }
    }
}

// Generates a formatted string summarizing the statistics of BLAS compaction results.
// The output includes the original and compacted sizes in megabytes (MB), the amount of memory saved,
// and the percentage reduction in size. This method is intended to provide a quick, human-readable
// summary of the compaction efficiency.
//
// Returns:
//   A string containing the formatted summary of the BLAS compaction statistics.
std::string BlasBuilder::Stats::to_string() const
{
    // Sizes in MB
    float originalSizeMB = totalOriginalSize / (1024.0f * 1024.0f);
    float compactSizeMB = totalCompactSize / (1024.0f * 1024.0f);
    float savedSizeMB = (totalOriginalSize - totalCompactSize) / (1024.0f * 1024.0f);

    float fractionSmaller = (totalOriginalSize == 0)
                                ? 0.0f
                                : (totalOriginalSize - totalCompactSize) / static_cast<float>(totalOriginalSize);

    std::string output = fmt::format("BLAS Compaction: {:.1f}MB -> {:.1f}MB ({:.1f}MB saved, {:.1f}% smaller)",
                                     originalSizeMB, compactSizeMB, savedSizeMB, fractionSmaller * 100.0f);

    return output;
}

// Returns the maximum scratch buffer size needed for building all provided acceleration structures.
// This function iterates through a vector of AccelerationStructureBuildData, comparing the scratch
// size required for each structure and returns the largest value found.
//
// Returns:
//   The maximum scratch size needed as a VkDeviceSize.
VkDeviceSize get_max_scratch_size(const std::vector<AccelerationStructureBuildData> &asBuildData)
{
    VkDeviceSize maxScratchSize = 0;
    for (const auto &blas : asBuildData) {
        maxScratchSize = std::max(maxScratchSize, blas.sizeInfo.buildScratchSize);
    }
    return maxScratchSize;
}

// Ray tracing BLAS and TLAS builder

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
RaytracingBuilderKHR::RaytracingBuilderKHR(const VkDevice &device, Allocator &allocator, uint32_t queueIndex)
{
    init(device, allocator, queueIndex);
}

void RaytracingBuilderKHR::init(const VkDevice &device, Allocator &allocator, uint32_t queueIndex)
{
    if (is_init()) {
        return;
    }
    m_device = device;
    m_queueIndex = queueIndex;
    m_debug.setup(device);
    m_alloc = &allocator;
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//

void RaytracingBuilderKHR::deinit()
{
    if (!is_init()) {
        return;
    }

    for (auto &b : m_blas) {
        m_alloc->destroy(b);
    }
    m_blas.clear();

    m_alloc->destroy(m_tlas);
    m_tlas = {};

    m_queueIndex = 0;
    m_debug = {};

    m_device = VK_NULL_HANDLE;
    m_alloc = nullptr;
}

RaytracingBuilderKHR::~RaytracingBuilderKHR() { deinit(); }

//--------------------------------------------------------------------------------------------------
// Returning the constructed top-level acceleration structure
//
VkAccelerationStructureKHR RaytracingBuilderKHR::get_tlas() const { return m_tlas.accel; }

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
VkDeviceAddress RaytracingBuilderKHR::get_blas_device_address(uint32_t blasId)
{
    ASSERT(size_t(blasId) < m_blas.size());
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = m_blas[blasId].accel;
    return vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);
}

//--------------------------------------------------------------------------------------------------
// Create all the BLAS from the vector of BlasInput
// - There will be one BLAS per input-vector entry
// - There will be as many BLAS as input.size()
// - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
//   and can be referenced by index.
// - if flag has the 'Compact' flag, the BLAS will be compacted
//
void RaytracingBuilderKHR::build_blas(const std::vector<BlasInput> &input, VkBuildAccelerationStructureFlagsKHR flags)
{
    auto numBlas = static_cast<uint32_t>(input.size());
    VkDeviceSize asTotalSize{0};    // Memory size of all allocated BLAS
    VkDeviceSize maxScratchSize{0}; // Largest scratch size

    std::vector<AccelerationStructureBuildData> blasBuildData(numBlas);
    m_blas.resize(numBlas); // Resize to hold all the BLAS
    for (uint32_t idx = 0; idx < numBlas; idx++) {
        blasBuildData[idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildData[idx].geometry = input[idx].geometry;
        blasBuildData[idx].asBuildRangeInfo = input[idx].range_info;

        auto sizeInfo = blasBuildData[idx].finalize_geometry(m_device, input[idx].flags | flags);
        maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }

    VkDeviceSize hintMaxBudget{256'000'000}; // 256 MB

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    Buffer blasScratchBuffer;

    bool hasCompaction = has_vk_flag(flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

    BlasBuilder blasBuilder(*m_alloc, m_device);

    uint32_t minAlignment = 128; /*m_rtASProperties.minAccelerationStructureScratchOffsetAlignment*/
    // 1) finding the largest scratch size
    VkDeviceSize scratchSize = blasBuilder.get_scratch_size(hintMaxBudget, blasBuildData, minAlignment);
    // 2) allocating the scratch buffer
    blasScratchBuffer = m_alloc->create_buffer_with_alignment(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = scratchSize,
                           .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        minAlignment);
    // 3) getting the device address for the scratch buffer
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.get_scratch_addresses(hintMaxBudget, blasBuildData, blasScratchBuffer.address, scratchAddresses,
                                      minAlignment);

    CommandPool m_cmdPool(m_device, m_queueIndex);

    bool finished = false;
    do {
        {
            VkCommandBuffer cmd = m_cmdPool.create_command_buffer();
            finished =
                blasBuilder.cmd_create_parallel_blas(cmd, blasBuildData, m_blas, scratchAddresses, hintMaxBudget);
            m_cmdPool.submit_and_wait(cmd);
        }
        if (hasCompaction) {
            VkCommandBuffer cmd = m_cmdPool.create_command_buffer();
            blasBuilder.cmd_compact_blas(cmd, blasBuildData, m_blas);
            m_cmdPool.submit_and_wait(cmd); // Submit command buffer and call vkQueueWaitIdle
            blasBuilder.destroy_non_compacted_blas();
        }
    } while (!finished);

    if (hasCompaction) {
        get_default_logger().info("{}", blasBuilder.get_statistics().to_string().c_str());
    }

    // Clean up
    // TODO: check this
    // m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(blasScratchBuffer);
}

void RaytracingBuilderKHR::build_tlas(
    const std::vector<VkAccelerationStructureInstanceKHR> &instances,
    VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR*/,
    bool update /*= false*/)
{
    build_tlas(instances, flags, update, false);
}

#ifdef VK_NV_ray_tracing_motion_blur
void RaytracingBuilderKHR::build_tlas(
    const std::vector<VkAccelerationStructureMotionInstanceNV> &instances,
    VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV*/,
    bool update /*= false*/)
{
    build_tlas(instances, flags, update, true);
}
#endif

//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
void RaytracingBuilderKHR::cmd_create_tlas(VkCommandBuffer cmdBuf, uint32_t countInstance,
                                           VkDeviceAddress instBufferAddr, Buffer &scratchBuffer,
                                           VkBuildAccelerationStructureFlagsKHR flags, bool update, bool motion)
{
    AccelerationStructureBuildData tlasBuildData;
    tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    AccelerationStructureGeometryInfo geo = tlasBuildData.make_instanceGeometry(countInstance, instBufferAddr);
    tlasBuildData.add_geometry(geo);

    auto sizeInfo = tlasBuildData.finalize_geometry(m_device, flags);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = update ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize;

    scratchBuffer = m_alloc->create_buffer(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = scratchSize,
                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT});
    VkDeviceAddress scratchAddress = scratchBuffer.address;
    NAME_VK(scratchBuffer.buffer);

    if (update) { // Update the acceleration structure
        tlasBuildData.geometry[0].geometry.instances.data.deviceAddress = instBufferAddr;
        tlasBuildData.cmd_update_acceleration_structure(cmdBuf, m_tlas.accel, scratchAddress);
    } else { // Create and build the acceleration structure
        VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.make_create_info();

#ifdef VK_NV_ray_tracing_motion_blur
        VkAccelerationStructureMotionInfoNV motionInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV};
        motionInfo.maxInstances = countInstance;

        if (motion) {
            createInfo.createFlags = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
            createInfo.pNext = &motionInfo;
        }
#endif
        m_tlas = m_alloc->create_accel(createInfo);
        NAME_VK(m_tlas.accel);
        NAME_VK(m_tlas.buffer);
        tlasBuildData.cmd_build_acceleration_structure(cmdBuf, m_tlas.accel, scratchAddress);
    }
}

//--------------------------------------------------------------------------------------------------
// Refit BLAS number blasIdx from updated buffer contents.
//
void RaytracingBuilderKHR::update_blas(uint32_t blasIdx, BlasInput &blas, VkBuildAccelerationStructureFlagsKHR flags)
{
    ASSERT(size_t(blasIdx) < m_blas.size());

    AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
    buildData.geometry = blas.geometry;
    buildData.asBuildRangeInfo = blas.range_info;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = buildData.finalize_geometry(m_device, flags);

    // Allocate the scratch buffer and setting the scratch info
    Buffer scratchBuffer = m_alloc->create_buffer(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = sizeInfo.updateScratchSize,
                           .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

    // Update the instance buffer on the device side and build the TLAS
    CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer cmdBuf = genCmdBuf.create_command_buffer();
    buildData.cmd_update_acceleration_structure(cmdBuf, m_blas[blasIdx].accel, scratchBuffer.address);
    genCmdBuf.submit_and_wait(cmdBuf);

    m_alloc->destroy(scratchBuffer);
}

//--------------------------------------------------------------------------------------------------
// Default setup
//
void SBTWrapper::init(VkDevice device, uint32_t familyIndex, Allocator *allocator,
                      const VkPhysicalDeviceRayTracingPipelinePropertiesKHR &rtProperties)
{
    if (is_init()) {
        return;
    }
    m_device = device;
    m_queueIndex = familyIndex;
    m_pAlloc = allocator;
    m_debug.setup(device);

    m_handleSize = rtProperties.shaderGroupHandleSize;           // Size of a program identifier
    m_handleAlignment = rtProperties.shaderGroupHandleAlignment; // Alignment in bytes for each SBT entry
    m_shaderGroupBaseAlignment = rtProperties.shaderGroupBaseAlignment;
}

//--------------------------------------------------------------------------------------------------
// Destroying the allocated buffers and clearing all vectors
//
void SBTWrapper::deinit()
{
    if (!is_init()) {
        return;
    }
    if (m_pAlloc) {
        for (auto &b : m_buffer)
            m_pAlloc->destroy(b);
    }

    for (auto &i : m_index)
        i = {};
}

//--------------------------------------------------------------------------------------------------
// Finding the handle index position of each group type in the pipeline creation info.
// If the pipeline was created like: raygen, miss, hit, miss, hit, hit
// The result will be: raygen[0], miss[1, 3], hit[2, 4, 5], callable[]
//
void SBTWrapper::add_indices(VkRayTracingPipelineCreateInfoKHR rayPipelineInfo,
                             const std::vector<VkRayTracingPipelineCreateInfoKHR> &libraries)
{
    for (auto &i : m_index)
        i = {};

    // Libraries contain stages referencing their internal groups. When those groups
    // are used in the final pipeline we need to offset them to ensure each group has
    // a unique index
    uint32_t groupOffset = 0;

    for (size_t i = 0; i < libraries.size() + 1; i++) {
        // When using libraries, their groups and stages are appended after the groups and
        // stages defined in the main VkRayTracingPipelineCreateInfoKHR
        const auto &info = (i == 0) ? rayPipelineInfo : libraries[i - 1];

        // Finding the handle position of each group, splitting by raygen, miss and hit group
        for (uint32_t g = 0; g < info.groupCount; g++) {
            if (info.pGroups[g].type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) {
                uint32_t genShader = info.pGroups[g].generalShader;
                assert(genShader < info.stageCount);
                if (info.pStages[genShader].stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
                    m_index[Raygen].push_back(g + groupOffset);
                } else if (info.pStages[genShader].stage == VK_SHADER_STAGE_MISS_BIT_KHR) {
                    m_index[Miss].push_back(g + groupOffset);
                } else if (info.pStages[genShader].stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR) {
                    m_index[Callable].push_back(g + groupOffset);
                }
            } else {
                m_index[Hit].push_back(g + groupOffset);
            }
        }

        groupOffset += info.groupCount;
    }
}

//--------------------------------------------------------------------------------------------------
// This function creates 4 buffers, for raygen, miss, hit and callable shader.
// Each buffer will have the handle + 'data (if any)', .. n-times they have entries in the pipeline.
//
void SBTWrapper::create(VkPipeline rtPipeline, VkRayTracingPipelineCreateInfoKHR rayPipelineInfo /*= {}*/,
                        const std::vector<VkRayTracingPipelineCreateInfoKHR> &librariesInfo /*= {}*/)
{
    for (auto &b : m_buffer)
        m_pAlloc->destroy(b);

    // Get the total number of groups and handle index position
    uint32_t totalGroupCount{0};
    std::vector<uint32_t> groupCountPerInput;
    // A pipeline is defined by at least its main VkRayTracingPipelineCreateInfoKHR, plus a number of external libraries
    groupCountPerInput.reserve(1 + librariesInfo.size());
    if (rayPipelineInfo.sType == VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR) {
        add_indices(rayPipelineInfo, librariesInfo);
        groupCountPerInput.push_back(rayPipelineInfo.groupCount);
        totalGroupCount += rayPipelineInfo.groupCount;
        for (const auto &lib : librariesInfo) {
            groupCountPerInput.push_back(lib.groupCount);
            totalGroupCount += lib.groupCount;
        }
    } else {
        // Find how many groups when added manually, by finding the largest index and adding 1
        // See also addIndex for manual entries
        for (auto &i : m_index) {
            if (!i.empty())
                totalGroupCount = std::max(totalGroupCount, *std::max_element(std::begin(i), std::end(i)));
        }
        totalGroupCount++;
        groupCountPerInput.push_back(totalGroupCount);
    }

    // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
    uint32_t sbtSize = totalGroupCount * m_handleSize;
    std::vector<uint8_t> shaderHandleStorage(sbtSize);

    vk_check(vkGetRayTracingShaderGroupHandlesKHR(m_device, rtPipeline, 0, totalGroupCount, sbtSize,
                                                  shaderHandleStorage.data()));
    // Find the max stride, minimum is the handle size + size of 'data (if any)' aligned to shaderGroupBaseAlignment
    auto findStride = [&](auto entry, auto &stride) {
        stride = align_up(m_handleSize, m_handleAlignment); // minimum stride
        for (auto &e : entry) {
            // Find the largest data + handle size, all aligned
            uint32_t dataHandleSize =
                align_up(static_cast<uint32_t>(m_handleSize + e.second.size() * sizeof(uint8_t)), m_handleAlignment);
            stride = std::max(stride, dataHandleSize);
        }
    };
    findStride(m_data[Raygen], m_stride[Raygen]);
    findStride(m_data[Miss], m_stride[Miss]);
    findStride(m_data[Hit], m_stride[Hit]);
    findStride(m_data[Callable], m_stride[Callable]);

    // Special case, all Raygen must start aligned on GroupBase
    m_stride[Raygen] = align_up(m_stride[Raygen], m_shaderGroupBaseAlignment);

    // Buffer holding the staging information
    std::array<std::vector<uint8_t>, 4> stage;
    stage[Raygen] = std::vector<uint8_t>(m_stride[Raygen] * index_count(Raygen));
    stage[Miss] = std::vector<uint8_t>(m_stride[Miss] * index_count(Miss));
    stage[Hit] = std::vector<uint8_t>(m_stride[Hit] * index_count(Hit));
    stage[Callable] = std::vector<uint8_t>(m_stride[Callable] * index_count(Callable));

    // Write the handles in the SBT buffer + data info (if any)
    auto copyHandles = [&](std::vector<uint8_t> &buffer, std::vector<uint32_t> &indices, uint32_t stride, auto &data) {
        auto *pBuffer = buffer.data();
        for (uint32_t index = 0; index < static_cast<uint32_t>(indices.size()); index++) {
            auto *pStart = pBuffer;
            // Copy the handle
            memcpy(pBuffer, shaderHandleStorage.data() + (indices[index] * m_handleSize), m_handleSize);
            // If there is data for this group index, copy it too
            auto it = data.find(index);
            if (it != std::end(data)) {
                pBuffer += m_handleSize;
                memcpy(pBuffer, it->second.data(), it->second.size() * sizeof(uint8_t));
            }
            pBuffer = pStart + stride; // Jumping to next group
        }
    };

    // Copy the handles/data to each staging buffer
    copyHandles(stage[Raygen], m_index[Raygen], m_stride[Raygen], m_data[Raygen]);
    copyHandles(stage[Miss], m_index[Miss], m_stride[Miss], m_data[Miss]);
    copyHandles(stage[Hit], m_index[Hit], m_stride[Hit], m_data[Hit]);
    copyHandles(stage[Callable], m_index[Callable], m_stride[Callable], m_data[Callable]);

    // Creating device local buffers where handles will be stored
    VkBufferUsageFlags usage_flags =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
    VkMemoryPropertyFlags mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // CommandPool genCmdBuf(m_device, m_queueIndex);
    // VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

    m_pAlloc->stage_session([&](Allocator &alloc) {
        for (uint32_t i = 0; i < 4; i++) {
            if (!stage[i].empty()) {
                m_buffer[i] = alloc.create_buffer(
                    VkBufferCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        .usage = usage_flags,
                    },
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VmaAllocationCreateFlags(0), stage[i]);
                NAME_IDX_VK(m_buffer[i].buffer, i);
            }
        }
    });

    // genCmdBuf.submitAndWait(cmdBuf);
    // m_pAlloc->finalizeAndReleaseStaging();
}

VkDeviceAddress SBTWrapper::get_address(GroupType t) const
{
    if (m_buffer[t].buffer == VK_NULL_HANDLE)
        return 0;
    VkBufferDeviceAddressInfo i{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_buffer[t].buffer};
    return vkGetBufferDeviceAddress(
        m_device, &i); // Aligned on VkMemoryRequirements::alignment which includes shaderGroupBaseAlignment
}

const VkStridedDeviceAddressRegionKHR SBTWrapper::get_region(GroupType t, uint32_t indexOffset) const
{
    return VkStridedDeviceAddressRegionKHR{get_address(t) + indexOffset * get_stride(t), get_stride(t), get_size(t)};
}

const std::array<VkStridedDeviceAddressRegionKHR, 4> SBTWrapper::get_regions(uint32_t rayGenIndexOffset) const
{
    std::array<VkStridedDeviceAddressRegionKHR, 4> regions{get_region(Raygen, rayGenIndexOffset), get_region(Miss),
                                                           get_region(Hit), get_region(Callable)};
    return regions;
}

} // namespace vk

static void imgui_vk_debug_callback(VkResult result)
{
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Vulkan error in imgui: %d", (int)result);
    }
}

void GfxApp::glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void GfxApp::glfw_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    //
    reinterpret_cast<GfxApp *>(glfwGetWindowUserPointer(window))->key_callback(key, scancode, action, mods);
}

void GfxApp::glfw_scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    //
    reinterpret_cast<GfxApp *>(glfwGetWindowUserPointer(window))->scroll_callback(xoffset, yoffset);
}

void GfxApp::glfw_cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    //
    reinterpret_cast<GfxApp *>(glfwGetWindowUserPointer(window))->cursor_position_callback(xpos, ypos);
}
void GfxApp::glfw_mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    //
    reinterpret_cast<GfxApp *>(glfwGetWindowUserPointer(window))->mouse_button_callback(button, action, mods);
}

GfxApp::GfxApp(const GfxAppArgs &args)
    : app_name(args.app_name), gpu(args.gpu_context), swapchain_image_count(args.swapchain_image_count),
      max_frames_in_flight(args.max_frames_in_flight), swapchain_image_usage(args.swapchain_image_usage),
      swapchain_image_format(args.swapchain_image_format)
{
    ASSERT(swapchain_image_count == 2 || swapchain_image_count == 3);
    ASSERT(max_frames_in_flight <= swapchain_image_count);

    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_DECORATED , GLFW_FALSE);
    window = glfwCreateWindow(args.window_width, args.window_height, args.app_name.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, glfw_key_callback);
    glfwSetScrollCallback(window, glfw_scroll_callback);
    glfwSetCursorPosCallback(window, glfw_cursor_position_callback);
    glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);

    vk::vk_check(glfwCreateWindowSurface(gpu->vk.instance, window, nullptr, &surface));

    // 1. swapchain
    present_complete_semaphores.resize(max_frames_in_flight);
    render_complete_semaphores.resize(max_frames_in_flight);
    in_flight_fences.resize(max_frames_in_flight);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkDevice device = gpu->vk.device;
    for (uint32_t i = 0; i < max_frames_in_flight; i++) {
        vk::vk_check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &present_complete_semaphores[i]));
        vk::vk_check(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &render_complete_semaphores[i]));
        vk::vk_check(vkCreateFence(device, &fenceInfo, nullptr, &in_flight_fences[i]));
    }

    create_swapchain_and_images(args.window_width, args.window_height);

    // 2. cmd frames
    cmd_frames.resize(max_frames_in_flight);
    for (auto &frame : cmd_frames) {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = gpu->vk.main_queue_family_index;
        poolInfo.flags = 0; // Optional
        vk::vk_check(vkCreateCommandPool(device, &poolInfo, nullptr, &frame.pool));
    }

    // 3. imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.FontGlobalScale = 1.5f; // TODO: Auto-scaling?

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsClassic();
    // ImGui::GetStyle().ScaleAllSizes(2.0f);

    ImGui_ImplVulkan_LoadFunctions(
        [](const char *function_name, void *user_data) {
            VkInstance instance = (VkInstance)user_data;
            return vkGetInstanceProcAddr(instance, function_name);
        },
        gpu->vk.instance);
    ImGui_ImplGlfw_InitForVulkan(window, true);

    std::array<VkDescriptorPoolSize, 11> poolSizes = {{{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                                       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                                       {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                                       {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}}};

    VkDescriptorPoolCreateInfo poolCI{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                      .maxSets = 1,
                                      .poolSizeCount = (uint32_t)poolSizes.size(),
                                      .pPoolSizes = poolSizes.data()};

    vk::vk_check(vkCreateDescriptorPool(gpu->vk.device, &poolCI, nullptr, &imgui_desc_pool));

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = gpu->vk.instance;
    initInfo.PhysicalDevice = gpu->vk.physical_device;
    initInfo.Device = gpu->vk.device;
    initInfo.QueueFamily = gpu->vk.main_queue_family_index;
    initInfo.Queue = gpu->vk.main_queue;
    initInfo.PipelineCache = VK_NULL_HANDLE;
    initInfo.DescriptorPool = imgui_desc_pool;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = initInfo.ImageCount = swapchain_image_count;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.ColorAttachmentFormat = swapchain_image_format;
    initInfo.CheckVkResultFn = imgui_vk_debug_callback;

    ImGui_ImplVulkan_Init(&initInfo, VK_NULL_HANDLE);

    ImFontConfig font_config;
    font_config.RasterizerDensity = 2.0f;
    io.Fonts->AddFontFromFileTTF((fs::path(DATA_DIR) / "NotoSans-Regular.ttf").string().c_str(), 16.0f, &font_config);
}

// https://docs.vulkan.org/samples/latest/samples/performance/hpp_swapchain_images/README.html
void GfxApp::create_swapchain_and_images(uint32_t width, uint32_t height)
{
    VkPhysicalDevice physical_device = gpu->vk.physical_device;
    VkDevice device = gpu->vk.device;

    std::vector<VkSurfaceFormatKHR> all_formats;
    uint32_t format_count = 0;
    vk::vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr));
    ASSERT_FATAL(format_count > 0);
    all_formats.resize(format_count);
    vk::vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, all_formats.data()));

    VkSurfaceFormatKHR surface_format;
    surface_format.format = swapchain_image_format;
    surface_format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool format_supported = false;
    for (const auto &fmt : all_formats) {
        if (fmt.format == surface_format.format && fmt.colorSpace == surface_format.colorSpace) {
            format_supported = true;
            break;
        }
    }
    ASSERT_FATAL(format_supported, "Requested surface format not compatible!");

    std::vector<VkPresentModeKHR> all_present_modes;
    uint32_t present_mode_count = 0;
    vk::vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count, nullptr));
    ASSERT_FATAL(present_mode_count > 0);
    all_present_modes.resize(present_mode_count);
    vk::vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_mode_count,
                                                           all_present_modes.data()));

    // We use mailbox for triple buffering, fifo for double buffering.
    VkPresentModeKHR present_mode =
        (swapchain_image_count == 3) ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(all_present_modes.begin(), all_present_modes.end(), present_mode) == all_present_modes.end()) {
        present_mode = VK_PRESENT_MODE_FIFO_KHR;
        fprintf(stderr, "Swapchain present mode Mailbox not available. Falls back to FIFO!\n");
        // FIFO is guaranteed to be available by vulkan.
        ASSERT_FATAL(std::find(all_present_modes.begin(), all_present_modes.end(), present_mode) !=
                     all_present_modes.end());
    }

    VkSurfaceCapabilitiesKHR capabilities;
    vk::vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    swapchain_extent = capabilities.currentExtent;
    if (swapchain_extent.width == UINT32_MAX) {
        swapchain_extent = {width, height};
        swapchain_extent.width =
            std::clamp(swapchain_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        swapchain_extent.height =
            std::clamp(swapchain_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    if (swapchain_image_count < capabilities.minImageCount ||
        (capabilities.maxImageCount > 0 && swapchain_image_count > capabilities.maxImageCount)) {
        fprintf(stderr, "Requested swapchain image count (%u) is not supported!\n", swapchain_image_count);
        std::abort();
    }

    VkSwapchainCreateInfoKHR swapchain_create_info{};
    swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_create_info.surface = surface;
    swapchain_create_info.minImageCount = swapchain_image_count;
    swapchain_create_info.imageFormat = surface_format.format;
    swapchain_create_info.imageColorSpace = surface_format.colorSpace;
    swapchain_create_info.imageExtent = swapchain_extent;
    swapchain_create_info.imageArrayLayers = 1;
    // We assume the last pass is imgui so or'ed with color attachment
    swapchain_create_info.imageUsage =
        swapchain_image_usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // We only use one queue now.
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.preTransform = capabilities.currentTransform;
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_create_info.presentMode = present_mode;
    swapchain_create_info.clipped = VK_TRUE;
    swapchain_create_info.oldSwapchain = VK_NULL_HANDLE;
    vk::vk_check(vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain));

    vk::vk_check(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr));
    swapchain_images.resize(swapchain_image_count);
    vk::vk_check(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data()));

    swapchain_image_views.resize(swapchain_images.size());
    for (uint32_t i = 0; i < (uint32_t)swapchain_images.size(); i++) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = swapchain_images[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = swapchain_image_format;
        view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        vk::vk_check(vkCreateImageView(device, &view_info, nullptr, &swapchain_image_views[i]));
    }
}

void GfxApp::destroy_swapchain_and_images()
{
    VkDevice device = gpu->vk.device;
    swapchain_images.clear();
    for (size_t i = 0; i < swapchain_image_views.size(); i++) {
        vkDestroyImageView(device, swapchain_image_views[i], nullptr);
    }
    swapchain_image_views.clear();
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

GfxApp::~GfxApp()
{
    // 3. imgui
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(gpu->vk.device, imgui_desc_pool, nullptr);

    // 2. cmd frames
    VkDevice device = gpu->vk.device;
    for (auto &frame : cmd_frames) {
        vkFreeCommandBuffers(device, frame.pool, (uint32_t)frame.cbs.size(), frame.cbs.data());
        vkDestroyCommandPool(device, frame.pool, nullptr);
    }

    // 1. swapchain
    destroy_swapchain_and_images();

    for (uint32_t i = 0; i < max_frames_in_flight; i++) {
        vkDestroySemaphore(device, render_complete_semaphores[i], nullptr);
        vkDestroySemaphore(device, present_complete_semaphores[i], nullptr);
        vkDestroyFence(device, in_flight_fences[i], nullptr);
    }

    vkDestroySurfaceKHR(gpu->vk.instance, surface, nullptr);

    glfwDestroyWindow(window);
}

void GfxApp::run()
{
    while (!glfwWindowShouldClose(window)) {
        process_frame();
    }

    vk::vk_check(vkDeviceWaitIdle(gpu->vk.device));
}

void GfxApp::process_frame()
{
    update_base();

    if (!acquire_swapchain()) {
        resize();
        return;
    }

    encode_cmds_base();

    if (!submit_cmds_and_present()) {
        resize();
    }
}

float GfxApp::delta_time() const { return std::chrono::duration<float>(curr_time - prev_time).count(); }

void GfxApp::update_base()
{
    prev_time = curr_time;
    curr_time = std::chrono::steady_clock::now();
    if (prev_time == time_point()) {
        prev_time = curr_time;
    }

    float delta_time_ms = std::chrono::duration<float, std::milli>(curr_time - prev_time).count();

    // Show simple stats.
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    std::array<char, 512> title;
#ifndef NDEBUG
    constexpr char build[] = "Debug";
#else
    constexpr char build[] = "Release";
#endif
    sprintf(title.data(), "%s (%s) [%d x %d] [%.3f ms / %.1f fps]", app_name.c_str(), build, width, height,
            delta_time_ms, 1000.0f / delta_time_ms);
    glfwSetWindowTitle(window, title.data());

    early_update();

    glfwPollEvents();

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    update_imgui();

    ImGui::End();

    update();
}

bool GfxApp::acquire_swapchain()
{
    VkDevice device = gpu->vk.device;
    vk::vk_check(vkWaitForFences(device, 1, &in_flight_fences[curr_frame_in_flight_index], VK_TRUE, UINT64_MAX));

    VkResult ret =
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, present_complete_semaphores[curr_frame_in_flight_index],
                              VK_NULL_HANDLE, &curr_swapchain_image_index);
    if (ret == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;
    }
    ASSERT_FATAL(ret == VK_SUCCESS || ret == VK_SUBOPTIMAL_KHR);
    return true;
}

void GfxApp::encode_cmds_base()
{
    vk::vk_check(vkResetCommandPool(gpu->vk.device, cmd_frames[curr_frame_in_flight_index].pool,
                                    VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
    cmd_frames[curr_frame_in_flight_index].next_cb = 0;

    encode_cmds();

    {
        VkCommandBuffer cb = acquire_cb();
        vk::CmdBufRecorder recorder(cb, VkCommandBufferBeginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                                                 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT});
        encode_imgui(cb);
    }
}

std::span<VkCommandBuffer> GfxApp::acquire_cbs(uint32_t count)
{
    auto &frame = cmd_frames[curr_frame_in_flight_index];

    uint32_t num_available = (uint32_t)frame.cbs.size() - frame.next_cb;
    if (num_available < count) {
        uint32_t old_count = (uint32_t)frame.cbs.size();
        frame.cbs.resize(old_count + count - num_available);
        VkCommandBufferAllocateInfo alloc_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                  .commandPool = frame.pool,
                                                  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                  .commandBufferCount = count};
        vk::vk_check(vkAllocateCommandBuffers(gpu->vk.device, &alloc_info, &frame.cbs[old_count]));
    }
    std::span<VkCommandBuffer> ret(frame.cbs.begin() + frame.next_cb, frame.cbs.begin() + frame.next_cb + count);
    frame.next_cb += count;
    return ret;
}

VkCommandBuffer GfxApp::acquire_cb() { return acquire_cbs(1)[0]; }

bool GfxApp::submit_cmds_and_present()
{
    VkDevice device = gpu->vk.device;
    VkQueue queue = gpu->vk.main_queue;

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = (uint32_t)cmd_frames[curr_frame_in_flight_index].cbs.size();
    submit_info.pCommandBuffers = cmd_frames[curr_frame_in_flight_index].cbs.data();

    std::array<VkSemaphore, 1> submitWaitSemaphores = {present_complete_semaphores[curr_frame_in_flight_index]};
    std::array<VkPipelineStageFlags, 1> submitWaitStages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = (uint32_t)submitWaitSemaphores.size();
    submit_info.pWaitSemaphores = submitWaitSemaphores.data();
    submit_info.pWaitDstStageMask = submitWaitStages.data();
    std::array<VkSemaphore, 1> submitSignalSemaphores = {render_complete_semaphores[curr_frame_in_flight_index]};
    submit_info.signalSemaphoreCount = (uint32_t)submitSignalSemaphores.size();
    submit_info.pSignalSemaphores = submitSignalSemaphores.data();

    vk::vk_check(vkResetFences(device, 1, &in_flight_fences[curr_frame_in_flight_index]));
    vk::vk_check(vkQueueSubmit(queue, 1, &submit_info, in_flight_fences[curr_frame_in_flight_index]));

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    std::array<VkSemaphore, 1> presentWaitSemaphores = {render_complete_semaphores[curr_frame_in_flight_index]};
    present_info.waitSemaphoreCount = (uint32_t)presentWaitSemaphores.size();
    present_info.pWaitSemaphores = presentWaitSemaphores.data();
    std::array<VkSwapchainKHR, 1> swapchains = {swapchain};
    present_info.swapchainCount = (uint32_t)swapchains.size();
    present_info.pSwapchains = swapchains.data();
    present_info.pImageIndices = &curr_swapchain_image_index;
    present_info.pResults = nullptr; // Optional

    VkResult ret = vkQueuePresentKHR(queue, &present_info);
    if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR) {
        return false;
    } else {
        vk::vk_check(ret);
    }

    curr_frame_in_flight_index = (curr_frame_in_flight_index + 1) % max_frames_in_flight;

    return true;
}

void GfxApp::resize()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vk::vk_check(vkDeviceWaitIdle(gpu->vk.device));
    uint32_t old_swapchain_image_count = (uint32_t)swapchain_images.size();

    destroy_swapchain_and_images();
    create_swapchain_and_images(width, height);

    ASSERT_FATAL(old_swapchain_image_count == swapchain_images.size(), "New swapchain has different number of images!");

    ImGui_ImplVulkan_SetMinImageCount((uint32_t)swapchain_images.size());
}

void GfxApp::encode_imgui(VkCommandBuffer cb)
{
    ImGui::Render();
    ImDrawData *drawData = ImGui::GetDrawData();
    bool is_minimized = (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f);
    if (is_minimized) {
        return;
    }

    // Dynamic rendering requires us to handle transitions ourselves.
    // If the app has already output to swapchain previously then it should handle the layout transition somewhere else.
    // Otherwise we transition it from undefined to color attachemnt and clear it.
    if (!has_output_to_swapchain) {
        VkImageMemoryBarrier swapchain_to_color_att{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                    .srcAccessMask = VK_ACCESS_NONE,
                                                    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                    .image = swapchain_images[curr_swapchain_image_index],
                                                    .subresourceRange =
                                                        VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                                .baseMipLevel = 0,
                                                                                .levelCount = VK_REMAINING_MIP_LEVELS,
                                                                                .baseArrayLayer = 0,
                                                                                .layerCount = 1}};
        vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             (VkDependencyFlags)(0), {}, {}, {&swapchain_to_color_att, 1});
    }
    VkRenderingAttachmentInfoKHR color_attachment_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
    };
    color_attachment_info.imageView = swapchain_image_views[curr_swapchain_image_index];
    color_attachment_info.clearValue.color.float32[0] = 0.0f;
    color_attachment_info.clearValue.color.float32[1] = 0.0f;
    color_attachment_info.clearValue.color.float32[2] = 0.0f;
    color_attachment_info.clearValue.color.float32[3] = 1.0f;
    color_attachment_info.loadOp = has_output_to_swapchain ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment_info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkRenderingInfo rendering_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = VkRect2D{VkOffset2D{}, swapchain_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_info,
    };

    vkCmdBeginRendering(cb, &rendering_info);
    ImGui_ImplVulkan_RenderDrawData(drawData, cb);
    vkCmdEndRendering(cb);

    VkImageMemoryBarrier swapchain_to_present{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                              .srcAccessMask = VK_ACCESS_NONE,
                                              .dstAccessMask = VK_ACCESS_NONE,
                                              .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                              .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                              .image = swapchain_images[curr_swapchain_image_index],
                                              .subresourceRange =
                                                  VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                                          .baseMipLevel = 0,
                                                                          .levelCount = VK_REMAINING_MIP_LEVELS,
                                                                          .baseArrayLayer = 0,
                                                                          .layerCount = 1}};
    vk::pipeline_barrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, (VkDependencyFlags)(0), {}, {},
                         {&swapchain_to_present, 1});
}

} // namespace ks