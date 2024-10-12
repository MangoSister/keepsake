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

void DebugUtil::setObjectName(const uint64_t object, const std::string &name, VkObjectType t) const
{
    if (s_enabled) {
        VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, t, object,
                                        name.c_str()};
        s_vkSetDebugUtilsObjectNameEXT(m_device, &s);
    }
}

void DebugUtil::beginLabel(VkCommandBuffer cmdBuf, const std::string &label)
{
    if (s_enabled) {
        VkDebugUtilsLabelEXT s{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        s_vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &s);
    }
}

void DebugUtil::endLabel(VkCommandBuffer cmdBuf)
{
    if (s_enabled) {
        s_vkCmdEndDebugUtilsLabelEXT(cmdBuf);
    }
}

void DebugUtil::insertLabel(VkCommandBuffer cmdBuf, const std::string &label)
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

Buffer Allocator::create_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage, VmaAllocationCreateFlags flags,
                                const std::byte *data)
{
    Buffer buf;
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;
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
    uint32_t numPixels = 0;
    for (uint32_t i = 0, w = info.extent.width, h = info.extent.height; i < info.mipLevels; ++i) {
        numPixels += w * h;
        w = std::max(w / 2, 1u);
        h = std::max(h / 2, 1u);
    }
    return numPixels * vkuFormatElementSize(info.format) * info.arrayLayers;
}

#include <vulkan/utility/vk_format_utils.h>

static VkImageViewCreateInfo viewInfoFromImageInfo(const VkImageCreateInfo &image_info, bool cubeMap)
{
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext = nullptr;
    viewInfo.format = image_info.format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    if (image_info.imageType == VK_IMAGE_TYPE_2D) {
        if (image_info.arrayLayers == 6 && cubeMap) {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        } else if (image_info.arrayLayers > 1) {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        } else {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        }
    } else if (image_info.imageType == VK_IMAGE_TYPE_1D) {
        if (image_info.arrayLayers > 1) {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        } else {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
        }
    } else {
        ASSERT_FATAL("TODO");
    }
    return viewInfo;
}

Image Allocator::create_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, VmaAllocationCreateFlags flags)
{
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    allocCI.flags = flags;
    Image image;
    vk_check(vmaCreateImage(vma, &info, &allocCI, &image.image, &image.allocation, nullptr));

    return image;
}

ImageWithView Allocator::create_image_with_view(const VkImageCreateInfo &info, VkImageViewCreateInfo &view_info,
                                                VmaMemoryUsage usage)
{
    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = usage;
    ImageWithView image;
    vk_check(vmaCreateImage(vma, &info, &allocCI, &image.image, &image.allocation, nullptr));
    view_info.image = image.image;
    vk_check(vkCreateImageView(device, &view_info, nullptr, &image.view));

    return image;
}

ImageWithView Allocator::create_image_with_view(const VkImageCreateInfo &info, VmaMemoryUsage usage, bool cubeMap)
{
    VkImageViewCreateInfo viewInfo = viewInfoFromImageInfo(info, cubeMap);
    return create_image_with_view(info, viewInfo, usage);
}

ImageWithView Allocator::create_color_buffer(uint32_t width, uint32_t height, VkFormat format, bool sample,
                                             bool storage)
{
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.format = format;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (sample) {
        imageInfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (storage) {
        imageInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.format = format;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.components = VkComponentMapping{};
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    return create_image_with_view(imageInfo, viewInfo, VMA_MEMORY_USAGE_GPU_ONLY);
}

ImageWithView Allocator::create_depth_buffer(uint32_t width, uint32_t height, bool sample, bool storage)
{
    VkImageCreateInfo depthInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    depthInfo.imageType = VK_IMAGE_TYPE_2D;
    depthInfo.extent = {width, height, 1};
    depthInfo.format = VK_FORMAT_D32_SFLOAT;
    depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthInfo.mipLevels = 1;
    depthInfo.arrayLayers = 1;
    depthInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (sample) {
        depthInfo.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (storage) {
        depthInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.components = VkComponentMapping{};
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.subresourceRange.baseMipLevel = 0;
    depthViewInfo.subresourceRange.levelCount = 1;
    depthViewInfo.subresourceRange.baseArrayLayer = 0;
    depthViewInfo.subresourceRange.layerCount = 1;

    return create_image_with_view(depthInfo, depthViewInfo, VMA_MEMORY_USAGE_GPU_ONLY);
}

ImageWithView Allocator::create_and_transit_image(const VkImageCreateInfo &info, VkImageViewCreateInfo &view_info,
                                                  VmaMemoryUsage usage, VkImageLayout layout)
{
    ImageWithView image = create_image_with_view(info, view_info, usage);

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

ImageWithView Allocator::create_and_transit_image(const VkImageCreateInfo &info, VmaMemoryUsage usage,
                                                  VkImageLayout layout, bool cube_map)
{
    VkImageViewCreateInfo viewInfo = viewInfoFromImageInfo(info, cube_map);
    return create_and_transit_image(info, viewInfo, usage, layout);
}

ImageWithView Allocator::create_and_upload_image(const VkImageCreateInfo &info, VmaMemoryUsage usage,
                                                 const std::byte *data, size_t byte_size, VkImageLayout layout,
                                                 MipmapOption mipmap_option, bool cube_map)
{
    ASSERT(upload_cb);

    VkImageViewCreateInfo viewInfo = viewInfoFromImageInfo(info, cube_map);
    ImageWithView image = create_image_with_view(info, viewInfo, usage);

    Buffer staging = create_staging_buffer(image_size(info), data, byte_size);

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

Texture Allocator::create_texture(const ImageWithView &image, const VkSamplerCreateInfo &sampler_info, bool own_image)
{
    Texture texture;
    texture.image = image;
    vk_check(vkCreateSampler(device, &sampler_info, nullptr, &texture.sampler));
    texture.own_image = own_image;

    return texture;
}

Buffer Allocator::create_staging_buffer(VkDeviceSize buffer_size, const std::byte *data, VkDeviceSize data_size)
{
    ASSERT(buffer_size >= data_size);

    VkBufferCreateInfo bufferCI{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferCI.size = buffer_size;
    bufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    Buffer buffer;
    VmaAllocationInfo info;
    vk_check(vmaCreateBuffer(vma, &bufferCI, &allocCI, &buffer.buffer, &buffer.allocation, &info));

    memcpy(info.pMappedData, (const void *)data, data_size);
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

void Allocator::destroy(const Image &image) { vmaDestroyImage(vma, image.image, image.allocation); }

void Allocator::destroy(const ImageWithView &image)
{
    vmaDestroyImage(vma, image.image, image.allocation);
    vkDestroyImageView(device, image.view, nullptr);
}

void Allocator::destroy(const Texture &texture)
{
    vkDestroySampler(device, texture.sampler, nullptr);
    if (texture.own_image && texture.image.image != VK_NULL_HANDLE) {
        vkDestroyImageView(device, texture.image.view, nullptr);
        vmaDestroyImage(vma, texture.image.image, texture.image.allocation);
    }
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
            fprintf(stderr, "Vulkan instance extension not available: [%s].", rext);
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
            fprintf(stderr, "Vulkan instance layer not available: [%s].", rlayer);
            std::abort();
        }
    }
}

ContextCreateInfo::~ContextCreateInfo()
{
    for (void *data : device_features_data) {
        free(data);
    }
}

void ContextCreateInfo::enable_validation()
{
    instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instance_layers.push_back("VK_LAYER_KHRONOS_validation");
    validation = true;
}

void ContextCreateInfo::enable_swapchain() { device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME); }

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

void Context::create_instance(const ContextCreateInfo &info)
{
    vk_check(volkInitialize());

    uint32_t instVersion = 0;
    vk_check(vkEnumerateInstanceVersion(&instVersion));
    uint32_t major = VK_VERSION_MAJOR(instVersion);
    uint32_t minor = VK_VERSION_MINOR(instVersion);
    uint32_t patch = VK_VERSION_PATCH(instVersion);
    printf("Vulkan instance version: %u.%u.%u.\n", major, minor, patch);

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
        debugMessagerCreateInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugMessagerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugMessagerCreateInfo.pfnUserCallback = vk_debug_callback;
        debugMessagerCreateInfo.pUserData = nullptr;

        instanceCI.pNext = &debugMessagerCreateInfo;
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

std::vector<CompatibleDevice> Context::query_compatible_devices(const ContextCreateInfo &info, VkSurfaceKHR surface)
{
    std::vector<const char *> rexts;
    for (const std::string &rext : info.device_extensions) {
        rexts.push_back(rext.c_str());
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        fprintf(stderr, "Cannot find any vulkan physical device.\n");
        std::abort();
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    printf("Compatible devices:\n");
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
        printf("GPU [%u]: %s.\n", i, prop.deviceName);
        compatibles.push_back({devices[i], i, queueFamilyIndex});
    }

    return compatibles;
}

void Context::create_device(const ContextCreateInfo &info, CompatibleDevice compatible)
{
    printf("Selected GPU index: [%u].\n", compatible.physical_device_index);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        fprintf(stderr, "Cannot find any vulkan physical device.\n");
        std::abort();
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

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
CommandPool::createCommandBuffer(VkCommandBufferLevel level /*= VK_COMMAND_BUFFER_LEVEL_PRIMARY*/, bool begin,
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

void CommandPool::submitAndWait(size_t count, const VkCommandBuffer *cmds, VkQueue queue)
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
// [Swap chain]
//-----------------------------------------------------------------------------

Swapchain::Swapchain(const SwapchainCreateInfo &info)
    : physical_device(info.physical_device), device(info.device), queue(info.queue), surface(info.surface),
      max_frames_ahead(info.max_frames_ahead), render_ahead_index(0)
{
    present_complete_semaphores.resize(max_frames_ahead);
    render_complete_semaphores.resize(max_frames_ahead);
    inflight_fences.resize(max_frames_ahead);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (uint32_t i = 0; i < max_frames_ahead; i++) {
        vk_check(vkCreateSemaphore(info.device, &semaphoreInfo, nullptr, &present_complete_semaphores[i]));
        vk_check(vkCreateSemaphore(info.device, &semaphoreInfo, nullptr, &render_complete_semaphores[i]));
        vk_check(vkCreateFence(info.device, &fenceInfo, nullptr, &inflight_fences[i]));
    }

    create_swapchain_and_images(info.width, info.height);
}

Swapchain::~Swapchain()
{
    if (physical_device == VK_NULL_HANDLE) {
        return;
    }
    destroy_swapchain_and_images();

    for (uint32_t i = 0; i < max_frames_ahead; i++) {
        vkDestroySemaphore(device, render_complete_semaphores[i], nullptr);
        vkDestroySemaphore(device, present_complete_semaphores[i], nullptr);
        vkDestroyFence(device, inflight_fences[i], nullptr);
    }
}

void Swapchain::create_swapchain_and_images(uint32_t width, uint32_t height)
{
    VkSurfaceCapabilitiesKHR capabilities;
    vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));

    std::vector<VkSurfaceFormatKHR> allFormats;
    uint32_t formatCount = 0;
    vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr));
    ASSERT_FATAL(formatCount > 0);
    allFormats.resize(formatCount);
    vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, allFormats.data()));

    std::vector<VkPresentModeKHR> allPresentModes;
    uint32_t presentModeCount = 0;
    vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentModeCount, nullptr));
    ASSERT_FATAL(presentModeCount > 0);
    allPresentModes.resize(presentModeCount);
    vk_check(
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &presentModeCount, allPresentModes.data()));

    VkSurfaceFormatKHR surfaceFormat;
    surfaceFormat.format = VK_FORMAT_B8G8R8A8_UNORM;
    surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool fmtSupported = false;
    for (const auto &fmt : allFormats) {
        if (fmt.format == surfaceFormat.format && fmt.colorSpace == surfaceFormat.colorSpace) {
            fmtSupported = true;
            break;
        }
    }
    ASSERT_FATAL(fmtSupported, "Requested surface format not compatible!");

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    if (std::find(allPresentModes.begin(), allPresentModes.end(), presentMode) == allPresentModes.end()) {
        presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ASSERT_FATAL(std::find(allPresentModes.begin(), allPresentModes.end(), presentMode) != allPresentModes.end());
    }

    extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent = {width, height};
        extent.width =
            std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, extent.width));
        extent.height =
            std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, extent.height));
    }

    // Simply sticking to this minimum means that we may sometimes have to wait on the driver to complete internal
    // operations before we can acquire another image to render to. Therefore it is recommended to request at least one
    // more image than the minimum:
    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = surface;

    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageFormat = surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainCreateInfo.imageExtent = extent;
    swapchainCreateInfo.imageArrayLayers = 1;

    // TODO: Need to change this based on what is rendered before.
    swapchainCreateInfo.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // We only use one queue now.

    swapchainCreateInfo.preTransform = capabilities.currentTransform;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = presentMode;
    swapchainCreateInfo.clipped = VK_TRUE;

    swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

    vk_check(vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain));

    format = surfaceFormat.format;
    extent = extent;
    vk_check(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr));
    images.resize(imageCount);
    vk_check(vkGetSwapchainImagesKHR(device, swapchain, &imageCount, images.data()));
    ASSERT_FATAL(max_frames_ahead <= imageCount);

    image_views.resize(images.size());
    for (uint32_t i = 0; i < (uint32_t)images.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = images[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device, &createInfo, nullptr, &image_views[i]));
    }
}

void Swapchain::destroy_swapchain_and_images()
{
    images.clear();
    for (size_t i = 0; i < image_views.size(); i++) {
        vkDestroyImageView(device, image_views[i], nullptr);
    }
    image_views.clear();
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

bool Swapchain::acquire()
{
    vk_check(vkWaitForFences(device, 1, &inflight_fences[render_ahead_index], VK_TRUE, UINT64_MAX));

    VkResult ret = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, present_complete_semaphores[render_ahead_index],
                                         VK_NULL_HANDLE, &frame_index);
    if (ret == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;
    }
    ASSERT_FATAL(ret == VK_SUCCESS || ret == VK_SUBOPTIMAL_KHR);
    return true;
}

bool Swapchain::submit_and_present(const std::vector<VkCommandBuffer> &cbs)
{
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submitInfo.commandBufferCount = (uint32_t)cbs.size();
    submitInfo.pCommandBuffers = cbs.data();

    std::array<VkSemaphore, 1> submitWaitSemaphores = {present_complete_semaphores[render_ahead_index]};
    std::array<VkPipelineStageFlags, 1> submitWaitStages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = (uint32_t)submitWaitSemaphores.size();
    submitInfo.pWaitSemaphores = submitWaitSemaphores.data();
    submitInfo.pWaitDstStageMask = submitWaitStages.data();
    std::array<VkSemaphore, 1> submitSignalSemaphores = {render_complete_semaphores[render_ahead_index]};
    submitInfo.signalSemaphoreCount = (uint32_t)submitSignalSemaphores.size();
    submitInfo.pSignalSemaphores = submitSignalSemaphores.data();

    vk_check(vkResetFences(device, 1, &inflight_fences[render_ahead_index]));
    vk_check(vkQueueSubmit(queue, 1, &submitInfo, inflight_fences[render_ahead_index]));

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    std::array<VkSemaphore, 1> presentWaitSemaphores = {render_complete_semaphores[render_ahead_index]};
    presentInfo.waitSemaphoreCount = (uint32_t)presentWaitSemaphores.size();
    presentInfo.pWaitSemaphores = presentWaitSemaphores.data();
    std::array<VkSwapchainKHR, 1> swapchains = {swapchain};
    presentInfo.swapchainCount = (uint32_t)swapchains.size();
    presentInfo.pSwapchains = swapchains.data();
    presentInfo.pImageIndices = &frame_index;
    presentInfo.pResults = nullptr; // Optional

    VkResult ret = vkQueuePresentKHR(queue, &presentInfo);
    if (ret == VK_ERROR_OUT_OF_DATE_KHR || ret == VK_SUBOPTIMAL_KHR) {
        return false;
    } else {
        vk_check(ret);
    }

    render_ahead_index = (render_ahead_index + 1) % max_frames_ahead;

    return true;
}

void Swapchain::resize(uint32_t width, uint32_t height)
{
    vk_check(vkDeviceWaitIdle(device));
    uint32_t frameCount = (uint32_t)images.size();

    destroy_swapchain_and_images();
    create_swapchain_and_images(width, height);

    ASSERT_FATAL(frameCount == images.size(), "New swapchain has different number of images!");
}

//-----------------------------------------------------------------------------
// [Command buffer management]
//-----------------------------------------------------------------------------

CmdBufManager::CmdBufManager(uint32_t frame_count, uint32_t queue_family_index, VkDevice device) : device(device)
{
    frames.resize(frame_count);
    for (auto &frame : frames) {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queue_family_index;
        poolInfo.flags = 0; // Optional
        vk_check(vkCreateCommandPool(device, &poolInfo, nullptr, &frame.pool));
    }
}

CmdBufManager::~CmdBufManager()
{
    for (auto &frame : frames) {
        vkFreeCommandBuffers(device, frame.pool, (uint32_t)frame.cbs.size(), frame.cbs.data());
        vkDestroyCommandPool(device, frame.pool, nullptr);
    }
}

void CmdBufManager::start_frame(uint32_t frame_index)
{
    this->frame_index = frame_index;
    vk_check(vkResetCommandPool(device, frames[frame_index].pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
    frames[frame_index].next_cb = 0;
}

std::vector<VkCommandBuffer> CmdBufManager::acquire_cbs(uint32_t count)
{
    auto &frame = frames[frame_index];

    uint32_t numAvailable = (uint32_t)frame.cbs.size() - frame.next_cb;
    if (numAvailable < count) {
        uint32_t oldCount = (uint32_t)frame.cbs.size();
        frame.cbs.resize(oldCount + count - numAvailable);
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = frame.pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = count;
        vk_check(vkAllocateCommandBuffers(device, &allocInfo, &frame.cbs[oldCount]));
    }
    std::vector<VkCommandBuffer> ret;
    ret.insert(ret.end(), frame.cbs.begin() + frame.next_cb, frame.cbs.begin() + frame.next_cb + count);
    frame.next_cb += count;
    return ret;
}

void encode_cmd_now(VkDevice device, uint32_t queue_family_index, VkQueue queue,
                    const std::function<void(VkCommandBuffer)> &func)
{
    VkCommandPoolCreateInfo cmdPoolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdPoolCI.queueFamilyIndex = queue_family_index;
    cmdPoolCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool cmdPool;
    vk_check(vkCreateCommandPool(device, &cmdPoolCI, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = cmdPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vk_check(vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmdBuf, &beginInfo));

    func(cmdBuf);

    vk_check(vkEndCommandBuffer(cmdBuf));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vk_check(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    vk_check(vkQueueSubmit(queue, 1, &submitInfo, fence));
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuf);

    vkDestroyCommandPool(device, cmdPool, nullptr);
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

    if (last_unbounded_array) {
        std::vector<VkDescriptorBindingFlags> flags(bindings.size(), (VkFlags)0);
        flags.back() = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = 3,
            .pBindingFlags = flags.data(),
        };
        setLayoutCI.pNext = &binding_flags;
    }
    VkDescriptorSetLayout setLayout;
    vkCreateDescriptorSetLayout(device, &setLayoutCI, nullptr, &setLayout);
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
}

void ParameterBlockMeta::allocate_blocks(uint32_t num, std::span<ParameterBlock> out)
{
    ASSERT(allocated_sets + num <= max_sets, "Exceeds max sets!");
    allocated_sets += num;

    std::vector<VkDescriptorSetLayout> set_layouts(num, desc_set_layout);

    VkDescriptorSetAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = desc_pool,
                                          .descriptorSetCount = num,
                                          .pSetLayouts = set_layouts.data()};
    std::vector<VkDescriptorSet> sets(num);
    vk_check(vkAllocateDescriptorSets(device, &allocInfo, sets.data()));
    for (uint32_t i = 0; i < num; ++i) {
        out[i].desc_set = sets[i];
        out[i].meta = this;
    }
}

ParameterWrite ParameterBlock::write(const std::string &binding_name,
                                     const std::optional<VkDescriptorBufferInfo> buffer_info,
                                     const std::optional<VkDescriptorImageInfo> image_info,
                                     VkBufferView texel_buffer_view) const
{
    ParameterWrite write;
    write.write = meta->desc_set_helper.make_write(desc_set, binding_name);

    if (buffer_info) {
        write.buffer_info = std::make_unique<VkDescriptorBufferInfo>(*buffer_info);
        write.write.pBufferInfo = write.buffer_info.get();
    }
    if (image_info) {
        write.image_info = std::make_unique<VkDescriptorImageInfo>(*image_info);
        write.write.pImageInfo = write.image_info.get();
    }
    if (texel_buffer_view != VK_NULL_HANDLE) {
        write.texel_buffer_view = std::make_unique<VkBufferView>(texel_buffer_view);
        write.write.pTexelBufferView = write.texel_buffer_view.get();
    }
    return write;
}

//-----------------------------------------------------------------------------
// [Ray tracing facilities (modified from nvvk)]
//-----------------------------------------------------------------------------

void AccelerationStructureBuildData::addGeometry(const VkAccelerationStructureGeometryKHR &asGeom,
                                                 const VkAccelerationStructureBuildRangeInfoKHR &offset)
{
    asGeometry.push_back(asGeom);
    asBuildRangeInfo.push_back(offset);
}

void AccelerationStructureBuildData::addGeometry(const AccelerationStructureGeometryInfo &asGeom)
{
    asGeometry.push_back(asGeom.geometry);
    asBuildRangeInfo.push_back(asGeom.rangeInfo);
}

VkAccelerationStructureBuildSizesInfoKHR
AccelerationStructureBuildData::finalizeGeometry(VkDevice device, VkBuildAccelerationStructureFlagsKHR flags)
{
    ASSERT(asGeometry.size() > 0 && "No geometry added to Build Structure");
    ASSERT(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");

    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = asType;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = static_cast<uint32_t>(asGeometry.size());
    buildInfo.pGeometries = asGeometry.data();
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

VkAccelerationStructureCreateInfoKHR AccelerationStructureBuildData::makeCreateInfo() const
{
    ASSERT(asType != VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR && "Acceleration Structure Type not set");
    ASSERT(sizeInfo.accelerationStructureSize > 0 && "Acceleration Structure Size not set");

    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.type = asType;
    createInfo.size = sizeInfo.accelerationStructureSize;

    return createInfo;
}

AccelerationStructureGeometryInfo
AccelerationStructureBuildData::makeInstanceGeometry(size_t numInstances, VkDeviceAddress instanceBufferAddr)
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

void AccelerationStructureBuildData::cmdBuildAccelerationStructure(VkCommandBuffer cmd,
                                                                   VkAccelerationStructureKHR accelerationStructure,
                                                                   VkDeviceAddress scratchAddress)
{
    ASSERT(asGeometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
    ASSERT(accelerationStructure != VK_NULL_HANDLE &&
           "Acceleration Structure not created, first call createAccelerationStructure");

    const VkAccelerationStructureBuildRangeInfoKHR *rangeInfo = asBuildRangeInfo.data();

    // Build the acceleration structure
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = accelerationStructure;
    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.pGeometries = asGeometry.data(); // In case the structure was copied, we need to update the pointer

    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &rangeInfo);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
    // is finished before starting the next one.
    accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

void AccelerationStructureBuildData::cmdUpdateAccelerationStructure(VkCommandBuffer cmd,
                                                                    VkAccelerationStructureKHR accelerationStructure,
                                                                    VkDeviceAddress scratchAddress)
{
    ASSERT(asGeometry.size() == asBuildRangeInfo.size() && "asGeometry.size() != asBuildRangeInfo.size()");
    ASSERT(accelerationStructure != VK_NULL_HANDLE &&
           "Acceleration Structure not created, first call createAccelerationStructure");

    const VkAccelerationStructureBuildRangeInfoKHR *rangeInfo = asBuildRangeInfo.data();

    // Build the acceleration structure
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = accelerationStructure;
    buildInfo.dstAccelerationStructure = accelerationStructure;
    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.pGeometries = asGeometry.data();
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
    destroyQueryPool();
    destroyNonCompactedBlas();
}

void BlasBuilder::createQueryPool(uint32_t maxBlasCount)
{
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpci.queryCount = maxBlasCount;
    vkCreateQueryPool(m_device, &qpci, nullptr, &m_queryPool);
}

// This will build multiple BLAS serially, one after the other, ensuring that the process
// stays within the specified memory budget.
bool BlasBuilder::cmdCreateBlas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                                std::vector<AccelKHR> &blasAccel, VkDeviceAddress scratchAddress,
                                VkDeviceSize hintMaxBudget)
{
    // It won't run in parallel, but will process all BLAS within the budget before returning
    return cmdCreateParallelBlas(cmd, blasBuildData, blasAccel, {scratchAddress}, hintMaxBudget);
}

// This function is responsible for building multiple Bottom-Level Acceleration Structures (BLAS) in parallel,
// ensuring that the process stays within the specified memory budget.
//
// Returns:
//   A boolean indicating whether all BLAS in the `blasBuildData` have been built by this function call.
//   Returns `true` if all BLAS were built, `false` otherwise.
bool BlasBuilder::cmdCreateParallelBlas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                                        std::vector<AccelKHR> &blasAccel,
                                        const std::vector<VkDeviceAddress> &scratchAddress, VkDeviceSize hintMaxBudget)
{
    // Initialize the query pool if necessary to handle queries for properties of built acceleration structures.
    initializeQueryPoolIfNeeded(blasBuildData);

    VkDeviceSize processBudget = 0;               // Tracks the total memory used in the construction process.
    uint32_t currentQueryIdx = m_currentQueryIdx; // Local copy of the current query index.

    // Process each BLAS in the data vector while staying under the memory budget.
    while (m_currentBlasIdx < blasBuildData.size() && processBudget < hintMaxBudget) {
        // Build acceleration structures and accumulate the total memory used.
        processBudget += buildAccelerationStructures(cmd, blasBuildData, blasAccel, scratchAddress, hintMaxBudget,
                                                     processBudget, currentQueryIdx);
    }

    // Check if all BLAS have been built.
    return m_currentBlasIdx >= blasBuildData.size();
}

// Initializes a query pool for recording acceleration structure properties if necessary.
// This function ensures a query pool is available if any BLAS in the build data is flagged for compaction.
void BlasBuilder::initializeQueryPoolIfNeeded(const std::vector<AccelerationStructureBuildData> &blasBuildData)
{
    if (!m_queryPool) {
        // Iterate through each BLAS build data element to check if the compaction flag is set.
        for (const auto &blas : blasBuildData) {
            if (blas.hasCompactFlag()) {
                createQueryPool(static_cast<uint32_t>(blasBuildData.size()));
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
VkDeviceSize BlasBuilder::buildAccelerationStructures(VkCommandBuffer cmd,
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
        VkAccelerationStructureCreateInfoKHR createInfo = data.makeCreateInfo();

        // Create and store acceleration structure
        blasAccel[m_currentBlasIdx] = m_alloc->create_accel(createInfo);
        collectedAccel.push_back(blasAccel[m_currentBlasIdx].accel);

        // Setup build information for the current BLAS
        data.buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        data.buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
        data.buildInfo.dstAccelerationStructure = blasAccel[m_currentBlasIdx].accel;
        data.buildInfo.scratchData.deviceAddress = scratchAddress[m_currentBlasIdx % scratchAddress.size()];
        data.buildInfo.pGeometries = data.asGeometry.data();
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
void BlasBuilder::cmdCompactBlas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
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

void BlasBuilder::destroyNonCompactedBlas()
{
    for (auto &blas : m_cleanupBlasAccel) {
        m_alloc->destroy(blas);
    }
    m_cleanupBlasAccel.clear();
}

void BlasBuilder::destroyQueryPool()
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
VkDeviceSize BlasBuilder::getScratchSize(VkDeviceSize hintMaxBudget,
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
void BlasBuilder::getScratchAddresses(VkDeviceSize hintMaxBudget,
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
std::string BlasBuilder::Stats::toString() const
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
VkDeviceSize getMaxScratchSize(const std::vector<AccelerationStructureBuildData> &asBuildData)
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
    m_device = device;
    m_queueIndex = queueIndex;
    m_debug.setup(device);
    m_alloc = &allocator;
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
RaytracingBuilderKHR::~RaytracingBuilderKHR()
{
    if (m_alloc) {
        for (auto &b : m_blas) {
            m_alloc->destroy(b);
        }
        m_alloc->destroy(m_tlas);
    }
    m_blas.clear();
}

//--------------------------------------------------------------------------------------------------
// Returning the constructed top-level acceleration structure
//
VkAccelerationStructureKHR RaytracingBuilderKHR::getAccelerationStructure() const { return m_tlas.accel; }

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
VkDeviceAddress RaytracingBuilderKHR::getBlasDeviceAddress(uint32_t blasId)
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
void RaytracingBuilderKHR::buildBlas(const std::vector<BlasInput> &input, VkBuildAccelerationStructureFlagsKHR flags)
{
    auto numBlas = static_cast<uint32_t>(input.size());
    VkDeviceSize asTotalSize{0};    // Memory size of all allocated BLAS
    VkDeviceSize maxScratchSize{0}; // Largest scratch size

    std::vector<AccelerationStructureBuildData> blasBuildData(numBlas);
    m_blas.resize(numBlas); // Resize to hold all the BLAS
    for (uint32_t idx = 0; idx < numBlas; idx++) {
        blasBuildData[idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildData[idx].asGeometry = input[idx].asGeometry;
        blasBuildData[idx].asBuildRangeInfo = input[idx].asBuildOffsetInfo;

        auto sizeInfo = blasBuildData[idx].finalizeGeometry(m_device, input[idx].flags | flags);
        maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }

    VkDeviceSize hintMaxBudget{256'000'000}; // 256 MB

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    Buffer blasScratchBuffer;

    bool hasCompaction = hasFlag(flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

    BlasBuilder blasBuilder(*m_alloc, m_device);

    uint32_t minAlignment = 128; /*m_rtASProperties.minAccelerationStructureScratchOffsetAlignment*/
    // 1) finding the largest scratch size
    VkDeviceSize scratchSize = blasBuilder.getScratchSize(hintMaxBudget, blasBuildData, minAlignment);
    // 2) allocating the scratch buffer
    blasScratchBuffer = m_alloc->create_buffer(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = scratchSize,
                           .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});
    // 3) getting the device address for the scratch buffer
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.getScratchAddresses(hintMaxBudget, blasBuildData, blasScratchBuffer.address, scratchAddresses,
                                    minAlignment);

    CommandPool m_cmdPool(m_device, m_queueIndex);

    bool finished = false;
    do {
        {
            VkCommandBuffer cmd = m_cmdPool.createCommandBuffer();
            finished = blasBuilder.cmdCreateParallelBlas(cmd, blasBuildData, m_blas, scratchAddresses, hintMaxBudget);
            m_cmdPool.submitAndWait(cmd);
        }
        if (hasCompaction) {
            VkCommandBuffer cmd = m_cmdPool.createCommandBuffer();
            blasBuilder.cmdCompactBlas(cmd, blasBuildData, m_blas);
            m_cmdPool.submitAndWait(cmd); // Submit command buffer and call vkQueueWaitIdle
            blasBuilder.destroyNonCompactedBlas();
        }
    } while (!finished);

    get_default_logger().info("{}", blasBuilder.getStatistics().toString().c_str());

    // Clean up
    // TODO: check this
    // m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(blasScratchBuffer);
}

void RaytracingBuilderKHR::buildTlas(
    const std::vector<VkAccelerationStructureInstanceKHR> &instances,
    VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR*/,
    bool update /*= false*/)
{
    buildTlas(instances, flags, update, false);
}

#ifdef VK_NV_ray_tracing_motion_blur
void RaytracingBuilderKHR::buildTlas(
    const std::vector<VkAccelerationStructureMotionInstanceNV> &instances,
    VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV*/,
    bool update /*= false*/)
{
    buildTlas(instances, flags, update, true);
}
#endif

//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
void RaytracingBuilderKHR::cmdCreateTlas(VkCommandBuffer cmdBuf, uint32_t countInstance, VkDeviceAddress instBufferAddr,
                                         Buffer &scratchBuffer, VkBuildAccelerationStructureFlagsKHR flags, bool update,
                                         bool motion)
{
    AccelerationStructureBuildData tlasBuildData;
    tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    AccelerationStructureGeometryInfo geo = tlasBuildData.makeInstanceGeometry(countInstance, instBufferAddr);
    tlasBuildData.addGeometry(geo);

    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, flags);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = update ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize;

    scratchBuffer = m_alloc->create_buffer(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = scratchSize,
                           .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT});
    VkDeviceAddress scratchAddress = scratchBuffer.address;
    NAME_VK(scratchBuffer.buffer);

    if (update) { // Update the acceleration structure
        tlasBuildData.asGeometry[0].geometry.instances.data.deviceAddress = instBufferAddr;
        tlasBuildData.cmdUpdateAccelerationStructure(cmdBuf, m_tlas.accel, scratchAddress);
    } else { // Create and build the acceleration structure
        VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();

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
        tlasBuildData.cmdBuildAccelerationStructure(cmdBuf, m_tlas.accel, scratchAddress);
    }
}

//--------------------------------------------------------------------------------------------------
// Refit BLAS number blasIdx from updated buffer contents.
//
void RaytracingBuilderKHR::updateBlas(uint32_t blasIdx, BlasInput &blas, VkBuildAccelerationStructureFlagsKHR flags)
{
    ASSERT(size_t(blasIdx) < m_blas.size());

    AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
    buildData.asGeometry = blas.asGeometry;
    buildData.asBuildRangeInfo = blas.asBuildOffsetInfo;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = buildData.finalizeGeometry(m_device, flags);

    // Allocate the scratch buffer and setting the scratch info
    Buffer scratchBuffer = m_alloc->create_buffer(
        VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = sizeInfo.updateScratchSize,
                           .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT});

    // Update the instance buffer on the device side and build the TLAS
    CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();
    buildData.cmdUpdateAccelerationStructure(cmdBuf, m_blas[blasIdx].accel, scratchBuffer.address);
    genCmdBuf.submitAndWait(cmdBuf);

    m_alloc->destroy(scratchBuffer);
}

//-----------------------------------------------------------------------------
// [Top wrapper class for graphics services and resources]
//-----------------------------------------------------------------------------

GFX::GFX(const GFXArgs &args)
{
    ContextCreateInfo vkctx_args{};
    vkctx_args.api_version_major = 1;
    vkctx_args.api_version_minor = 3;

    uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    for (uint32_t i = 0; i < glfw_extension_count; ++i) {
        vkctx_args.instance_extensions.push_back(glfw_extensions[i]);
    }
#ifndef NDEBUG
    vkctx_args.enable_validation();
#endif
    vkctx_args.enable_swapchain();

    ctx.create_instance(vkctx_args);

    vk_check(glfwCreateWindowSurface(ctx.instance, args.window, nullptr, &surface.surface));
    surface.instance = ctx.instance;

    auto compatibles = ctx.query_compatible_devices(vkctx_args, surface.surface);
    if (compatibles.empty()) {
        get_default_logger().critical("No compatible vulkan devices.");
        std::abort();
    }
    ctx.create_device(vkctx_args, compatibles[0]);

    SwapchainCreateInfo swapchain_args{};
    swapchain_args.physical_device = ctx.physical_device;
    swapchain_args.device = ctx.device;
    swapchain_args.queue = ctx.main_queue;
    swapchain_args.surface = surface.surface;
    swapchain_args.width = args.width;
    swapchain_args.height = args.height;
    swapchain_args.max_frames_ahead = 2;
    swapchain = Swapchain(swapchain_args);

    cb_manager = CmdBufManager((uint32_t)swapchain.image_views.size(), ctx.main_queue_family_index, ctx.device);
}

GFX::~GFX()
{
    // swapchain.shutdown();
    // cb_manager.shutdown();
    // vkDestroySurfaceKHR(ctx.instance, surface, nullptr);
    // ctx.shutdown();
}

//-----------------------------------------------------------------------------
// [ImGui integration]
//-----------------------------------------------------------------------------

static void imgui_vk_debug_callback(VkResult result)
{
    if (result != VK_SUCCESS) {
        get_default_logger().error("Vulkan error in imgui: {}", (int)result);
    }
}

GUI::GUI(const GUICreateInfo &info) : window(info.window), gfx(info.gfx)
{
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
        gfx->ctx.instance);
    ImGui_ImplGlfw_InitForVulkan(info.window, true);

    create_render_pass();
    create_framebuffers();

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

    vk_check(vkCreateDescriptorPool(info.gfx->ctx.device, &poolCI, nullptr, &pool));

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = info.gfx->ctx.instance;
    initInfo.PhysicalDevice = info.gfx->ctx.physical_device;
    initInfo.Device = info.gfx->ctx.device;
    initInfo.QueueFamily = info.gfx->ctx.main_queue_family_index;
    initInfo.Queue = info.gfx->ctx.main_queue;
    initInfo.MinImageCount = initInfo.ImageCount = (uint32_t)gfx->swapchain.images.size();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.DescriptorPool = pool;
    initInfo.CheckVkResultFn = imgui_vk_debug_callback;

    ImGui_ImplVulkan_Init(&initInfo, render_pass);

    ImFontConfig font_config;
    font_config.RasterizerDensity = 2.0f;
    io.Fonts->AddFontFromFileTTF((fs::path(DATA_DIR) / "CascadiaCode.ttf").string().c_str(), 16.0f, &font_config);
}

void GUI::create_render_pass()
{
    const Swapchain &swapchain = gfx->swapchain;
    // TODO: Need to change this based on what is rendered before.
    VkAttachmentDescription colorAtt{};
    colorAtt.format = swapchain.format;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pColorAttachments = &ref;
    subpass.colorAttachmentCount = 1;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassCI.pAttachments = &colorAtt;
    renderPassCI.attachmentCount = 1;
    renderPassCI.pSubpasses = &subpass;
    renderPassCI.subpassCount = 1;
    renderPassCI.pDependencies = &dep;
    renderPassCI.dependencyCount = 1;

    vk_check(vkCreateRenderPass(gfx->ctx.device, &renderPassCI, nullptr, &render_pass));
}

void GUI::create_framebuffers()
{
    const Swapchain &swapchain = gfx->swapchain;
    framebuffers.resize(swapchain.images.size());
    for (uint32_t i = 0; i < (uint32_t)swapchain.images.size(); ++i) {
        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.attachmentCount = 1;
        fbCI.pAttachments = &swapchain.image_views[i];
        fbCI.layers = 1;
        fbCI.renderPass = render_pass;
        fbCI.width = swapchain.extent.width;
        fbCI.height = swapchain.extent.height;
        vk_check(vkCreateFramebuffer(gfx->ctx.device, &fbCI, nullptr, &framebuffers[i]));
    }
}

void GUI::render(VkCommandBuffer cmdBuf)
{
    ImGui::Render();
    ImDrawData *drawData = ImGui::GetDrawData();
    const bool isMinimized = (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f);
    if (isMinimized) {
        return;
    }

    VkRenderPassBeginInfo beginInfo = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    beginInfo.renderPass = render_pass;
    beginInfo.framebuffer = framebuffers[gfx->frame_index()];
    beginInfo.renderArea.extent = gfx->swapchain.extent;
    beginInfo.clearValueCount = 0;
    RenderPassRecorder guiPass(cmdBuf, beginInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (show) {
        ImGui_ImplVulkan_RenderDrawData(drawData, cmdBuf);
    }
}

void GUI::resize()
{
    const Swapchain &swapchain = gfx->swapchain;
    ImGui_ImplVulkan_SetMinImageCount((uint32_t)swapchain.images.size());
    // recreate render pass
    destroy_render_pass();
    create_render_pass();
    // recreate framebuffer
    destroy_framebuffers();
    create_framebuffers();
}

GUI::~GUI()
{
    if (!gfx) {
        return;
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    destroy_framebuffers();
    destroy_render_pass();

    vkDestroyDescriptorPool(gfx->ctx.device, pool, nullptr);
}

void GUI::destroy_render_pass() { vkDestroyRenderPass(gfx->ctx.device, render_pass, nullptr); }

void GUI::destroy_framebuffers()
{
    for (auto fb : framebuffers) {
        vkDestroyFramebuffer(gfx->ctx.device, fb, nullptr);
    }
}

void GUI::upload_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (update_fn)
        update_fn();
}

} // namespace vk
} // namespace ks