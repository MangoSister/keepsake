#pragma once

#include "../cuda/memory_low_level.h"
#include "ksvk.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace ks
{

struct CudaBackedVkBuffer
{
    CudaBackedVkBuffer() = default;
    CudaBackedVkBuffer(const ksc::CudaShareableLowLevelMemory &cuda, VkBufferUsageFlags usage, VkDevice vk_device,
                       VkPhysicalDevice vk_physical_device);
    ~CudaBackedVkBuffer();

    VkDevice vk_device = VK_NULL_HANDLE;
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    VkBuffer mmap_buffer = VK_NULL_HANDLE;
    VkDeviceMemory mmap_buffer_mem = VK_NULL_HANDLE;
    const ksc::CudaShareableLowLevelMemory *cuda = nullptr;
};

} // namespace ks
