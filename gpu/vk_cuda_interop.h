#pragma once

#include "../cuda/memory_low_level.h"
#include "ksvk.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace ks
{

struct CudaBackedVkBuffer
{
    CudaBackedVkBuffer(const ksc::CudaShareableLowLevelMemory &cuda, VkBufferUsageFlags usage, VkDevice vk_device,
                       VkPhysicalDevice vk_physical_device);
    ~CudaBackedVkBuffer();

    VkDevice vk_device;
    VkPhysicalDevice vk_physical_device;
    VkBuffer mmap_buffer;
    VkDeviceMemory mmap_buffer_mem;
    const ksc::CudaShareableLowLevelMemory *cuda;
};

} // namespace ks
