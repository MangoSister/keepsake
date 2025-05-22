#include "vk_cuda_interop.h"

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
#include <volk.h>
#include <vulkan/vulkan_win32.h>
// clang-format on
#elif defined(__linux__)

#endif

namespace ks
{

CudaBackedVkBuffer::CudaBackedVkBuffer(const ksc::CudaShareableLowLevelMemory &cuda, VkBufferUsageFlags usage,
                                       VkDevice vk_device, VkPhysicalDevice vk_physical_device)
    : cuda(&cuda), vk_device(vk_device), vk_physical_device(vk_physical_device)
{
    VkBufferUsageFlags cuda_mmap_buffer_usage = usage;

    VkMemoryPropertyFlags cuda_mmap_buffer_mem_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkExternalMemoryBufferCreateInfo cuda_buffer_ext_info{.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                                                          .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#elif defined(__linux__)
                                                          .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif
    };

    VkBufferCreateInfo cuda_buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &cuda_buffer_ext_info,
        .size = cuda.allocated_size,
        .usage = cuda_mmap_buffer_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    vk::vk_check(vkCreateBuffer(vk_device, &cuda_buffer_info, nullptr, &mmap_buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vk_device, mmap_buffer, &memRequirements);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    VkImportMemoryWin32HandleInfoKHR handleInfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .pNext = nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
        .handle = cuda.shareable_handle,
        .name = nullptr,
    };
#elif defined(__linux__)
    VkImportMemoryFdInfoKHR handleInfo = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        .fd = (int)(uintptr_t)interop.cuda.shareable_handle,
    };
#endif

    uint32_t memory_type_index = (uint32_t)(~0);
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (memRequirements.memoryTypeBits & (1 << i) &&
            (memProperties.memoryTypes[i].propertyFlags & cuda_mmap_buffer_mem_properties) ==
                cuda_mmap_buffer_mem_properties) {
            memory_type_index = i;
            break;
        }
    }
    if (memory_type_index == (uint32_t)(~0)) {
        fprintf(stderr, "Cannot find suitable memory type.\n");
        std::abort();
    }

    VkMemoryAllocateInfo memAllocation = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = (void *)&handleInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = memory_type_index,
    };

    vk::vk_check(vkAllocateMemory(vk_device, &memAllocation, nullptr, &mmap_buffer_mem));
    vk::vk_check(vkBindBufferMemory(vk_device, mmap_buffer, mmap_buffer_mem, 0));
}

CudaBackedVkBuffer::~CudaBackedVkBuffer()
{
    vkDestroyBuffer(vk_device, mmap_buffer, nullptr);
    vkFreeMemory(vk_device, mmap_buffer_mem, nullptr);
    // ksc::cuda_free_device_low_level(*cuda);
}

} // namespace ks