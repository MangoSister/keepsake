#pragma once
#include "../assertion.h"

#include <functional>
#include <source_location>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include <volk.h>

namespace ks
{
namespace vk
{

// We use volk as the custom loader for Vulkan and we define VK_NO_PROTOTYPES for the whole project (in CMakeLists.txt).
// See discussions on imgui+custom loader: https://github.com/ocornut/imgui/issues/4854
// For VMA, we need to specify VmaVulkanFunctions.

// Some structs has destruction logic in shutdown() instead of dtor because we need to specify the order of
// destruction...

inline void vk_check(VkResult err, const std::source_location location = std::source_location::current())
{
    if (err == 0)
        return;
    fprintf(stderr, "[File: %s (%u:%u), in `%s`] Vulkan Error: VkResult = %d\n", location.file_name(), location.line(),
            location.column(), location.function_name(), err);
    if (err < 0)
        std::abort();
}

//-----------------------------------------------------------------------------
// [Memory allocation]
//-----------------------------------------------------------------------------

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

struct TexelBuffer
{
    Buffer buffer;
    VkBufferView bufferView = VK_NULL_HANDLE;
};

struct PerFrameBuffer
{
    Buffer buffer;
    uint32_t perFrameSize;
    uint32_t numFrames;

    std::vector<uint32_t> getAllOffsets() const
    {
        std::vector<uint32_t> offsets(numFrames);
        for (uint32_t f = 0; f < numFrames; ++f)
            offsets[f] = perFrameSize * f;
        return offsets;
    }
};

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageLayout layout;
};

struct ImageWithView
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageView view;
    VkImageLayout layout;

    VkDescriptorImageInfo imageInfo() const { return VkDescriptorImageInfo{VK_NULL_HANDLE, view, layout}; }
};

struct Texture
{
    ImageWithView image;
    VkSampler sampler;

    VkDescriptorImageInfo imageInfo() const { return VkDescriptorImageInfo{sampler, image.view, image.layout}; }
};

enum class MipmapOption
{
    AutoGenerate,
    PreGenerated,
    OnlyAllocate,
};

struct VulkanAllocator
{
    VulkanAllocator() = default;
    VulkanAllocator(const VmaAllocatorCreateInfo &vmaInfo, uint32_t uploadQueueFamilyIndex, VkQueue uploadQueue);
    void shutdown();

    VulkanAllocator(const VulkanAllocator &other) = delete;
    VulkanAllocator &operator=(const VulkanAllocator &other) = delete;
    VulkanAllocator(VulkanAllocator &&other) = default;
    VulkanAllocator &operator=(VulkanAllocator &&other) = default;

    Buffer createBuffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage);
    Buffer createBuffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage, const uint8_t *data,
                        VkCommandBuffer customCmdBuf = VK_NULL_HANDLE);
    TexelBuffer createTexelBuffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage,
                                  VkBufferViewCreateInfo &bufferViewInfo);
    TexelBuffer createTexelBuffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage,
                                  VkBufferViewCreateInfo &bufferViewInfo, const uint8_t *data,
                                  VkCommandBuffer customCmdBuf = VK_NULL_HANDLE);

    std::byte *map(VmaAllocation allocation);
    void unmap(VmaAllocation allocation);
    void flush(VmaAllocation allocation);

    template <typename TWork>
    void map(VmaAllocation allocation, bool flush, const TWork &work)
    {
        std::byte *ptr = map(allocation);
        work(ptr);
        if (flush) {
            this->flush(allocation);
        }
        unmap(allocation);
    }

    PerFrameBuffer createPerFrameBuffer(const VkBufferCreateInfo &perFrameInfo, VmaMemoryUsage usage,
                                        uint32_t numFrames);

    template <typename TWork>
    void map(const PerFrameBuffer &buffer, uint32_t frameIndex, bool flush, const TWork &work)
    {
        ASSERT(frameIndex < buffer.numFrames);
        std::byte *ptr = map(buffer.buffer.allocation) + frameIndex * buffer.perFrameSize;
        work(ptr);
        if (flush) {
            this->flush(buffer, frameIndex);
        }
        unmap(buffer.buffer.allocation);
    }

    void flush(const PerFrameBuffer &buffer, uint32_t frameIndex);

    template <typename TWork>
    void stageSession(const TWork &work)
    {
        beginStagingSession();
        work(*this);
        endStagingSession();
    }

    void beginStagingSession();
    void endStagingSession();

    VkDeviceSize imageSize(const VkImageCreateInfo &info) const;

    Image createImage(const VkImageCreateInfo &info, VmaMemoryUsage usage, VmaAllocationCreateFlags flags);
    ImageWithView createImageWithView(const VkImageCreateInfo &info, VkImageViewCreateInfo &viewInfo,
                                      VmaMemoryUsage usage);
    ImageWithView createImageWithView(const VkImageCreateInfo &info, VmaMemoryUsage usage, bool cubeMap);
    ImageWithView createColorBuffer(uint32_t width, uint32_t height, VkFormat format, bool sample, bool storage);
    ImageWithView createDepthBuffer(uint32_t width, uint32_t height, bool sample, bool storage);
    ImageWithView createAndTransitImage(const VkImageCreateInfo &info, VkImageViewCreateInfo &viewInfo,
                                        VmaMemoryUsage usage, VkImageLayout layout);
    ImageWithView createAndTransitImage(const VkImageCreateInfo &info, VmaMemoryUsage usage, VkImageLayout layout,
                                        bool cubeMap);
    // Regular 2D texture or 2D texture array only (TODO: cube map, 3D texture, etc).
    ImageWithView createAndUploadImage(const VkImageCreateInfo &info, VmaMemoryUsage usage, const uint8_t *data,
                                       size_t byteSize, VkImageLayout layout, MipmapOption mipmapOption, bool cubeMap);
    Texture createTexture(const ImageWithView &image, const VkSamplerCreateInfo &samplerInfo);

  private:
    // Note: delay destruction of staging buffers.
    Buffer createStagingBuffer(VkDeviceSize bufferSize, const uint8_t *data, VkDeviceSize dataSize,
                               bool autoMapped = true);
    void clearStagingBuffers();

  public:
    void destroy(Buffer &buffer);
    void destroy(TexelBuffer &texelBuffer);
    void destroy(PerFrameBuffer &perFrameBuffer);
    void destroy(Image &image);
    void destroy(ImageWithView &image);
    void destroy(Texture &texture, bool destroyImage);

    VkDevice device;
    VmaAllocator vma;

    VkQueue uploadQueue;
    VkCommandPool uploadCmdPool;
    VkCommandBuffer uploadCmdBuf;

    std::vector<Buffer> stagingBuffers;

    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
};

//-----------------------------------------------------------------------------
// [Basic vulkan object management]
//-----------------------------------------------------------------------------

struct VulkanContextCreateInfo
{
    ~VulkanContextCreateInfo();

    void enable_validation();
    void enable_swapchain();

    uint32_t api_version_major;
    uint32_t api_version_minor;

    std::vector<std::string> instance_extensions;
    std::vector<std::string> instance_layers;

    VkPhysicalDeviceFeatures2 device_features = VkPhysicalDeviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    std::vector<void *> device_features_data;

    std::vector<std::string> device_extensions;

    bool validation = false;

    template <typename T>
    T &add_device_feature()
    {

        T *feature = (T *)malloc(sizeof(T));
        memset(feature, 0, sizeof(T));
        device_features_data.push_back(feature);

        auto getNext = [](const void *ptr) { return (void *)((uint8_t *)ptr + offsetof(T, pNext)); };

        void *ptr = &device_features;
        void *next = getNext(ptr);
        constexpr void *null = nullptr;
        while (memcmp(next, &null, sizeof(void *))) {
            memcpy(&ptr, next, sizeof(void *));
            next = getNext(ptr);
        }
        memcpy(next, &feature, sizeof(T *));
        return *feature;
    }
};

struct CompatibleDevice
{
    VkPhysicalDevice physical_device;
    uint32_t physical_device_index;
    uint32_t queue_family_index;
};

struct VulkanContext
{
    VulkanContext() = default;
    void shutdown();

    VulkanContext(const VulkanContext &other) = delete;
    VulkanContext &operator=(const VulkanContext &other) = delete;
    VulkanContext(VulkanContext &&other) = default;
    VulkanContext &operator=(VulkanContext &&other) = default;

    void create_instance(const VulkanContextCreateInfo &info);
    std::vector<CompatibleDevice> query_compatible_devices(const VulkanContextCreateInfo &info, VkSurfaceKHR surface);
    void create_device(const VulkanContextCreateInfo &info, CompatibleDevice compatible, VkSurfaceKHR surface);

    template <typename TWork>
    void submit_once(const TWork &task);

    VkInstance instance;
    bool validation;
    VkDebugUtilsMessengerEXT debug_messenger;

    // Single physical/logical device.
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceFeatures physical_device_features;
    VkPhysicalDeviceProperties physical_device_properties;

    VkDevice device;

    // Single queue
    // TODO: multi-queue architecture for e.g. async-compute.
    uint32_t main_queue_family_index;
    VkQueue main_queue;

    // Single allocator
    VulkanAllocator allocator;
};

template <typename TWork>
void VulkanContext::submit_once(const TWork &task)
{
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = main_queue_family_index;
    VkCommandPool cmdPool;
    vk_check(vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool));
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = cmdPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    vk_check(vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf));

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(cmdBuf, &beginInfo));
    //////////////////////////////////////////////////////////////////////////
    task(cmdBuf);
    //////////////////////////////////////////////////////////////////////////
    vk_check(vkEndCommandBuffer(cmdBuf));
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    vk_check(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    vk_check(vkQueueSubmit(main_queue, 1, &submitInfo, fence));
    vk_check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuf);
    vkDestroyCommandPool(device, cmdPool, nullptr);
}

//-----------------------------------------------------------------------------
// [Swap chain]
//-----------------------------------------------------------------------------

struct SwapchainCreateInfo
{
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkSurfaceKHR surface;
    uint32_t width;
    uint32_t height;
    uint32_t maxFramesAhead;
};

struct Swapchain
{
    Swapchain() = default;
    explicit Swapchain(const SwapchainCreateInfo &info);
    void shutdown();

    Swapchain(const Swapchain &other) = delete;
    Swapchain &operator=(const Swapchain &other) = delete;
    Swapchain(Swapchain &&other) = default;
    Swapchain &operator=(Swapchain &&other) = default;

    bool acquire();
    bool submitAndPresent(const std::vector<VkCommandBuffer> &cmdBufs);
    void resize(uint32_t width, uint32_t height);

    void createSwapchainAndImages(uint32_t width, uint32_t height);
    void destroySwapchainAndImages();

    uint32_t frameCount() const { return (uint32_t)imageViews.size(); }
    float aspect() const { return (float)extent.width / (float)extent.height; }

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkSurfaceKHR surface;

    VkSwapchainKHR swapchain;
    VkFormat format;
    VkExtent2D extent;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;

    std::vector<VkSemaphore> presentCompleteSemaphores;
    std::vector<VkSemaphore> renderCompleteSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t maxFramesAhead = 0;
    uint32_t renderAheadIndex = 0;
    uint32_t frameIndex;
};

//-----------------------------------------------------------------------------
// [Command buffer management]
//-----------------------------------------------------------------------------

struct CmdBufManager
{
    CmdBufManager() = default;
    CmdBufManager(uint32_t frameCount, uint32_t queueFamilyIndex, VkDevice device);
    void shutdown();

    CmdBufManager(const CmdBufManager &other) = delete;
    CmdBufManager &operator=(const CmdBufManager &other) = delete;
    CmdBufManager(CmdBufManager &&other) = default;
    CmdBufManager &operator=(CmdBufManager &&other) = default;

    void startFrame(uint32_t frameIndex);
    std::vector<VkCommandBuffer> acquireCmdBufs(uint32_t count);
    std::vector<VkCommandBuffer> allAcquired() const
    {
        auto &frame = frames[frameIndex];
        std::vector<VkCommandBuffer> acquired;
        acquired.insert(acquired.end(), frame.cmdBufs.begin(), frame.cmdBufs.begin() + frame.nextCmdBuf);
        return acquired;
    }
    VkCommandBuffer lastAcquired() const
    {
        const auto &frame = frames[frameIndex];
        return frame.nextCmdBuf > 0 ? frame.cmdBufs[frame.nextCmdBuf - 1] : VK_NULL_HANDLE;
    }

  private:
    struct Frame
    {
        VkCommandPool pool;
        std::vector<VkCommandBuffer> cmdBufs;
        uint32_t nextCmdBuf = 0;
    };
    std::vector<Frame> frames;
    VkDevice device;
    uint32_t frameIndex;
};

void encodeCmdNow(VkDevice device, uint32_t queueFamilyIndex, VkQueue queue,
                  const std::function<void(VkCommandBuffer)> &func);

// Convenience RAII wrappers.
struct CmdBufRecorder
{
    CmdBufRecorder(VkCommandBuffer cmdBuf, const VkCommandBufferBeginInfo &beginInfo) : cmdBuf(cmdBuf)
    {
        vk_check(vkBeginCommandBuffer(cmdBuf, &beginInfo));
    }
    ~CmdBufRecorder() { vk_check(vkEndCommandBuffer(cmdBuf)); }

    CmdBufRecorder(const CmdBufRecorder &other) = delete;
    CmdBufRecorder(CmdBufRecorder &&other) = delete;
    CmdBufRecorder &operator=(const CmdBufRecorder &other) = delete;
    CmdBufRecorder &operator=(CmdBufRecorder &&other) = delete;

    VkCommandBuffer cmdBuf;
};

struct RenderPassRecorder
{
    RenderPassRecorder(VkCommandBuffer cmdBuf, const VkRenderPassBeginInfo &beginInfo,
                       VkSubpassContents subpassContents)
        : cmdBuf(cmdBuf)
    {
        vkCmdBeginRenderPass(cmdBuf, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
    }
    ~RenderPassRecorder() { vkCmdEndRenderPass(cmdBuf); }

    RenderPassRecorder(const RenderPassRecorder &other) = delete;
    RenderPassRecorder(RenderPassRecorder &&other) = delete;
    RenderPassRecorder &operator=(const RenderPassRecorder &other) = delete;
    RenderPassRecorder &operator=(RenderPassRecorder &&other) = delete;

    VkCommandBuffer cmdBuf;
};

//-----------------------------------------------------------------------------
// [Convenience helper for setting up descriptor sets]
//-----------------------------------------------------------------------------

struct DescriptorSetHelper
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorPool createPool(VkDevice device, uint32_t maxSets) const;
    VkDescriptorSetLayout createSetLayout(VkDevice device) const;
    VkWriteDescriptorSet makeWrite(VkDescriptorSet dstSet, uint32_t dstBinding) const;
    VkWriteDescriptorSet makeWriteArray(VkDescriptorSet dstSet, uint32_t dstBinding, uint32_t start,
                                        uint32_t count) const;
};

//-----------------------------------------------------------------------------
// [Top wrapper class for graphics services and resources]
//-----------------------------------------------------------------------------

struct GFXArgs
{
    int width, height;
    GLFWwindow *window = nullptr;
};

struct GFX
{
    GFX() = default;
    GFX(const GFX &other) = delete;
    GFX &operator=(const GFX &other) = delete;
    GFX(GFX &&other) = default;
    GFX &operator=(GFX &&other) = default;

    explicit GFX(const GFXArgs &args);
    void shutdown();

    uint32_t frameIndex() const { return swapchain.frameIndex; }
    uint32_t renderAheadIndex() const { return swapchain.renderAheadIndex; }

    bool acquireFrame()
    {
        if (!swapchain.acquire()) {
            return false;
        }
        return true;
    }

    void startFrame()
    {
        uint32_t frameIndex = swapchain.frameIndex;
        cmdBufManager.startFrame(frameIndex);
    }

    bool submitFrame()
    {
        uint32_t frameIndex = swapchain.frameIndex;
        return swapchain.submitAndPresent(cmdBufManager.allAcquired());
    }

    Swapchain swapchain;
    CmdBufManager cmdBufManager;
    VkSurfaceKHR surface;
    VulkanContext vkctx;
};

//-----------------------------------------------------------------------------
// [ImGui integration]
//-----------------------------------------------------------------------------

struct GUICreateInfo
{
    const GFX *gfx = nullptr;
    GLFWwindow *window = nullptr;
};

struct GUI
{
    GUI() = default;
    explicit GUI(const GUICreateInfo &info);
    void shutdown();

    void render(VkCommandBuffer cmdBuf);

    void resize();
    void createRenderPass();
    void destroyRenderPass();
    void createFramebuffers();
    void destroyFramebuffers();

    void updateFrame();

    GLFWwindow *window = nullptr;
    const GFX *gfx = nullptr;

    VkDescriptorPool pool;
    VkRenderPass renderPass;
    std::vector<VkFramebuffer> framebuffers;

    std::function<void()> update_fn;
    bool show = true;
};

} // namespace vk
} // namespace ks