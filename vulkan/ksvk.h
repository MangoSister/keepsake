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
    VkBufferView buffer_view = VK_NULL_HANDLE;
};

struct PerFrameBuffer
{
    Buffer buffer;
    uint32_t per_frame_size;
    uint32_t num_frames;

    std::vector<uint32_t> get_all_offsets() const
    {
        std::vector<uint32_t> offsets(num_frames);
        for (uint32_t f = 0; f < num_frames; ++f)
            offsets[f] = per_frame_size * f;
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

    VkDescriptorImageInfo image_info() const { return VkDescriptorImageInfo{VK_NULL_HANDLE, view, layout}; }
};

struct Texture
{
    ImageWithView image;
    VkSampler sampler;

    VkDescriptorImageInfo image_info() const { return VkDescriptorImageInfo{sampler, image.view, image.layout}; }
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
    VulkanAllocator(const VmaAllocatorCreateInfo &vma_info, uint32_t upload_queu_family_index, VkQueue upload_queue);
    void shutdown();

    VulkanAllocator(const VulkanAllocator &other) = delete;
    VulkanAllocator &operator=(const VulkanAllocator &other) = delete;
    VulkanAllocator(VulkanAllocator &&other) = default;
    VulkanAllocator &operator=(VulkanAllocator &&other) = default;

    Buffer create_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage);
    Buffer create_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage, const uint8_t *data,
                         VkCommandBuffer custom_cb = VK_NULL_HANDLE);
    TexelBuffer create_texel_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage,
                                    VkBufferViewCreateInfo &buffer_view_info);
    TexelBuffer create_texel_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage,
                                    VkBufferViewCreateInfo &buffer_view_info, const uint8_t *data,
                                    VkCommandBuffer custom_cb = VK_NULL_HANDLE);

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

    PerFrameBuffer create_per_frame_buffer(const VkBufferCreateInfo &per_frame_info, VmaMemoryUsage usage,
                                           uint32_t num_frames);

    template <typename TWork>
    void map(const PerFrameBuffer &buffer, uint32_t frame_index, bool flush, const TWork &work)
    {
        ASSERT(frame_index < buffer.num_frames);
        std::byte *ptr = map(buffer.buffer.allocation) + frame_index * buffer.per_frame_size;
        work(ptr);
        if (flush) {
            this->flush(buffer, frame_index);
        }
        unmap(buffer.buffer.allocation);
    }

    void flush(const PerFrameBuffer &buffer, uint32_t frame_index);

    template <typename TWork>
    void stage_session(const TWork &work)
    {
        begin_staging_session();
        work(*this);
        end_staging_session();
    }

    void begin_staging_session();
    void end_staging_session();

    VkDeviceSize image_size(const VkImageCreateInfo &info) const;

    Image create_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, VmaAllocationCreateFlags flags);
    ImageWithView create_image_with_view(const VkImageCreateInfo &info, VkImageViewCreateInfo &view_info,
                                         VmaMemoryUsage usage);
    ImageWithView create_image_with_view(const VkImageCreateInfo &info, VmaMemoryUsage usage, bool cube_map);
    ImageWithView create_color_buffer(uint32_t width, uint32_t height, VkFormat format, bool sample, bool storage);
    ImageWithView create_depth_buffer(uint32_t width, uint32_t height, bool sample, bool storage);
    ImageWithView create_and_transit_image(const VkImageCreateInfo &info, VkImageViewCreateInfo &view_info,
                                           VmaMemoryUsage usage, VkImageLayout layout);
    ImageWithView create_and_transit_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, VkImageLayout layout,
                                           bool cube_map);
    // Regular 2D texture or 2D texture array only (TODO: cube map, 3D texture, etc).
    ImageWithView create_and_upload_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, const uint8_t *data,
                                          size_t byte_size, VkImageLayout layout, MipmapOption mipmap_option,
                                          bool cube_map);
    Texture create_texture(const ImageWithView &image, const VkSamplerCreateInfo &sampler_info);

  private:
    // Note: delay destruction of staging buffers.
    Buffer create_staging_buffer(VkDeviceSize buffer_size, const uint8_t *data, VkDeviceSize data_size,
                                 bool auto_mapped = true);
    void clear_staging_buffer();

  public:
    void destroy(Buffer &buffer);
    void destroy(TexelBuffer &texel_buffer);
    void destroy(PerFrameBuffer &per_frame_buffer);
    void destroy(Image &image);
    void destroy(ImageWithView &image);
    void destroy(Texture &texture, bool destroy_image);

    VkDevice device;
    VmaAllocator vma;

    VkQueue upload_queue;
    VkCommandPool upload_cp;
    VkCommandBuffer upload_cb;

    std::vector<Buffer> staging_buffers;

    VkDeviceSize min_uniform_buffer_offset_alignment;
    VkDeviceSize min_storage_buffer_offset_alignment;
    VkDeviceSize min_texel_buffer_offset_alignment;
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
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkSurfaceKHR surface;
    uint32_t width;
    uint32_t height;
    uint32_t max_frames_ahead;
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
    bool submit_and_present(const std::vector<VkCommandBuffer> &cbs);
    void resize(uint32_t width, uint32_t height);

    void create_swapchain_and_images(uint32_t width, uint32_t height);
    void destroy_swapchain_and_images();

    uint32_t frame_count() const { return (uint32_t)image_views.size(); }
    float aspect() const { return (float)extent.width / (float)extent.height; }

    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    VkSurfaceKHR surface;

    VkSwapchainKHR swapchain;
    VkFormat format;
    VkExtent2D extent;
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;

    std::vector<VkSemaphore> present_complete_semaphores;
    std::vector<VkSemaphore> render_complete_semaphores;
    std::vector<VkFence> inflight_fences;
    uint32_t max_frames_ahead = 0;
    uint32_t render_ahead_index = 0;
    uint32_t frame_index;
};

//-----------------------------------------------------------------------------
// [Command buffer management]
//-----------------------------------------------------------------------------

struct CmdBufManager
{
    CmdBufManager() = default;
    CmdBufManager(uint32_t frame_count, uint32_t queue_family_index, VkDevice device);
    void shutdown();

    CmdBufManager(const CmdBufManager &other) = delete;
    CmdBufManager &operator=(const CmdBufManager &other) = delete;
    CmdBufManager(CmdBufManager &&other) = default;
    CmdBufManager &operator=(CmdBufManager &&other) = default;

    void start_frame(uint32_t frame_index);
    std::vector<VkCommandBuffer> acquire_cbs(uint32_t count);
    std::vector<VkCommandBuffer> all_acquired() const
    {
        auto &frame = frames[frame_index];
        std::vector<VkCommandBuffer> acquired;
        acquired.insert(acquired.end(), frame.cbs.begin(), frame.cbs.begin() + frame.next_cb);
        return acquired;
    }
    VkCommandBuffer last_acquired() const
    {
        const auto &frame = frames[frame_index];
        return frame.next_cb > 0 ? frame.cbs[frame.next_cb - 1] : VK_NULL_HANDLE;
    }

  private:
    struct Frame
    {
        VkCommandPool pool;
        std::vector<VkCommandBuffer> cbs;
        uint32_t next_cb = 0;
    };
    std::vector<Frame> frames;
    VkDevice device;
    uint32_t frame_index;
};

void encode_cmd_now(VkDevice device, uint32_t queue_family_index, VkQueue queue,
                    const std::function<void(VkCommandBuffer)> &func);

// Convenience RAII wrappers.
struct CmdBufRecorder
{
    CmdBufRecorder(VkCommandBuffer cb, const VkCommandBufferBeginInfo &begin_info) : cb(cb)
    {
        vk_check(vkBeginCommandBuffer(cb, &begin_info));
    }
    ~CmdBufRecorder() { vk_check(vkEndCommandBuffer(cb)); }

    CmdBufRecorder(const CmdBufRecorder &other) = delete;
    CmdBufRecorder(CmdBufRecorder &&other) = delete;
    CmdBufRecorder &operator=(const CmdBufRecorder &other) = delete;
    CmdBufRecorder &operator=(CmdBufRecorder &&other) = delete;

    VkCommandBuffer cb;
};

struct RenderPassRecorder
{
    RenderPassRecorder(VkCommandBuffer cmdBuf, const VkRenderPassBeginInfo &beginInfo,
                       VkSubpassContents subpassContents)
        : cb(cmdBuf)
    {
        vkCmdBeginRenderPass(cmdBuf, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
    }
    ~RenderPassRecorder() { vkCmdEndRenderPass(cb); }

    RenderPassRecorder(const RenderPassRecorder &other) = delete;
    RenderPassRecorder(RenderPassRecorder &&other) = delete;
    RenderPassRecorder &operator=(const RenderPassRecorder &other) = delete;
    RenderPassRecorder &operator=(RenderPassRecorder &&other) = delete;

    VkCommandBuffer cb;
};

//-----------------------------------------------------------------------------
// [Convenience helper for setting up descriptor sets]
//-----------------------------------------------------------------------------

struct DescriptorSetHelper
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorPool create_pool(VkDevice device, uint32_t max_sets) const;
    VkDescriptorSetLayout create_set_layout(VkDevice device) const;
    VkWriteDescriptorSet make_write(VkDescriptorSet dst_set, uint32_t dst_binding) const;
    VkWriteDescriptorSet make_write_array(VkDescriptorSet dst_set, uint32_t dst_binding, uint32_t start,
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

    uint32_t frame_index() const { return swapchain.frame_index; }
    uint32_t render_ahead_index() const { return swapchain.render_ahead_index; }

    bool acquire_frame()
    {
        if (!swapchain.acquire()) {
            return false;
        }
        return true;
    }

    void start_frame()
    {
        uint32_t frameIndex = swapchain.frame_index;
        cb_manager.start_frame(frameIndex);
    }

    bool submit_frame()
    {
        uint32_t frameIndex = swapchain.frame_index;
        return swapchain.submit_and_present(cb_manager.all_acquired());
    }

    Swapchain swapchain;
    CmdBufManager cb_manager;
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
    void create_render_pass();
    void destroy_render_pass();
    void create_framebuffers();
    void destroy_framebuffers();

    void upload_frame();

    GLFWwindow *window = nullptr;
    const GFX *gfx = nullptr;

    VkDescriptorPool pool;
    VkRenderPass render_pass;
    std::vector<VkFramebuffer> framebuffers;

    std::function<void()> update_fn;
    bool show = true;
};

} // namespace vk
} // namespace ks