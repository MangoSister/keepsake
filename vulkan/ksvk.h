#pragma once
#include "../assertion.h"
#include "../hash.h"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <source_location>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

// A lot of the components are adapted from nvvk: https://github.com/nvpro-samples/nvpro_core

// TODO: consider integration with:
// https://github.com/wolfpld/tracy
// https://developer.nvidia.com/nsight-aftermath

inline void vk_check(VkResult err, const std::source_location location = std::source_location::current())
{
    if (err == 0)
        return;
    fprintf(stderr, "[File: %s (%u:%u), in `%s`] Vulkan Error: VkResult = %d\n", location.file_name(), location.line(),
            location.column(), location.function_name(), err);
    if (err < 0)
        std::abort();
}

// For object lifecycle management, we do RAII when convenient.

// However, a lot of classes can't acquire resources during construction, and need to repeatedly release/re-acquire
// resources. For those classes, instead of strict RAII we follow these rules instead of for lifecycle management.
// - Allow default construction to an "empty/uninitialized" but valid state (no dangling pointers, handles, etc).
// - Non-default construction to a "ready" state.
// - Destructor should handle both empty and ready states, no need to manually deinit().
// - init()/deinit() to switch between empty and ready state (the object is still constructed). Internally use is_init()
// to safeguard repeat init/deinit. is_init() can also be queried by others.
//   init() is usually the internal implementation of non-default ctors, and deinit() is usually the internal
//   implementation of dtor.

// Many structs/classes are move-only and implement the move as swap.
#define NO_COPY_AND_SWAP_AS_MOVE(STRUCT_TYPE)                                                                          \
    STRUCT_TYPE(const STRUCT_TYPE &other) = delete;                                                                    \
                                                                                                                       \
    STRUCT_TYPE &operator=(const STRUCT_TYPE &other) = delete;                                                         \
                                                                                                                       \
    STRUCT_TYPE(STRUCT_TYPE &&other) noexcept : STRUCT_TYPE()                                                          \
    {                                                                                                                  \
        swap(*this, other);                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    STRUCT_TYPE &operator=(STRUCT_TYPE &&other) noexcept                                                               \
    {                                                                                                                  \
        if (this != &other) {                                                                                          \
            swap(*this, other);                                                                                        \
        }                                                                                                              \
        return *this;                                                                                                  \
    }

//-----------------------------------------------------------------------------
// [Debug util]
//-----------------------------------------------------------------------------
class DebugUtil
{
  public:
    DebugUtil() = default;
    DebugUtil(VkDevice device) { setup(device); }

    void setup(VkDevice device);
    void setObjectName(const uint64_t object, const std::string &name, VkObjectType t) const;
    static bool isEnabled() { return s_enabled; }

    // clang-format off
    void setObjectName(VkBuffer object, const std::string& name) const               { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_BUFFER); }
    void setObjectName(VkBufferView object, const std::string& name) const           { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_BUFFER_VIEW); }
    void setObjectName(VkCommandBuffer object, const std::string& name) const        { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_BUFFER ); }
    void setObjectName(VkCommandPool object, const std::string& name) const          { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_COMMAND_POOL ); }
    void setObjectName(VkDescriptorPool object, const std::string& name) const       { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_POOL); }
    void setObjectName(VkDescriptorSet object, const std::string& name) const        { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET); }
    void setObjectName(VkDescriptorSetLayout object, const std::string& name) const  { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT); }
    void setObjectName(VkDevice object, const std::string& name) const               { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DEVICE); }
    void setObjectName(VkDeviceMemory object, const std::string& name) const         { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_DEVICE_MEMORY); }
    void setObjectName(VkFramebuffer object, const std::string& name) const          { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_FRAMEBUFFER); }
    void setObjectName(VkImage object, const std::string& name) const                { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_IMAGE); }
    void setObjectName(VkImageView object, const std::string& name) const            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_IMAGE_VIEW); }
    void setObjectName(VkPipeline object, const std::string& name) const             { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE); }
    void setObjectName(VkPipelineLayout object, const std::string& name) const       { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_PIPELINE_LAYOUT); }
    void setObjectName(VkQueryPool object, const std::string& name) const            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_QUERY_POOL); }
    void setObjectName(VkQueue object, const std::string& name) const                { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_QUEUE); }
    void setObjectName(VkRenderPass object, const std::string& name) const           { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_RENDER_PASS); }
    void setObjectName(VkSampler object, const std::string& name) const              { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SAMPLER); }
    void setObjectName(VkSemaphore object, const std::string& name) const            { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SEMAPHORE); }
    void setObjectName(VkShaderModule object, const std::string& name) const         { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SHADER_MODULE); }
    void setObjectName(VkSwapchainKHR object, const std::string& name) const         { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_SWAPCHAIN_KHR); }

    #if VK_NV_ray_tracing
    void setObjectName(VkAccelerationStructureNV object, const std::string& name) const  { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV); }
    #endif
    #if VK_KHR_acceleration_structure
    void setObjectName(VkAccelerationStructureKHR object, const std::string& name) const { setObjectName((uint64_t)object, name, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR); }
    #endif
    // clang-format on

    //
    //---------------------------------------------------------------------------
    //
    void beginLabel(VkCommandBuffer cmdBuf, const std::string &label);
    void endLabel(VkCommandBuffer cmdBuf);
    void insertLabel(VkCommandBuffer cmdBuf, const std::string &label);
    //
    // Begin and End Command Label MUST be balanced, this helps as it will always close the opened label
    //
    struct ScopedCmdLabel
    {
        ScopedCmdLabel(VkCommandBuffer cmdBuf, const std::string &label) : m_cmdBuf(cmdBuf)
        {
            if (DebugUtil::s_enabled) {
                VkDebugUtilsLabelEXT s{
                    VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
                DebugUtil::s_vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &s);
            }
        }
        ~ScopedCmdLabel()
        {
            if (DebugUtil::s_enabled) {
                DebugUtil::s_vkCmdEndDebugUtilsLabelEXT(m_cmdBuf);
            }
        }
        void setLabel(const std::string &label) const
        {
            if (DebugUtil::s_enabled) {
                VkDebugUtilsLabelEXT s{
                    VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
                DebugUtil::s_vkCmdInsertDebugUtilsLabelEXT(m_cmdBuf, &s);
            }
        }

      private:
        VkCommandBuffer m_cmdBuf;
    };

    ScopedCmdLabel scopeLabel(VkCommandBuffer cmdBuf, const std::string &label)
    {
        return ScopedCmdLabel(cmdBuf, label);
    }

  protected:
    VkDevice m_device{VK_NULL_HANDLE};
    static bool s_enabled;

    // Extension function pointers
    static PFN_vkCmdBeginDebugUtilsLabelEXT s_vkCmdBeginDebugUtilsLabelEXT;
    static PFN_vkCmdEndDebugUtilsLabelEXT s_vkCmdEndDebugUtilsLabelEXT;
    static PFN_vkCmdInsertDebugUtilsLabelEXT s_vkCmdInsertDebugUtilsLabelEXT;
    static PFN_vkSetDebugUtilsObjectNameEXT s_vkSetDebugUtilsObjectNameEXT;
};

//////////////////////////////////////////////////////////////////////////
/// Macros to help automatically naming variables.
/// Names will be in the form of MyClass::m_myBuffer (in example.cpp:123)
///
/// To use:
/// - Debug member class MUST be named 'm_debug'
/// - Individual name: NAME_VK(m_myBuffer.buffer) or with and index NAME_IDX_VK(m_texture.image, i)
/// - Create/associate and name, instead of
///     pipeline = createPipeline();
///     NAME_VK(pipeline)
///   call
///     CREATE_NAMED_VK(pipeline , createPipeline());
/// - Scope functions can also be automatically named, at the beginning of a function
///   call LABEL_SCOPE_VK( commandBuffer )
///
///
// clang-format off
#define S__(x) #x
#define S_(x) S__(x)
#define S__LINE__ S_(__LINE__)

inline const char* fileNameSplitter(const char* n) { return std::max<const char*>(n, std::max(strrchr(n, '\\') + 1, strrchr(n, '/') + 1)); }
inline const char* upToLastSpace(const char* n) { return std::max<const char*>(n, strrchr(n, ' ') + 1); }
#define CLASS_NAME upToLastSpace(typeid(*this).name())
#define NAME_FILE_LOCATION  std::string(" in ") + std::string(fileNameSplitter(__FILE__)) + std::string(":" S__LINE__ ")")

// Individual naming
#define NAME_VK(_x) m_debug.setObjectName(_x, (std::string(CLASS_NAME) + std::string("::") + std::string(#_x " (") + NAME_FILE_LOCATION).c_str())
#define NAME2_VK(_x, _s) m_debug.setObjectName(_x, (std::string(_s) + std::string(" (" #_x) + NAME_FILE_LOCATION).c_str())
#define NAME_IDX_VK(_x, _i) m_debug.setObjectName(_x, \
                            (std::string(CLASS_NAME) + std::string("::") + std::string(#_x " (" #_i "=") + std::to_string(_i) + std::string(", ") + NAME_FILE_LOCATION).c_str())

// Name in creation
#define CREATE_NAMED_VK(_x, _c)              \
  _x = _c;                                   \
  NAME_VK(_x);
#define CREATE_NAMED_IDX_VK(_x, _i, _c)      \
  _x = _c;                                   \
  NAME_IDX_VK(_x, _i);

// Running scope
#define LABEL_SCOPE_VK(_cmd)                                                                                                                \
  auto _scopeLabel =  m_debug.scopeLabel(_cmd, std::string(CLASS_NAME) + std::string("::") + std::string(__func__) + std::string(", in ")   \
                                   + std::string(fileNameSplitter(__FILE__)) + std::string(":" S__LINE__ ")"))


// Non-defined named variable of the above macros (Ex: m_myDbg->DBG_NAME(vulan_obj); )
#define DBG_NAME(_x)                                                                                                   \
  setObjectName(_x, (std::string(CLASS_NAME) + std::string("::") + std::string(#_x " (") + NAME_FILE_LOCATION).c_str())
#define DBG_NAME_IDX(_x, _i)                                                                                           \
  setObjectName(_x, (std::string(CLASS_NAME) + std::string("::") + std::string(#_x " (" #_i "=") + std::to_string(_i)  \
                     + std::string(", ") + NAME_FILE_LOCATION)                                                         \
                        .c_str())
#define DBG_SCOPE(_cmd)                                                                                                \
  scopeLabel(_cmd, std::string(CLASS_NAME) + std::string("::") + std::string(__func__) + std::string(", in ")          \
                       + std::string(fileNameSplitter(__FILE__)) + std::string(":" S__LINE__ ")"))

// clang-format on

//-----------------------------------------------------------------------------
// [Memory allocation]
//-----------------------------------------------------------------------------

struct Allocator;

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkDeviceAddress address;
};

struct TexelBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkBufferView buffer_view = VK_NULL_HANDLE;
};

struct PerFrameBuffer
{
    std::vector<uint32_t> get_all_offsets() const
    {
        std::vector<uint32_t> offsets(num_frames);
        for (uint32_t f = 0; f < num_frames; ++f)
            offsets[f] = per_frame_size * f;
        return offsets;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    uint32_t per_frame_size;
    uint32_t num_frames;
};

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
};

struct ImageWithView
{
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageView view;
};

struct Texture
{
    ImageWithView image;
    VkSampler sampler;

    bool own_image;
};

enum class MipmapOption
{
    AutoGenerate,
    PreGenerated,
    OnlyAllocate,
};

struct AccelKHR
{
    VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkDeviceAddress address{0};
};

struct Allocator
{
    Allocator() = default;
    Allocator(const VmaAllocatorCreateInfo &vma_info, uint32_t upload_queu_family_index, VkQueue upload_queue);
    ~Allocator();

    friend void swap(Allocator &x, Allocator &y) noexcept
    {
        using std::swap;

        swap(x.device, y.device);
        swap(x.vma, y.vma);
        swap(x.upload_queue, y.upload_queue);
        swap(x.upload_cp, y.upload_cp);
        swap(x.upload_cb, y.upload_cb);
        swap(x.staging_buffers, y.staging_buffers);
        swap(x.min_uniform_buffer_offset_alignment, y.min_uniform_buffer_offset_alignment);
        swap(x.min_storage_buffer_offset_alignment, y.min_storage_buffer_offset_alignment);
        swap(x.min_texel_buffer_offset_alignment, y.min_texel_buffer_offset_alignment);
    }

    NO_COPY_AND_SWAP_AS_MOVE(Allocator)

    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
    Buffer create_buffer(const VkBufferCreateInfo &info, VmaMemoryUsage usage = VMA_MEMORY_USAGE_AUTO,
                         VmaAllocationCreateFlags flags = 0, const std::byte *data = nullptr);

    template <typename T>
    Buffer create_buffer(const VkBufferCreateInfo &info_, VmaMemoryUsage usage, VmaAllocationCreateFlags flags,
                         const std::vector<T> &vec)
    {
        VkBufferCreateInfo info = info_;
        info.size = vec.size() * sizeof(T);
        return create_buffer(info, usage, flags, reinterpret_cast<const std::byte *>(vec.data()));
    }

    TexelBuffer create_texel_buffer(const VkBufferCreateInfo &info, VkBufferViewCreateInfo &buffer_view_info,
                                    VmaMemoryUsage usage = VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags flags = 0,
                                    const std::byte *data = nullptr);

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
    ImageWithView create_and_upload_image(const VkImageCreateInfo &info, VmaMemoryUsage usage, const std::byte *data,
                                          size_t byte_size, VkImageLayout layout, MipmapOption mipmap_option,
                                          bool cube_map);
    Texture create_texture(const ImageWithView &image, const VkSamplerCreateInfo &sampler_info, bool own_image);

    AccelKHR create_accel(const VkAccelerationStructureCreateInfoKHR &accel_info);

    template <typename T, typename... Args>
        requires(std::same_as<T, Buffer> || std::same_as<T, TexelBuffer> || std::same_as<T, PerFrameBuffer> ||
                 std::same_as<T, Image> || std::same_as<T, ImageWithView> || std::same_as<T, Texture> ||
                 std::same_as<T, AccelKHR>)
    T create(Args &&...args)
    {
        if constexpr (std::is_same_v<T, Buffer>) {
            return create_buffer(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<T, TexelBuffer>) {
            return create_texel_buffer(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<T, PerFrameBuffer>) {
            return create_per_frame_buffer(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<T, Image>) {
            return create_image(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<T, ImageWithView>) {
            return create_image_with_view(std::forward<Args>(args)...);
        } else if constexpr (std::is_same_v<T, AccelKHR>) {
            return create_texture(std::forward<Args>(args)...);
        } else {
            return create_accel(std::forward<Args>(args)...);
        }
    }

    void destroy(const Buffer &buffer);
    void destroy(const TexelBuffer &texel_buffer);
    void destroy(const PerFrameBuffer &per_frame_buffer);
    void destroy(const Image &image);
    void destroy(const ImageWithView &image_with_view);
    void destroy(const Texture &texture);
    void destroy(const AccelKHR &accel);

  public:
    VkDevice device = VK_NULL_HANDLE;

  private:
    // Note: delay destruction of staging buffers.
    Buffer create_staging_buffer(VkDeviceSize buffer_size, const std::byte *data, VkDeviceSize data_size);
    void clear_staging_buffer();

    VmaAllocator vma = VK_NULL_HANDLE;

    VkQueue upload_queue = VK_NULL_HANDLE;
    VkCommandPool upload_cp = VK_NULL_HANDLE;
    VkCommandBuffer upload_cb = VK_NULL_HANDLE;

    std::vector<Buffer> staging_buffers;

    VkDeviceSize min_uniform_buffer_offset_alignment = 0;
    VkDeviceSize min_storage_buffer_offset_alignment = 0;
    VkDeviceSize min_texel_buffer_offset_alignment = 0;
};

template <typename T>
struct AutoRelease
{
    AutoRelease() = default;

    template <typename... Args>
    AutoRelease(std::shared_ptr<Allocator> allocator, Args &&...args)
        : allocator(allocator), obj(allocator->create<T>(std::forward<Args>(args)...))
    {}

    ~AutoRelease()
    {
        if (allocator) {
            allocator->destroy(obj);
        }
    }

    friend void swap(AutoRelease &x, AutoRelease &y)
    {
        using std::swap;
        swap(x.obj, y.obj);
        swap(x.allocator, y.allocator);
    }

    NO_COPY_AND_SWAP_AS_MOVE(AutoRelease)

    T *operator->() { return &obj; }
    const T *operator->() const { return &obj; }

    T &operator*() { return obj; }
    const T *&operator*() const { return obj; }

    T obj;
    std::shared_ptr<Allocator> allocator;
};

//-----------------------------------------------------------------------------
// [Basic vulkan object management]
//-----------------------------------------------------------------------------

struct ContextCreateInfo
{
    ~ContextCreateInfo();

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

        auto get_next = [](const void *ptr) { return (void *)((std::byte *)ptr + offsetof(T, pNext)); };

        void *ptr = &device_features;
        void *next = get_next(ptr);
        constexpr void *null = nullptr;
        while (memcmp(next, &null, sizeof(void *))) {
            memcpy(&ptr, next, sizeof(void *));
            next = get_next(ptr);
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

struct Context
{
    Context() = default;
    ~Context();

    friend void swap(Context &x, Context &y) noexcept
    {
        using std::swap;

        swap(x.instance, y.instance);
        swap(x.validation, y.validation);
        swap(x.debug_messenger, y.debug_messenger);
        swap(x.physical_device, y.physical_device);
        swap(x.physical_device_features, y.physical_device_features);
        swap(x.physical_device_properties, y.physical_device_properties);
        swap(x.device, y.device);
        swap(x.main_queue_family_index, y.main_queue_family_index);
        swap(x.main_queue, y.main_queue);
        swap(x.allocator, y.allocator);
    }

    NO_COPY_AND_SWAP_AS_MOVE(Context)

    void create_instance(const ContextCreateInfo &info);
    std::vector<CompatibleDevice> query_compatible_devices(const ContextCreateInfo &info, VkSurfaceKHR surface);
    void create_device(const ContextCreateInfo &info, CompatibleDevice compatible);

    template <typename TWork>
    void submit_once(const TWork &task);

    VkInstance instance = VK_NULL_HANDLE;
    bool validation = false;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

    // Single physical/logical device.
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceFeatures physical_device_features = {};
    VkPhysicalDeviceProperties physical_device_properties = {};

    VkDevice device = VK_NULL_HANDLE;

    // Single queue
    // TODO: multi-queue architecture for e.g. async-compute.
    uint32_t main_queue_family_index = 0;
    VkQueue main_queue = VK_NULL_HANDLE;

    // Single allocator
    // Shared ptr to allow auto release...
    std::shared_ptr<Allocator> allocator;
};

template <typename TWork>
void Context::submit_once(const TWork &task)
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

// Utility class for a single VkCommandPool to create VkCommandBuffers from it.
// Example:
//{
//   CommandPool cmdPool(...);

//  // some setup/one shot work
//  {
//    vkCommandBuffer cmd = scopePool.createAndBegin();
//    ... record commands ...
//    // trigger execution with a blocking operation
//    // not recommended for performance
//    // but useful for sample setup
//    scopePool.submitAndWait(cmd, queue);
//  }

//  // other cmds you may batch, or recycle
//  std::vector<VkCommandBuffer> cmds;
//  {
//    vkCommandBuffer cmd = scopePool.createAndBegin();
//    ... record commands ...
//    cmds.push_back(cmd);
//  }
//  {
//    vkCommandBuffer cmd = scopePool.createAndBegin();
//    ... record commands ...
//    cmds.push_back(cmd);
//  }

//  // do some form of batched submission of cmds

//  // after completion destroy cmd
//  cmdPool.destroy(cmds.size(), cmds.data());
//  cmdPool.deinit();
//}

class CommandPool
{
  public:
    CommandPool() = default;
    // if defaultQueue is null, uses first queue from familyIndex as default
    CommandPool(VkDevice device, uint32_t familyIndex,
                VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                VkQueue defaultQueue = VK_NULL_HANDLE);

    ~CommandPool();

    friend void swap(CommandPool &x, CommandPool &y) noexcept
    {
        using std::swap;

        swap(x.m_device, y.m_device);
        swap(x.m_queue, y.m_queue);
        swap(x.m_commandPool, y.m_commandPool);
    }

    NO_COPY_AND_SWAP_AS_MOVE(CommandPool)

    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, bool begin = true,
                                        VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                        const VkCommandBufferInheritanceInfo *pInheritanceInfo = nullptr);

    // free cmdbuffers from this pool
    void destroy(size_t count, const VkCommandBuffer *cmds);
    void destroy(const std::vector<VkCommandBuffer> &cmds) { destroy(cmds.size(), cmds.data()); }
    void destroy(VkCommandBuffer cmd) { destroy(1, &cmd); }

    VkCommandPool getCommandPool() const { return m_commandPool; }

    // Ends command buffer recording and submits to queue, if 'fence' is not
    // VK_NULL_HANDLE, it will be used to signal the completion of the command
    // buffer execution. Does NOT destroy the command buffers! This is not
    // optimal use for queue submission asity may lead to a large number of
    // vkQueueSubmit() calls per frame. . Consider batching submissions up via
    // FencedCommandPools and BatchedSubmission classes down below.
    void submit(size_t count, const VkCommandBuffer *cmds, VkQueue queue, VkFence fence = VK_NULL_HANDLE);
    void submit(size_t count, const VkCommandBuffer *cmds, VkFence fence = VK_NULL_HANDLE);
    void submit(const std::vector<VkCommandBuffer> &cmds, VkFence fence = VK_NULL_HANDLE);

    // Non-optimal usage pattern using wait for idles, avoid in production use.
    // Consider batching submissions up via FencedCommandPools and
    // BatchedSubmission classes down below. Ends command buffer recording and
    // submits to queue, waits for queue idle and destroys cmds.
    void submitAndWait(size_t count, const VkCommandBuffer *cmds, VkQueue queue);
    void submitAndWait(const std::vector<VkCommandBuffer> &cmds, VkQueue queue)
    {
        submitAndWait(cmds.size(), cmds.data(), queue);
    }
    void submitAndWait(VkCommandBuffer cmd, VkQueue queue) { submitAndWait(1, &cmd, queue); }

    // ends and submits to default queue, waits for queue idle and destroys cmds
    void submitAndWait(size_t count, const VkCommandBuffer *cmds) { submitAndWait(count, cmds, m_queue); }
    void submitAndWait(const std::vector<VkCommandBuffer> &cmds) { submitAndWait(cmds.size(), cmds.data(), m_queue); }
    void submitAndWait(VkCommandBuffer cmd) { submitAndWait(1, &cmd, m_queue); }

  protected:
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
};

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
    ~Swapchain();

    friend void swap(Swapchain &x, Swapchain &y) noexcept
    {
        using std::swap;

        swap(x.physical_device, y.physical_device);
        swap(x.device, y.device);
        swap(x.queue, y.queue);
        swap(x.surface, y.surface);
        swap(x.swapchain, y.swapchain);
        swap(x.format, y.format);
        swap(x.extent, y.extent);
        swap(x.images, y.images);
        swap(x.image_views, y.image_views);
        swap(x.present_complete_semaphores, y.present_complete_semaphores);
        swap(x.render_complete_semaphores, y.render_complete_semaphores);
        swap(x.inflight_fences, y.inflight_fences);
        swap(x.max_frames_ahead, y.max_frames_ahead);
        swap(x.render_ahead_index, y.render_ahead_index);
        swap(x.frame_index, y.frame_index);
    }

    NO_COPY_AND_SWAP_AS_MOVE(Swapchain);

    bool acquire();
    bool submit_and_present(const std::vector<VkCommandBuffer> &cbs);
    void resize(uint32_t width, uint32_t height);

    void create_swapchain_and_images(uint32_t width, uint32_t height);
    void destroy_swapchain_and_images();

    uint32_t frame_count() const { return (uint32_t)image_views.size(); }
    float aspect() const { return (float)extent.width / (float)extent.height; }

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = {};
    VkExtent2D extent = {};
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;

    std::vector<VkSemaphore> present_complete_semaphores;
    std::vector<VkSemaphore> render_complete_semaphores;
    std::vector<VkFence> inflight_fences;
    uint32_t max_frames_ahead = 0;
    uint32_t render_ahead_index = 0;
    uint32_t frame_index = 0;
};

//-----------------------------------------------------------------------------
// [Command buffer management]
//-----------------------------------------------------------------------------

struct CmdBufManager
{
    CmdBufManager() = default;
    CmdBufManager(uint32_t frame_count, uint32_t queue_family_index, VkDevice device);
    ~CmdBufManager();

    friend void swap(CmdBufManager &x, CmdBufManager &y) noexcept
    {
        using std::swap;

        swap(x.frames, y.frames);
        swap(x.device, y.device);
        swap(x.frame_index, y.frame_index);
    }

    NO_COPY_AND_SWAP_AS_MOVE(CmdBufManager)

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
    VkDevice device = VK_NULL_HANDLE;
    uint32_t frame_index = 0;
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
    // Support for unbounded array : an unbounded array must be the last descriptor in the set.
    void add_binding(std::string name, VkDescriptorSetLayoutBinding binding, bool unbounded_array = false);
    VkDescriptorPool create_pool(VkDevice device, uint32_t max_sets) const;
    VkDescriptorSetLayout create_set_layout(VkDevice device) const;
    VkWriteDescriptorSet make_write(VkDescriptorSet dst_set, uint32_t dst_binding) const;
    VkWriteDescriptorSet make_write(VkDescriptorSet dst_set, const std::string &binding_name) const;
    VkWriteDescriptorSet make_write_array(VkDescriptorSet dst_set, uint32_t dst_binding, uint32_t start,
                                          uint32_t count) const;
    VkWriteDescriptorSet make_write_array(VkDescriptorSet dst_set, const std::string &binding_name, uint32_t start,
                                          uint32_t count) const;

    // TODO: ultimately to be figured out automatically via reflection...
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bool last_unbounded_array = false;
    std::unordered_map<std::string, uint32_t> name_map;
};

struct ParameterBlockMeta;

struct ParameterWrite
{
    VkWriteDescriptorSet write{};
    std::unique_ptr<VkDescriptorBufferInfo> buffer_info;
    std::unique_ptr<VkDescriptorImageInfo> image_info;
    std::unique_ptr<VkBufferView> texel_buffer_view;
};

struct ParameterWriteArray
{
    static void make_recurse(ParameterWriteArray &arr) {}

    template <std::same_as<ParameterWrite>... W>
    static void make_recurse(ParameterWriteArray &arr, ParameterWrite &&w, W &&...args)
    {
        arr.writes.emplace_back(std::move(w.write));
        arr.buffer_infos.emplace_back(std::move(w.buffer_info));
        arr.image_infos.emplace_back(std::move(w.image_info));
        arr.texel_buffer_views.emplace_back(std::move(w.texel_buffer_view));

        make_recurse(arr, std::forward<W>(args)...);
    }

    template <std::same_as<ParameterWrite>... W>
    static ParameterWriteArray make(W &&...writes)
    {
        constexpr size_t N = (sizeof(writes) + ... + 0) / sizeof(ParameterWrite);
        ParameterWriteArray arr;
        arr.writes.reserve(N);
        arr.buffer_infos.reserve(N);
        arr.image_infos.reserve(N);
        arr.texel_buffer_views.reserve(N);

        make_recurse(arr, std::forward<W>(writes)...);
        return arr;
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::unique_ptr<VkDescriptorBufferInfo>> buffer_infos;
    std::vector<std::unique_ptr<VkDescriptorImageInfo>> image_infos;
    std::vector<std::unique_ptr<VkBufferView>> texel_buffer_views;
};

struct ParameterBlock
{
    ParameterWrite write(const std::string &binding_name, const std::optional<VkDescriptorBufferInfo> buffer_info = {},
                         const std::optional<VkDescriptorImageInfo> image_info = {},
                         VkBufferView texel_buffer_view = VK_NULL_HANDLE) const;

    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    ParameterBlockMeta *meta = nullptr;
};

struct ParameterBlockMeta
{
    ParameterBlockMeta() = default;
    ParameterBlockMeta(VkDevice device, uint32_t max_sets, DescriptorSetHelper &&helper)
    {
        init(device, max_sets, std::move(helper));
    }
    ~ParameterBlockMeta() { deinit(); }

    void init(VkDevice device, uint32_t max_sets, DescriptorSetHelper &&helper);
    bool is_init() const { return desc_pool != VK_NULL_HANDLE; }
    void deinit();

    friend void swap(ParameterBlockMeta &x, ParameterBlockMeta &y) noexcept
    {
        using std::swap;

        swap(x.desc_set_helper, y.desc_set_helper);
        swap(x.desc_set_layout, y.desc_set_layout);
        swap(x.desc_pool, y.desc_pool);
        swap(x.device, y.device);
        swap(x.max_sets, y.max_sets);
        swap(x.allocated_sets, y.allocated_sets);
    }

    NO_COPY_AND_SWAP_AS_MOVE(ParameterBlockMeta)

    ParameterBlock allocate_block()
    {
        ParameterBlock block;
        allocate_blocks(1, {&block, 1});
        return block;
    }
    void allocate_blocks(uint32_t num, std::span<ParameterBlock> out);

    DescriptorSetHelper desc_set_helper;
    VkDescriptorSetLayout desc_set_layout = VK_NULL_HANDLE;
    // TODO: so far never really need multiple pools...
    // TODO: so far never really need to use vkFreeDescriptorSets()...
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    uint32_t max_sets = 0;
    uint32_t allocated_sets = 0;
};

//-----------------------------------------------------------------------------
// [Other convenience wrappers]
//-----------------------------------------------------------------------------

template <uint32_t DIM_X, uint32_t DIM_Y, uint32_t DIM_Z>
void dispatch_compute(VkCommandBuffer cb, ks::arr3u n_elements)
{
    uint32_t nx = (uint32_t)ceil((float)n_elements.x() / (float)DIM_X);
    uint32_t ny = (uint32_t)ceil((float)n_elements.y() / (float)DIM_Y);
    uint32_t nz = (uint32_t)ceil((float)n_elements.z() / (float)DIM_Z);
    vkCmdDispatch(cb, nx, ny, nz);
}

inline void pipeline_barrier(VkCommandBuffer cb, //
                             VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                             VkDependencyFlags dependency_flags, std::span<const VkMemoryBarrier> memory_barriers,
                             std::span<const VkBufferMemoryBarrier> buffer_barriers,
                             std::span<const VkImageMemoryBarrier> image_barriers)
{
    vkCmdPipelineBarrier(cb, src_stage_mask, dst_stage_mask, dependency_flags, memory_barriers.size(),
                         memory_barriers.data(), buffer_barriers.size(), buffer_barriers.data(), image_barriers.size(),
                         image_barriers.data());
}

// For debug only
inline void full_pipeline_barrier(VkCommandBuffer cb)
{
    VkMemoryBarrier debug_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                         VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                         VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT |
                         VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                         VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                         VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT |
                         VK_ACCESS_HOST_WRITE_BIT};

    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         (VkDependencyFlags)0, 1, &debug_barrier, 0, nullptr, 0, nullptr);
}

//-----------------------------------------------------------------------------
// [Ray tracing facilities]
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

// Convert a Mat4x4 to the matrix required by acceleration structures
inline VkTransformMatrixKHR toTransformMatrixKHR(ks::mat4 matrix)
{
    // VkTransformMatrixKHR uses a row-major memory layout, while Eigen
    // uses a column-major memory layout. We transpose the matrix so we can
    // memcpy the matrix's data directly.
    matrix.transposeInPlace();
    VkTransformMatrixKHR out_matrix;
    memcpy(&out_matrix, &matrix, sizeof(VkTransformMatrixKHR));
    return out_matrix;
}

inline VkTransformMatrixKHR toTransformMatrixKHR(const Transform &transform)
{
    return toTransformMatrixKHR(transform.m);
}

// Single Geometry information, multiple can be used in a single BLAS
struct AccelerationStructureGeometryInfo
{
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
};

// Template for building Vulkan Acceleration Structures of a specified type.
struct AccelerationStructureBuildData
{
    VkAccelerationStructureTypeKHR asType = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR; // Mandatory to set

    // Collection of geometries for the acceleration structure.
    std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
    // Build range information corresponding to each geometry.
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildRangeInfo;
    // Build information required for acceleration structure.
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    // Size information for acceleration structure build resources.
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    // Adds a geometry with its build range information to the acceleration structure.
    void addGeometry(const VkAccelerationStructureGeometryKHR &asGeom,
                     const VkAccelerationStructureBuildRangeInfoKHR &offset);
    void addGeometry(const AccelerationStructureGeometryInfo &asGeom);

    AccelerationStructureGeometryInfo makeInstanceGeometry(size_t numInstances, VkDeviceAddress instanceBufferAddr);

    // Configures the build information and calculates the necessary size information.
    VkAccelerationStructureBuildSizesInfoKHR finalizeGeometry(VkDevice device,
                                                              VkBuildAccelerationStructureFlagsKHR flags);

    // Creates an acceleration structure based on the current build and size info.
    VkAccelerationStructureCreateInfoKHR makeCreateInfo() const;

    // Commands to build the acceleration structure in a Vulkan command buffer.
    void cmdBuildAccelerationStructure(VkCommandBuffer cmd, VkAccelerationStructureKHR accelerationStructure,
                                       VkDeviceAddress scratchAddress);

    // Commands to update the acceleration structure in a Vulkan command buffer.
    void cmdUpdateAccelerationStructure(VkCommandBuffer cmd, VkAccelerationStructureKHR accelerationStructure,
                                        VkDeviceAddress scratchAddress);

    // Checks if the compact flag is set for the build.
    bool hasCompactFlag() const { return buildInfo.flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR; }
};

// Get the maximum scratch buffer size required for the acceleration structure build
VkDeviceSize getMaxScratchSize(const std::vector<AccelerationStructureBuildData> &asBuildData);

/**
 * @brief Manages the construction and optimization of Bottom-Level Acceleration Structures (BLAS) for Vulkan Ray
 * Tracing.
 *
 * This class is designed to  facilitates the  construction of BLAS based on provided build information, queries for
 * compaction potential, compacts BLAS for efficient memory usage, and cleans up resources.
 * It ensures that operations are performed within a specified memory budget, if possible,
 * and provides statistical data on the compaction results. This class is crucial for optimizing ray tracing performance
 * by managing BLAS efficiently in Vulkan-enabled applications.
 *
 * Usage:
 * - Create a BlasBuilder object with a resource allocator and device.
 * - Call cmdCreateBlas or cmdCreateParallelBlas in a loop to create the BLAS from the provided BlasBuildData, until it
 * returns true.
 * - Call cmdCompactBlas to compact the BLAS that have been built.
 * - Call destroyNonCompactedBlas to destroy the original BLAS that were compacted.
 * - Call destroy to clean up all resources.
 * - Call getStatistics to get statistics about the compacted BLAS.
 *
 * For parallel BLAS creation, the scratch buffer size strategy can be used to find the best size needed for the scratch
 * buffer.
 * - Call getScratchSize to get the maximum size needed for the scratch buffer.
 * - User allocate the scratch buffer with the size returned by getScratchSize.
 * - Call getScratchAddresses to get the address for each BLAS.
 * - Use the scratch buffer addresses to build the BLAS in parallel.
 *
 */

class BlasBuilder
{
  public:
    BlasBuilder(Allocator &allocator, VkDevice device);
    ~BlasBuilder();

    struct Stats
    {
        VkDeviceSize totalOriginalSize = 0;
        VkDeviceSize totalCompactSize = 0;

        std::string toString() const;
    };

    // Create the BLAS from the vector of BlasBuildData
    // Each BLAS will be created in sequence and share the same scratch buffer
    // Return true if ALL the BLAS were created within the budget
    // if not, this function needs to be called again until it returns true
    bool cmdCreateBlas(VkCommandBuffer cmd,
                       std::vector<AccelerationStructureBuildData> &blasBuildData, // List of the BLAS to build */
                       std::vector<AccelKHR> &blasAccel,                           // List of the acceleration structure
                       VkDeviceAddress scratchAddress,                             //  Address of the scratch buffer
                       VkDeviceSize hintMaxBudget = 512'000'000);

    // Create the BLAS from the vector of BlasBuildData in parallel
    // The advantage of this function is that it will try to build as many BLAS as possible in parallel
    // but it requires a scratch buffer per BLAS, or less but then each of them must large enough to hold the largest
    // BLAS This function needs to be called until it returns true
    bool cmdCreateParallelBlas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                               std::vector<AccelKHR> &blasAccel, const std::vector<VkDeviceAddress> &scratchAddress,
                               VkDeviceSize hintMaxBudget = 512'000'000);

    // Compact the BLAS that have been built
    // Synchronization must be done by the application between the build and the compact
    void cmdCompactBlas(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                        std::vector<AccelKHR> &blasAccel);

    // Destroy the original BLAS that was compacted
    void destroyNonCompactedBlas();

    // Return the statistics about the compacted BLAS
    Stats getStatistics() const { return m_stats; };

    // Scratch size strategy:
    // Find the maximum size of the scratch buffer needed for the BLAS build
    //
    // Strategy:
    // - Loop over all BLAS to find the maximum size
    // - If the max size is within the budget, return it. This will return as many addresses as there are BLAS
    // - Otherwise, return n*maxBlasSize, where n is the number of BLAS and maxBlasSize is the maximum size found for a
    // single BLAS.
    //   In this case, fewer addresses will be returned than the number of BLAS, but each can be used to build any BLAS
    //
    // Usage
    // - Call this function to get the maximum size needed for the scratch buffer
    // - User allocate the scratch buffer with the size returned by this function
    // - Call getScratchAddresses to get the address for each BLAS
    //
    // Note: 128 is the default alignment for the scratch buffer
    //       (VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment)
    VkDeviceSize getScratchSize(VkDeviceSize hintMaxBudget,
                                const std::vector<AccelerationStructureBuildData> &buildData,
                                uint32_t minAlignment = 128) const;

    void getScratchAddresses(VkDeviceSize hintMaxBudget, const std::vector<AccelerationStructureBuildData> &buildData,
                             VkDeviceAddress scratchBufferAddress, std::vector<VkDeviceAddress> &scratchAddresses,
                             uint32_t minAlignment = 128);

  private:
    void destroyQueryPool();
    void createQueryPool(uint32_t maxBlasCount);
    void initializeQueryPoolIfNeeded(const std::vector<AccelerationStructureBuildData> &blasBuildData);
    VkDeviceSize
    buildAccelerationStructures(VkCommandBuffer cmd, std::vector<AccelerationStructureBuildData> &blasBuildData,
                                std::vector<AccelKHR> &blasAccel, const std::vector<VkDeviceAddress> &scratchAddress,
                                VkDeviceSize hintMaxBudget, VkDeviceSize currentBudget, uint32_t &currentQueryIdx);

    VkDevice m_device;
    Allocator *m_alloc = nullptr;
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    uint32_t m_currentBlasIdx{0};
    uint32_t m_currentQueryIdx{0};

    std::vector<AccelKHR> m_cleanupBlasAccel;

    // Stats
    Stats m_stats;
};

// Ray tracing BLAS and TLAS builder
struct RaytracingBuilderKHR
{
    RaytracingBuilderKHR(const VkDevice &device, Allocator &allocator, uint32_t queueIndex);

    ~RaytracingBuilderKHR();

    // Inputs used to build Bottom-level acceleration structure.
    // You manage the lifetime of the buffer(s) referenced by the VkAccelerationStructureGeometryKHRs within.
    // In particular, you must make sure they are still valid and not being modified when the BLAS is built or updated.
    struct BlasInput
    {
        // Data used to build acceleration structure geometry
        std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
        VkBuildAccelerationStructureFlagsKHR flags{0};
    };

    // Returning the constructed top-level acceleration structure
    VkAccelerationStructureKHR getAccelerationStructure() const;

    // Return the Acceleration Structure Device Address of a BLAS Id
    VkDeviceAddress getBlasDeviceAddress(uint32_t blasId);

    // Create all the BLAS from the vector of BlasInput
    void
    buildBlas(const std::vector<BlasInput> &input,
              VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Refit BLAS number blasIdx from updated buffer contents.
    void updateBlas(uint32_t blasIdx, BlasInput &blas, VkBuildAccelerationStructureFlagsKHR flags);

    // Build TLAS for static acceleration structures
    void
    buildTlas(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
              VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
              bool update = false);

#ifdef VK_NV_ray_tracing_motion_blur
    // Build TLAS for mix of motion and static acceleration structures
    void buildTlas(const std::vector<VkAccelerationStructureMotionInstanceNV> &instances,
                   VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV,
                   bool update = false);
#endif

    // Build TLAS from an array of VkAccelerationStructureInstanceKHR
    // - Use motion=true with VkAccelerationStructureMotionInstanceNV
    // - The resulting TLAS will be stored in m_tlas
    // - update is to rebuild the Tlas with updated matrices, flag must have the 'allow_update'
    template <class T>
    void
    buildTlas(const std::vector<T> &instances,
              VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
              bool update = false, bool motion = false)
    {
        // Cannot call buildTlas twice except to update.
        ASSERT(m_tlas.accel == VK_NULL_HANDLE || update);
        uint32_t countInstance = static_cast<uint32_t>(instances.size());

        // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
        // This uses a separate command pool/buffer. Consider consolidating?
        Buffer instancesBuffer; // Buffer of instances containing the matrices and BLAS ids
        m_alloc->stage_session([&](Allocator &alloc) {
            instancesBuffer = alloc.create_buffer(
                VkBufferCreateInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .size = sizeof(T) * instances.size(),
                                   .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR},
                VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)0,
                reinterpret_cast<const std::byte *>(instances.data()));
        });
        NAME_VK(instancesBuffer.buffer);

        // Command buffer to create the TLAS
        CommandPool genCmdBuf(m_device, m_queueIndex);
        VkCommandBuffer cmd = genCmdBuf.createCommandBuffer();

        VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                             instancesBuffer.buffer};
        VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(m_device, &bufferInfo);

        // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0,
                             nullptr);

        // Creating the TLAS
        Buffer scratchBuffer;
        cmdCreateTlas(cmd, countInstance, instBufferAddr, scratchBuffer, flags, update, motion);

        // Finalizing and destroying temporary data
        genCmdBuf.submitAndWait(cmd); // queueWaitIdle inside.
        m_alloc->destroy(scratchBuffer);
        m_alloc->destroy(instancesBuffer);
    }

    // Creating the TLAS, called by buildTlas
    void cmdCreateTlas(VkCommandBuffer cmdBuf,                     // Command buffer
                       uint32_t countInstance,                     // number of instances
                       VkDeviceAddress instBufferAddr,             // Buffer address of instances
                       Buffer &scratchBuffer,                      // Scratch buffer for construction
                       VkBuildAccelerationStructureFlagsKHR flags, // Build creation flag
                       bool update,                                // Update == animation
                       bool motion                                 // Motion Blur
    );

  private:
    std::vector<AccelKHR> m_blas; // Bottom-level acceleration structure
    AccelKHR m_tlas;              // Top-level acceleration structure

    // Setup
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_queueIndex = 0;
    Allocator *m_alloc = nullptr;
    DebugUtil m_debug;

    bool hasFlag(VkFlags item, VkFlags flag) { return (item & flag) == flag; }
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
    explicit GFX(const GFXArgs &args);
    ~GFX();

    friend void swap(GFX &x, GFX &y) noexcept
    {
        using std::swap;

        swap(x.swapchain, y.swapchain);
        swap(x.cb_manager, y.cb_manager);
        swap(x.surface, y.surface);
        swap(x.ctx, y.ctx);
    }

    NO_COPY_AND_SWAP_AS_MOVE(GFX)

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

    struct SurfaceWrapper
    {
        SurfaceWrapper() = default;

        ~SurfaceWrapper()
        {
            if (instance != VK_NULL_HANDLE) {
                vkDestroySurfaceKHR(instance, surface, nullptr);
            }
        }

        friend void swap(SurfaceWrapper &x, SurfaceWrapper &y)
        {
            using std::swap;
            swap(x.surface, y.surface);
            swap(x.instance, y.instance);
        }

        NO_COPY_AND_SWAP_AS_MOVE(SurfaceWrapper)

        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkInstance instance = VK_NULL_HANDLE;
    };

    // Note the destruction order...
    Context ctx;
    SurfaceWrapper surface;
    Swapchain swapchain;
    CmdBufManager cb_manager;
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
    ~GUI();

    friend void swap(GUI &x, GUI &y) noexcept
    {
        using std::swap;

        swap(x.window, y.window);
        swap(x.gfx, y.gfx);
        swap(x.pool, y.pool);
        swap(x.render_pass, y.render_pass);
        swap(x.framebuffers, y.framebuffers);
        swap(x.update_fn, y.update_fn);
        swap(x.show, y.show);
    }

    NO_COPY_AND_SWAP_AS_MOVE(GUI)

    void render(VkCommandBuffer cmdBuf);

    void resize();
    void create_render_pass();
    void destroy_render_pass();
    void create_framebuffers();
    void destroy_framebuffers();

    void upload_frame();

    GLFWwindow *window = nullptr;
    const GFX *gfx = nullptr;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    std::function<void()> update_fn;
    bool show = true;
};

} // namespace vk
} // namespace ks