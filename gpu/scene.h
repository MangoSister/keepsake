#pragma once

#include "../maths.h"
#include "../mesh_asset.h"
#include "ksvk.h"

namespace ks
{

namespace host_device
{
// TODO: These structs/constants need to be kept consistent between CPU and GPU. Use Slang reflection later?
struct GPUMeshDataAddresses
{
    uint64_t vertices_address;
    uint64_t texcoords_address;
    uint64_t vertex_normals_address;
    uint64_t indices_address;
};

struct GPUMaterialPrincipledBSDFDataBlock
{
    vec4 basecolor_uv_scale_offset;
    vec4 emissive_uv_scale_offset;
    vec4 roughness_uv_scale_offset;
    vec4 metallic_uv_scale_offset;
    vec4 ior_uv_scale_offset;
    vec4 specular_r0_mul_uv_scale_offset;
    vec4 specular_trans_uv_scale_offset;
    vec4 diffuse_trans_uv_scale_offset;
    vec4 diffuse_trans_fwd_uv_scale_offset;
    vec4 normal_uv_scale_offset;
    vec4 opacity_uv_scale_offset;
    // 44 * sizeof(uint32_t)

    vec3 basecolor_constant_or_scale;
    uint32_t basecolor_map_id;
    // 48 * sizeof(uint32_t)

    vec3 emissive_constant_or_scale;
    uint32_t emissive_map_id;
    // 52 * sizeof(uint32_t)

    float roughness_constant_or_scale;
    uint32_t roughness_map_id;
    float metallic_constant_or_scale;
    uint32_t metallic_map_id;
    float ior_constant_or_scale;
    uint32_t ior_map_id;
    float specular_r0_mul_constant_or_scale;
    uint32_t specular_r0_mul_map_id;
    float specular_trans_constant_or_scale;
    uint32_t specular_trans_map_id;
    float diffuse_trans_constant_or_scale;
    uint32_t diffuse_trans_map_id;
    float diffuse_trans_fwd_constant_or_scale;
    uint32_t diffuse_trans_fwd_map_id;
    float normal_strength;
    uint32_t normal_map_id;
    float opacity_constant_or_scale;
    uint32_t opacity_map_id;
    // 70 * sizeof(uint32_t)

    // padding
    uint32_t padding[2];
    // 72 * sizeof(uint32_t)
};

constexpr size_t gpu_material_data_block_size = sizeof(GPUMaterialPrincipledBSDFDataBlock);
constexpr uint32_t gpu_material_empty_map_id = (uint32_t)(~0);

enum class SceneBindings : uint32_t
{
    TLAS = 0,
    SubSceneOffsets = 1,
    MeshDataAddresses = 2,
    MaterialIndices = 3,
    MaterialBlocks = 4,
    MaterialTextures2D = 5
};

} // namespace host_device

// Corresponds to one geometry in the BLAS.
struct GPUMeshData
{
    // buffer needs the following bits for ray tracing
    // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    // VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    vk::AutoRelease<vk::Buffer> vertices;
    vk::AutoRelease<vk::Buffer> texcoords;
    vk::AutoRelease<vk::Buffer> vertex_normals;
    vk::AutoRelease<vk::Buffer> indices;
    uint32_t vertex_count;
    uint32_t tri_count;
};

// We have one material per "geometry"
// Index material by (subscene index, geom index) -> material index.
struct GPUMaterial
{
    // TODO
    // interface with CPU material and
    // determine binding (descriptor?) for both raster and ray tracing

    // TODO: Use Slang for polymorphism material system (reintepret, interface).

    virtual ~GPUMaterial() = default;

    virtual void emit_data_block(std::span<std::byte, ks::host_device::gpu_material_data_block_size> dest) const = 0;
    virtual bool allow_any_hit() const = 0;
};

template <int N>
struct GPUMaterialPrincipledBSDFField;

// Corresponds to one BLAS.
struct GPUSubScene
{
    vk::RaytracingBuilderKHR::BlasInput generate_blas_input(VkDevice device);

    std::vector<std::unique_ptr<GPUMeshData>> mesh_datas;
    std::vector<std::unique_ptr<GPUMaterial>> materials;
};

// Corresponds to one instance in the TLAS.
struct GPUSubSceneInstance
{
    uint32_t prototype = 0;
    Transform transform;
};

// Corresponds to one TLAS.
struct GPUScene
{
  public:
    GPUScene(const CompoundMeshAsset &compound, const vk::Context &ctx);

    void prepare_for_raster();
    void prepare_for_ray_tracing();

    VkDescriptorSetLayout get_ray_tracing_desc_set_layout() const { return material_param_block_meta.desc_set_layout; }
    VkDescriptorSet get_ray_tracing_desc_set() const { return material_param_block.desc_set; }

  private:
    std::unique_ptr<GPUMaterial>
    convert_gpu_material(const Material &cpu_material,
                         const std::unordered_map<const Texture *, uint32_t> &texture_id_map);
    template <int N>
    void convert_shader_field(const ShaderField<color<N>> &cpu_field, GPUMaterialPrincipledBSDFField<N> &gpu_field,
                              const std::unordered_map<const Texture *, uint32_t> &texture_id_map);

    void build_accel();
    // allocate gpu material data block buffer
    // fill GPU material data blocks
    void build_materials();

    std::vector<GPUSubScene> subscenes;
    std::vector<GPUSubSceneInstance> instances;
    vk::RaytracingBuilderKHR rt_builder;

    vk::ParameterBlockMeta material_param_block_meta;
    vk::ParameterBlock material_param_block;

    // per-geometry array indexing: array[subscene_offsets[gl_InstanceCustomIndexEXT] + gl_GeometryIndexEXT]
    vk::AutoRelease<vk::Buffer> subscene_offsets_buf;
    vk::AutoRelease<vk::Buffer> mesh_data_addresses_buf; // per geometry
    vk::AutoRelease<vk::Buffer> material_indices_buf;    // per geometry
    vk::AutoRelease<vk::Buffer> material_blocks_buf;     // per material

    // one big descriptor set for the entire scene?
    // material data
    // big array of each type of resources

    std::vector<vk::AutoRelease<vk::Image>> material_textures_2d;
    std::vector<VkImageCreateInfo> material_textures_2d_ci;
    // It's common to have one multiple views for a texture (roughness/metallic channels).
    vk::ImageViewCache material_textures_2d_view_cache;
    vk::SamplerCache sampler_cache;
    // (image_view_id, sampler_id) -> (combined_id)
    ByteOpHashTable<std::pair<uint32_t, uint32_t>> material_texture_2d_combined_sampler_map;

    VkDevice device = VK_NULL_HANDLE;
    std::shared_ptr<vk::Allocator> allocator;
};

} // namespace ks