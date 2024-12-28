#include "scene.h"

#include "../log_util.h"
#include "../normal_map.h"
#include "../opacity_map.h"
#include "../principled_bsdf.h"

namespace ks
{

// GPU scene infrastructure
// scene need to support both ray tracing (more important) and raster (less important).
// need to support (single-level) instancing

// 1. load a gltf scene
// 2. transfer cpu data to gpu
// 3. build acceleration structures
// 4. build material data
// 5. application specific stuff: pipeline, SBT, etc
// 6. command recording

// GPU small pt

// rendering features: BSDF, light sources, etc
// need to support opacity map
// support micromap??

// GUI integration

// reflection for binding, uniform types, etc?

template <int N>
struct GPUMaterialPrincipledBSDFField
{
    color<N> constant_or_scale = color<N>::Zero();
    uint32_t map_id = ks::host_device::gpu_material_empty_map_id;
    vec2 uv_scale = vec2::Ones();
    vec2 uv_offset = vec2::Zero();
};

struct GPUMaterialPrincipledBSDF : GPUMaterial
{
    void emit_data_block(std::span<std::byte, ks::host_device::gpu_material_data_block_size> dest) const
    {
        using ks::host_device::GPUMaterialPrincipledBSDFDataBlock;

        GPUMaterialPrincipledBSDFDataBlock *block = reinterpret_cast<GPUMaterialPrincipledBSDFDataBlock *>(dest.data());
        block->basecolor_constant_or_scale = basecolor.constant_or_scale;
        block->basecolor_map_id = basecolor.map_id;
        block->basecolor_uv_scale_offset =
            vec4(basecolor.uv_scale[0], basecolor.uv_scale[1], basecolor.uv_offset[0], basecolor.uv_offset[1]);

        block->roughness_constant_or_scale = roughness.constant_or_scale[0];
        block->roughness_map_id = roughness.map_id;
        block->roughness_uv_scale_offset =
            vec4(roughness.uv_scale[0], roughness.uv_scale[1], roughness.uv_offset[0], roughness.uv_offset[1]);

        block->metallic_constant_or_scale = metallic.constant_or_scale[0];
        block->metallic_map_id = metallic.map_id;
        block->metallic_uv_scale_offset =
            vec4(metallic.uv_scale[0], metallic.uv_scale[1], metallic.uv_offset[0], metallic.uv_offset[1]);

        block->ior_constant_or_scale = ior.constant_or_scale[0];
        block->ior_map_id = ior.map_id;
        block->ior_uv_scale_offset = vec4(ior.uv_scale[0], ior.uv_scale[1], ior.uv_offset[0], ior.uv_offset[1]);

        block->specular_r0_mul_constant_or_scale = specular_r0_mul.constant_or_scale[0];
        block->specular_r0_mul_map_id = specular_r0_mul.map_id;
        block->specular_r0_mul_uv_scale_offset = vec4(specular_r0_mul.uv_scale[0], specular_r0_mul.uv_scale[1],
                                                      specular_r0_mul.uv_offset[0], specular_r0_mul.uv_offset[1]);

        block->specular_trans_constant_or_scale = specular_trans.constant_or_scale[0];
        block->specular_trans_map_id = specular_trans.map_id;
        block->specular_trans_uv_scale_offset = vec4(specular_trans.uv_scale[0], specular_trans.uv_scale[1],
                                                     specular_trans.uv_offset[0], specular_trans.uv_offset[1]);

        block->diffuse_trans_constant_or_scale = diffuse_trans.constant_or_scale[0];
        block->diffuse_trans_map_id = diffuse_trans.map_id;
        block->diffuse_trans_uv_scale_offset = vec4(diffuse_trans.uv_scale[0], diffuse_trans.uv_scale[1],
                                                    diffuse_trans.uv_offset[0], diffuse_trans.uv_offset[1]);

        block->diffuse_trans_fwd_constant_or_scale = diffuse_trans_fwd.constant_or_scale[0];
        block->diffuse_trans_fwd_map_id = diffuse_trans_fwd.map_id;
        block->diffuse_trans_fwd_uv_scale_offset = vec4(diffuse_trans_fwd.uv_scale[0], diffuse_trans_fwd.uv_scale[1],
                                                        diffuse_trans_fwd.uv_offset[0], diffuse_trans_fwd.uv_offset[1]);

        block->emissive_constant_or_scale = emissive.constant_or_scale;
        block->emissive_map_id = emissive.map_id;
        block->emissive_uv_scale_offset =
            vec4(emissive.uv_scale[0], emissive.uv_scale[1], emissive.uv_offset[0], emissive.uv_offset[1]);

        block->normal_strength = normal.constant_or_scale[0];
        block->normal_map_id = normal.map_id;
        block->normal_uv_scale_offset =
            vec4(normal.uv_scale[0], normal.uv_scale[1], normal.uv_offset[0], normal.uv_offset[1]);

        block->opacity_constant_or_scale = opacity.constant_or_scale[0];
        block->opacity_map_id = opacity.map_id;
        block->opacity_uv_scale_offset =
            vec4(opacity.uv_scale[0], opacity.uv_scale[1], opacity.uv_offset[0], opacity.uv_offset[1]);
    }

    bool allow_any_hit() const
    {
        return opacity.constant_or_scale[0] < 1.0f || opacity.map_id != ks::host_device::gpu_material_empty_map_id;
    }

    GPUMaterialPrincipledBSDFField<3> basecolor;
    GPUMaterialPrincipledBSDFField<1> roughness;
    GPUMaterialPrincipledBSDFField<1> metallic;
    GPUMaterialPrincipledBSDFField<1> ior;
    GPUMaterialPrincipledBSDFField<1> specular_r0_mul;
    GPUMaterialPrincipledBSDFField<1> specular_trans;
    GPUMaterialPrincipledBSDFField<1> diffuse_trans;
    GPUMaterialPrincipledBSDFField<1> diffuse_trans_fwd;
    GPUMaterialPrincipledBSDFField<3> emissive;
    // normal map is 3 channel, but strength is only a scalar
    GPUMaterialPrincipledBSDFField<3> normal;
    GPUMaterialPrincipledBSDFField<1> opacity;
};

vk::RaytracingBuilderKHR::BlasInput GPUSubScene::generate_blas_input(VkDevice device)
{
    vk::RaytracingBuilderKHR::BlasInput blas_input;

    for (uint32_t geom_id = 0; geom_id < (uint32_t)mesh_datas.size(); ++geom_id) {
        VkDeviceAddress vertex_buffer_addr = vk::getBufferDeviceAddress(device, mesh_datas[geom_id]->vertices->buffer);
        VkDeviceAddress index_buffer_addr = vk::getBufferDeviceAddress(device, mesh_datas[geom_id]->indices->buffer);

        // Describe buffer as array of VertexObj.
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // vec3 vertex position data.
        triangles.vertexData.deviceAddress = vertex_buffer_addr;
        triangles.vertexStride = sizeof(float[3]);
        // Describe index data (32-bit unsigned int)
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = index_buffer_addr;
        // Indicate identity transform by setting transformData to null device pointer.
        triangles.transformData = {};
        triangles.maxVertex = mesh_datas[geom_id]->vertex_count - 1;

        // Identify the above data as containing opaque triangles.
        VkAccelerationStructureGeometryKHR geometry{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .flags = materials[geom_id]->allow_any_hit() ? (VkFlags)0 : VK_GEOMETRY_OPAQUE_BIT_KHR,
        };
        geometry.geometry.triangles = triangles;

        // The entire array will be used to build the BLAS.
        VkAccelerationStructureBuildRangeInfoKHR offset{
            .primitiveCount = mesh_datas[geom_id]->tri_count,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0,
        };

        blas_input.geometry.emplace_back(geometry);
        blas_input.range_info.emplace_back(offset);
    }

    return blas_input;
}

template <int N>
void GPUScene::convert_shader_field(const ShaderField<color<N>> &cpu_field,
                                    GPUMaterialPrincipledBSDFField<N> &gpu_field,
                                    const std::unordered_map<const Texture *, uint32_t> &texture_id_map)
{
    if (const TextureField<N> *f = dynamic_cast<const TextureField<N> *>(&cpu_field); f) {
        uint32_t texture_id = texture_id_map.at(f->texture);

        const auto &textures_ci = material_textures_2d_ci;
        const auto &textures = material_textures_2d;
        auto &view_cache = material_textures_2d_view_cache;
        auto &combined_map = material_texture_2d_combined_sampler_map;

        VkImageViewCreateInfo view_info =
            vk::simple_view_info_from_image_info(textures_ci[texture_id], *textures[texture_id], false);

        VkComponentSwizzle *swizzle = reinterpret_cast<VkComponentSwizzle *>(&view_info.components);
        for (int i = 0; i < N; ++i) {
            switch (f->swizzle[i]) {
            case 0:
                swizzle[i] = VK_COMPONENT_SWIZZLE_R;
                break;
            case 1:
                swizzle[i] = VK_COMPONENT_SWIZZLE_G;
                break;
            case 2:
                swizzle[i] = VK_COMPONENT_SWIZZLE_B;
                break;
            case 3:
                swizzle[i] = VK_COMPONENT_SWIZZLE_A;
                break;
            default:
                ASSERT(false);
                break;
            };
        }
        uint32_t view_id;
        view_cache.get_or_create(view_info, &view_id);

        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        };

        VkSamplerAddressMode address_mode_u, address_mode_v;
        switch (f->sampler->wrap_mode_u) {
        case TextureWrapMode::Repeat:
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            break;
        case TextureWrapMode::Clamp:
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            break;
        default:
            get_default_logger().critical("Invalid sampler address mode!");
            std::abort();
            break;
        }

        switch (f->sampler->wrap_mode_v) {
        case TextureWrapMode::Repeat:
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            break;
        case TextureWrapMode::Clamp:
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            break;
        default:
            get_default_logger().critical("Invalid sampler address mode!");
            std::abort();
            break;
        }

        sampler_info.anisotropyEnable = false;
        sampler_info.maxAnisotropy = 1.0f;
        if (const auto s = dynamic_cast<const NearestSampler *>(f->sampler.get()); s) {
            sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_NEAREST;
        } else if (const auto s = dynamic_cast<const LinearSampler *>(f->sampler.get()); s) {
            sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_LINEAR;
        } else if (const auto s = dynamic_cast<const EWASampler *>(f->sampler.get()); s) {
            sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.anisotropyEnable = true;
            sampler_info.maxAnisotropy = 16.0f;
        } else {
            // TODO: cubic?
            get_default_logger().critical("Invalid sampler filter type!");
            std::abort();
        }

        uint32_t sampler_id;
        sampler_cache.get_or_create(sampler_info, &sampler_id);

        uint32_t combined_id;
        auto it = combined_map.find({view_id, sampler_id});
        if (it == combined_map.end()) {
            combined_id = (uint32_t)combined_map.size();
            combined_map.insert({{view_id, sampler_id}, combined_id});
        } else {
            combined_id = it->second;
        }

        gpu_field.map_id = combined_id;
        gpu_field.constant_or_scale = f->scale;
        gpu_field.uv_scale = f->uv_scale;
        gpu_field.uv_offset = f->uv_offset;

    } else if (const ConstantField<color<N>> *f = dynamic_cast<const ConstantField<color<N>> *>(&cpu_field); f) {
        gpu_field.constant_or_scale = f->value;
    } else {
        get_default_logger().critical("GPU Material now only supports constant and texture fields!");
        std::abort();
    }
}

std::unique_ptr<GPUMaterial>
GPUScene::convert_gpu_material(const Material &cpu_material,
                               const std::unordered_map<const Texture *, uint32_t> &texture_id_map)
{
    const PrincipledBSDF *principled = dynamic_cast<const PrincipledBSDF *>(cpu_material.bsdf);
    if (!principled) {
        get_default_logger().critical("GPU Material now only supports PrincipledBSDF!");
        std::abort();
    }
    std::unique_ptr<GPUMaterialPrincipledBSDF> gpu_material = std::make_unique<GPUMaterialPrincipledBSDF>();

    convert_shader_field<3>(*principled->basecolor, gpu_material->basecolor, texture_id_map);
    convert_shader_field<1>(*principled->roughness, gpu_material->roughness, texture_id_map);
    convert_shader_field<1>(*principled->metallic, gpu_material->metallic, texture_id_map);
    convert_shader_field<1>(*principled->ior, gpu_material->ior, texture_id_map);
    convert_shader_field<1>(*principled->specular_r0_mul, gpu_material->specular_r0_mul, texture_id_map);
    convert_shader_field<1>(*principled->specular_trans, gpu_material->specular_trans, texture_id_map);
    convert_shader_field<1>(*principled->diffuse_trans, gpu_material->diffuse_trans, texture_id_map);
    convert_shader_field<1>(*principled->diffuse_trans_fwd, gpu_material->diffuse_trans_fwd, texture_id_map);
    convert_shader_field<3>(*principled->emissive, gpu_material->emissive, texture_id_map);

    if (cpu_material.normal_map) {
        convert_shader_field<3>(*cpu_material.normal_map->map, gpu_material->normal, texture_id_map);
        if (gpu_material->normal.map_id != ks::host_device::gpu_material_empty_map_id) {
            // Normal map strength is stored differently.
            gpu_material->normal.constant_or_scale = cpu_material.normal_map->strength;
        }
    } else {
        // Default to tangent space z-up vector.
        gpu_material->normal.constant_or_scale = color3(0.0f, 0.0f, 1.0f);
    }
    if (cpu_material.opacity_map) {
        convert_shader_field<1>(*cpu_material.opacity_map->map, gpu_material->opacity, texture_id_map);
    } else {
        gpu_material->opacity.constant_or_scale = 1.0f;
    }

    return gpu_material;
}

VkFormat convert_texture_format(TextureDataType data_type, ColorSpace color_space, int num_channels)
{
    if (data_type == TextureDataType::u8) {
        if (color_space == ColorSpace::sRGB) {
            if (num_channels == 1)
                return VK_FORMAT_R8_SRGB;
            else if (num_channels == 2)
                return VK_FORMAT_R8G8_SRGB;
            else if (num_channels == 3)
                return VK_FORMAT_R8G8B8_SRGB;
            else if (num_channels == 4)
                return VK_FORMAT_R8G8B8A8_SRGB;
            else {
                get_default_logger().critical("Unsupported texture num channels!");
                std::abort();
            }
        } else {
            if (num_channels == 1)
                return VK_FORMAT_R8_UNORM;
            else if (num_channels == 2)
                return VK_FORMAT_R8G8_UNORM;
            else if (num_channels == 3)
                return VK_FORMAT_R8G8B8_UNORM;
            else if (num_channels == 4)
                return VK_FORMAT_R8G8B8A8_UNORM;
            else {
                get_default_logger().critical("Unsupported texture num channels!");
                std::abort();
            }
        }
    } else if (data_type == TextureDataType::u16) {
        if (color_space == ColorSpace::sRGB) {
            // TODO
            get_default_logger().critical("Unsupported sRGB format!");
            std::abort();
        } else {
            if (num_channels == 1)
                return VK_FORMAT_R16_UNORM;
            else if (num_channels == 2)
                return VK_FORMAT_R16G16_UNORM;
            else if (num_channels == 3)
                return VK_FORMAT_R16G16B16_UNORM;
            else if (num_channels == 4)
                return VK_FORMAT_R16G16B16A16_UNORM;
            else {
                get_default_logger().critical("Unsupported texture num channels!");
                std::abort();
            }
        }
    } else if (data_type == TextureDataType::f32) {
        if (color_space == ColorSpace::sRGB) {
            // TODO
            get_default_logger().critical("Unsupported sRGB format!");
            std::abort();
        } else {
            if (num_channels == 1)
                return VK_FORMAT_R32_SFLOAT;
            else if (num_channels == 2)
                return VK_FORMAT_R32G32_SFLOAT;
            else if (num_channels == 3)
                return VK_FORMAT_R32G32B32_SFLOAT;
            else if (num_channels == 4)
                return VK_FORMAT_R32G32B32A32_SFLOAT;
            else {
                get_default_logger().critical("Unsupported texture num channels!");
                std::abort();
            }
        }
    } else {
        get_default_logger().critical("Unsupported texture data type!");
        std::abort();
    }
}

GPUScene::GPUScene(const CompoundMeshAsset &compound, const vk::Context &ctx)
    : rt_builder(ctx.device, *ctx.allocator, ctx.main_queue_family_index), sampler_cache(ctx.device),
      material_textures_2d_view_cache(ctx.device), device(ctx.device), allocator(ctx.allocator)
{
    std::unordered_map<const Texture *, uint32_t> texture_id_map;
    // Upload textures (batch to avoid peak memory consumption?)
    ctx.allocator->stage_session([&](vk::Allocator &self) {
        for (uint32_t texture_id = 0; texture_id < compound.textures.size(); ++texture_id) {
            const Texture &cpu_texture = *compound.textures[texture_id];
            auto copy_fn = [&](std::byte *dest) { cpu_texture.mips[0].copy_to_linear_array(dest); };

            VkImageCreateInfo image_ci{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VkImageType::VK_IMAGE_TYPE_2D,
                .format =
                    convert_texture_format(cpu_texture.data_type, cpu_texture.color_space, cpu_texture.num_channels),
                .extent = VkExtent3D(cpu_texture.width, cpu_texture.height, 1),
                .mipLevels = (uint32_t)cpu_texture.mips.size(),
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            };
            vk::Image image = ctx.allocator->create_and_upload_image(
                image_ci, VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), copy_fn,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, vk::MipmapOption::AutoGenerate, false);

            material_textures_2d.emplace_back(std::move(image), ctx.allocator);
            material_textures_2d_ci.push_back(image_ci);
            texture_id_map.insert({compound.textures[texture_id].get(), texture_id});
        }
    });

    for (const auto &p : compound.prototypes) {
        GPUSubScene subscene;
        subscene.mesh_datas.reserve(p.meshes.size());
        // Upload mesh data (batch to avoid peak memory consumption?)
        ctx.allocator->stage_session([&](vk::Allocator &self) {
            constexpr VkBufferUsageFlags vb_flags =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

            constexpr VkBufferUsageFlags ib_flags =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

            for (const auto &m : p.meshes) {
                std::unique_ptr<GPUMeshData> mesh_data = std::make_unique<GPUMeshData>();
                mesh_data->vertex_count = m->vertex_count();
                mesh_data->tri_count = m->tri_count();

                mesh_data->vertices =
                    vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                                VkBufferCreateInfo{
                                                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                    .size = m->vertex_count() * sizeof(float[3]),
                                                    .usage = vb_flags,
                                                },
                                                VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), m->vertices);
                if (m->has_texcoord()) {
                    mesh_data->texcoords =
                        vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                                    VkBufferCreateInfo{
                                                        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                        .size = m->vertex_count() * sizeof(float[2]),
                                                        .usage = vb_flags,
                                                    },
                                                    VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), m->texcoords);
                }

                if (m->has_vertex_normal()) {
                    mesh_data->vertex_normals = vk::AutoRelease<vk::Buffer>(
                        ctx.allocator,
                        VkBufferCreateInfo{
                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = m->vertex_count() * sizeof(float[3]),
                            .usage = vb_flags,
                        },
                        VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), m->vertex_normals);
                }

                mesh_data->indices =
                    vk::AutoRelease<vk::Buffer>(ctx.allocator,
                                                VkBufferCreateInfo{
                                                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                    .size = m->tri_count() * sizeof(uint32_t[3]),
                                                    .usage = ib_flags,
                                                },
                                                VMA_MEMORY_USAGE_AUTO, VmaAllocationCreateFlags(0), m->indices);

                subscene.mesh_datas.push_back(std::move(mesh_data));
            }
        });

        subscene.materials.reserve(p.materials.size());
        for (const auto &m : p.materials) {
            subscene.materials.push_back(convert_gpu_material(*m, texture_id_map));
        }

        subscenes.push_back(std::move(subscene));
    }

    std::vector<uint32_t> subscene_offsets(subscenes.size());
    std::vector<ks::host_device::GPUMeshDataAddresses> mesh_data_addresses;
    for (uint32_t subscene_id = 0; subscene_id < (uint32_t)subscenes.size(); ++subscene_id) {
        uint32_t n_geom = (uint32_t)subscenes[subscene_id].mesh_datas.size();
        if (subscene_id == 0) {
            subscene_offsets[subscene_id] = 0;
        } else {
            subscene_offsets[subscene_id] = subscene_offsets[subscene_id - 1] + n_geom;
        }
        for (uint32_t geom_id = 0; geom_id < n_geom; ++geom_id) {
            const GPUMeshData &mesh_data = *subscenes[subscene_id].mesh_datas[geom_id];
            ks::host_device::GPUMeshDataAddresses addr;
            addr.vertices_address = vk::getBufferDeviceAddress(device, mesh_data.vertices->buffer);
            addr.texcoords_address = vk::getBufferDeviceAddress(device, mesh_data.texcoords->buffer);
            addr.vertex_normals_address = vk::getBufferDeviceAddress(device, mesh_data.vertex_normals->buffer);
            addr.indices_address = vk::getBufferDeviceAddress(device, mesh_data.indices->buffer);
            mesh_data_addresses.push_back(addr);
        }
    }

    allocator->stage_session([&](vk::Allocator &self) {
        subscene_offsets_buf =
            vk::AutoRelease<vk::Buffer>(allocator,
                                        VkBufferCreateInfo{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        },
                                        VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), subscene_offsets);
        mesh_data_addresses_buf =
            vk::AutoRelease<vk::Buffer>(allocator,
                                        VkBufferCreateInfo{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        },
                                        VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), mesh_data_addresses);
    });

    for (const auto &[prototype, transform] : compound.instances) {
        GPUSubSceneInstance instance;
        instance.prototype = prototype;
        instance.transform = transform;
        instances.push_back(instance);
    }
}

void GPUScene::prepare_for_raster()
{
    // TODO
    //
}

void GPUScene::prepare_for_ray_tracing()
{
    build_accel();
    build_materials();
}

void GPUScene::build_accel()
{
    // BLAS
    std::vector<vk::RaytracingBuilderKHR::BlasInput> all_blas;
    all_blas.reserve(subscenes.size());
    for (uint32_t subscene_id = 0; subscene_id < (uint32_t)subscenes.size(); ++subscene_id) {
        vk::RaytracingBuilderKHR::BlasInput blas_input = subscenes[subscene_id].generate_blas_input(device);
        all_blas.push_back(blas_input);
    }
    rt_builder.build_blas(all_blas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // TLAS
    std::vector<VkAccelerationStructureInstanceKHR> all_inst;
    all_inst.reserve(instances.size());
    for (const GPUSubSceneInstance &inst : instances) {
        VkAccelerationStructureInstanceKHR inst_khr{};
        inst_khr.transform = vk::to_transform_matrix_KHR(inst.transform); // Position of the instance
        inst_khr.instanceCustomIndex = inst.prototype;                    // gl_InstanceCustomIndexEXT
        inst_khr.accelerationStructureReference = rt_builder.get_blas_device_address(inst.prototype);
        // TODO: reconsider the following parameters later.
        inst_khr.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst_khr.mask = 0xFF;                                //  Only be hit if rayMask & instance.mask != 0
        inst_khr.instanceShaderBindingTableRecordOffset = 0; // We will use the same hit group for all objects
        all_inst.push_back(inst_khr);
    }
    rt_builder.build_tlas(all_inst, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

void GPUScene::build_materials()
{
    // We have one material per "geometry"
    // VkAccelerationStructureInstanceKHR::instanceCustomIndex (gl_InstanceCustomIndexEXT)
    // Index material by subscene index + geom index -> material index?
    //
    // https://www.reddit.com/r/vulkan/comments/d79pyw/getting_the_index_of_the_vkgeometrynv_within_the/
    // https://gist.github.com/DethRaid/0171f3cfcce51950ee4ef96c64f59617
    // https://docs.vulkan.org/spec/latest/chapters/accelstructures.html#acceleration-structure-geometry-index
    /*
    The index of each element of the pGeometries or ppGeometries members of
    VkAccelerationStructureBuildGeometryInfoKHR is used as the geometry index during ray traversal. The geometry
    index is available in ray shaders via the RayGeometryIndexKHR built-in, and is used to determine hit and
    intersection shaders executed during traversal. The geometry index is available to ray queries via the
    OpRayQueryGetIntersectionGeometryIndexKHR instruction.

    glsl: gl_GeometryIndexEXT
    hlsl: GeometryIndex()
    */
    std::vector<uint32_t> subscene_offsets(subscenes.size());
    std::vector<uint32_t> material_indices;
    std::unordered_map<const GPUMaterial *, uint32_t> material_map;
    for (uint32_t subscene_id = 0; subscene_id < (uint32_t)subscenes.size(); ++subscene_id) {
        uint32_t n_geom = (uint32_t)subscenes[subscene_id].mesh_datas.size();
        for (uint32_t geom_id = 0; geom_id < n_geom; ++geom_id) {
            uint32_t mat_id = (uint32_t)material_map.size();
            auto insert = material_map.insert({subscenes[subscene_id].materials[geom_id].get(), mat_id});
            if (!insert.second) {
                mat_id = insert.first->second;
            }
            material_indices.push_back(mat_id);
        }
        if (subscene_id == 0) {
            subscene_offsets[subscene_id] = 0;
        } else {
            subscene_offsets[subscene_id] = subscene_offsets[subscene_id - 1] + n_geom;
        }
    }
    std::vector<std::byte> material_blocks(material_map.size() * ks::host_device::gpu_material_data_block_size);
    for (auto [mat, mat_id] : material_map) {
        std::span<std::byte, ks::host_device::gpu_material_data_block_size> dest(
            material_blocks.begin() + mat_id * ks::host_device::gpu_material_data_block_size,
            ks::host_device::gpu_material_data_block_size);
        mat->emit_data_block(dest);
    }
    allocator->stage_session([&](vk::Allocator &self) {
        subscene_offsets_buf =
            vk::AutoRelease<vk::Buffer>(allocator,
                                        VkBufferCreateInfo{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        },
                                        VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), subscene_offsets);

        material_indices_buf =
            vk::AutoRelease<vk::Buffer>(allocator,
                                        VkBufferCreateInfo{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        },
                                        VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), material_indices);

        material_blocks_buf =
            vk::AutoRelease<vk::Buffer>(allocator,
                                        VkBufferCreateInfo{
                                            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        },
                                        VMA_MEMORY_USAGE_AUTO, (VmaAllocationCreateFlags)(0), material_blocks);
    });

    constexpr VkShaderStageFlags ray_trace_shader_stages =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
        VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    vk::DescriptorSetHelper helper;
    helper.add_binding("tlas", {.binding = (uint32_t)ks::host_device::SceneBindings::TLAS,
                                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                .descriptorCount = 1,
                                .stageFlags = ray_trace_shader_stages});
    helper.add_binding("subscene_offsets", {.binding = (uint32_t)ks::host_device::SceneBindings::SubSceneOffsets,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = 1,
                                            .stageFlags = ray_trace_shader_stages});
    helper.add_binding("mesh_data_addresses", {.binding = (uint32_t)ks::host_device::SceneBindings::MeshDataAddresses,
                                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                               .descriptorCount = 1,
                                               .stageFlags = ray_trace_shader_stages});
    helper.add_binding("material_indices", {.binding = (uint32_t)ks::host_device::SceneBindings::MaterialIndices,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                            .descriptorCount = 1,
                                            .stageFlags = ray_trace_shader_stages});
    helper.add_binding("material_blocks", {.binding = (uint32_t)ks::host_device::SceneBindings::MaterialBlocks,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           .descriptorCount = 1,
                                           .stageFlags = ray_trace_shader_stages});
    // We only need to specify a maximum array length, but here the max is same as the actual allocated number anyway.
    // Also cap to 1 to avoid 0 descriptorCount validation error.
    uint32_t num_texture2d_combined_sampler = std::max((uint32_t)material_texture_2d_combined_sampler_map.size(), 1u);
    helper.add_binding("material_textures_2d",
                       {.binding = (uint32_t)ks::host_device::SceneBindings::MaterialTextures2D,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = num_texture2d_combined_sampler,
                        .stageFlags = ray_trace_shader_stages},
                       true);
    material_param_block_meta.init(device, 1, std::move(helper));

    material_param_block = material_param_block_meta.allocate_block(num_texture2d_combined_sampler);

    vk::ParameterWriteArray write_array;

    VkAccelerationStructureKHR tlas = rt_builder.get_tlas();
    material_param_block.write_accels("tlas",
                                      VkWriteDescriptorSetAccelerationStructureKHR{
                                          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                          .accelerationStructureCount = 1,
                                          .pAccelerationStructures = &tlas},
                                      0, write_array);

    material_param_block.write_buffer(
        "subscene_offsets",
        VkDescriptorBufferInfo{.buffer = subscene_offsets_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);

    material_param_block.write_buffer(
        "mesh_data_addresses",
        VkDescriptorBufferInfo{.buffer = mesh_data_addresses_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);

    material_param_block.write_buffer(
        "material_indices",
        VkDescriptorBufferInfo{.buffer = material_indices_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);

    material_param_block.write_buffer(
        "material_blocks",
        VkDescriptorBufferInfo{.buffer = material_blocks_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        write_array);

    // batch change layout?
    if (!material_texture_2d_combined_sampler_map.empty()) {
        std::vector<VkDescriptorImageInfo> image_infos(material_texture_2d_combined_sampler_map.size());
        for (auto it = material_texture_2d_combined_sampler_map.begin();
             it != material_texture_2d_combined_sampler_map.end(); ++it) {
            auto [view_id, sampler_id] = it->first;
            uint32_t combined_id = it->second;
            image_infos[combined_id].imageView = material_textures_2d_view_cache.views[view_id];
            image_infos[combined_id].sampler = sampler_cache.samplers[sampler_id];
            image_infos[combined_id].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        material_param_block.write_images("material_textures_2d", std::move(image_infos), 0, write_array);
    }
    vkUpdateDescriptorSets(device, write_array.writes.size(), write_array.writes.data(), 0, nullptr);
}

} // namespace ks