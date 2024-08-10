#include "mesh_asset.h"
#include "bsdf.h"
#include "config.h"
#include "file_util.h"
#include "hash.h"
#include "maths.h"
#include "normal_map.h"
#include "opacity_map.h"
#include "parallel.h"
#include "principled_bsdf.h"

// clang-format off
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <stb_image.h>
#include <stb_image_write.h>
#include "tiny_gltf.h"
// clang-format on

#include <algorithm>
#include <array>
#include <iostream>
#include <unordered_map>

namespace ks
{

static void parse_tinyobj_material(const tinyobj::material_t &mat, const fs::path &base_path, MeshAsset &asset)
{
    std::unique_ptr<Lambertian> lambert;
    if (!mat.diffuse_texname.empty()) {
        fs::path path = base_path / mat.diffuse_texname;
        std::unique_ptr<Texture> albedo_map = create_texture_from_image(3, true, ColorSpace::sRGB, path);
        std::unique_ptr<EWASampler> sampler = std::make_unique<EWASampler>();
        std::unique_ptr<TextureField<3>> albedo = std::make_unique<TextureField<3>>(*albedo_map, std::move(sampler));
        asset.textures.push_back(std::move(albedo_map));
        asset.texture_names.push_back(mat.diffuse_texname);
        lambert = std::make_unique<Lambertian>(std::move(albedo));
    } else {
        color3 albedo(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        lambert = std::make_unique<Lambertian>(albedo);
    }
    std::unique_ptr<Material> material = std::make_unique<Material>();
    material->bsdf = &*lambert;
    asset.materials.push_back(std::move(material));
    asset.bsdfs.push_back(std::move(lambert));
    asset.material_names.push_back(mat.name);
}

struct IndexHash
{
    std::size_t operator()(const tinyobj::index_t &t) const
    {
        return hash(t.vertex_index, t.normal_index, t.texcoord_index);
    }
};
struct IndexEqual
{
    bool operator()(const tinyobj::index_t &x, const tinyobj::index_t &y) const
    {
        return x.vertex_index == y.vertex_index && x.normal_index == y.normal_index &&
               x.texcoord_index == y.texcoord_index;
    }
};

void MeshAsset::load_from_obj(const fs::path &path, bool load_materials, bool twosided, bool use_smooth_normal)
{
    fs::path base_path = path.parent_path();
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), reader_config)) {
        printf("Failed to load meshes from %s\n", path.string().c_str());
        if (!reader.Error().empty()) {
            printf("TinyObjReader: %s\n", reader.Error().c_str());
        }
        return;
    }

    if (!reader.Warning().empty()) {
        printf("TinyObjReader: %s\n", reader.Warning().c_str());
    }

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();
    const auto &materials = reader.GetMaterials();

    meshes.reserve(shapes.size());
    if (load_materials)
        bsdfs.reserve(shapes.size());

    for (int s = 0; s < (int)shapes.size(); s++) {
        const tinyobj::shape_t &shape = shapes[s];
        std::unique_ptr<MeshData> mesh = std::make_unique<MeshData>();
        mesh->twosided = twosided;
        mesh->use_smooth_normal = use_smooth_normal;
        mesh->indices.reserve(shape.mesh.num_face_vertices.size() * 3);
        // Need to reconstruct index buffer per shape.
        std::unordered_map<tinyobj::index_t, uint32_t, IndexHash, IndexEqual> index_remap;
        mesh->vertices.reserve(shape.mesh.num_face_vertices.size() * 3 * 3);
        mesh->texcoords.reserve(shape.mesh.num_face_vertices.size() * 3 * 2);
        mesh->vertex_normals.reserve(shape.mesh.num_face_vertices.size() * 3 * 3);

        int material_id = -1;
        for (int f = 0; f < (int)shape.mesh.num_face_vertices.size(); f++) {
            int fv = int(shape.mesh.num_face_vertices[f]);
            if (load_materials) {
                if (material_id == -1) {
                    material_id = shape.mesh.material_ids[f];
                } else {
                    ASSERT(material_id == shape.mesh.material_ids[f], "Don't allow per-face material.");
                }
            }
            ASSERT(fv == 3, "Only accept triangular meshes.");
            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[3 * f + v];
                ASSERT(idx.vertex_index >= 0);
                // idx.texcoord_index
                auto insert = index_remap.insert({idx, (uint32_t)mesh->vertices.size() / 3});
                if (insert.second) {
                    tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                    mesh->vertices.push_back(vx);
                    mesh->vertices.push_back(vy);
                    mesh->vertices.push_back(vz);

                    if (idx.texcoord_index >= 0) {
                        tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                        mesh->texcoords.push_back(tx);
                        mesh->texcoords.push_back(ty);
                    } else {
                        ASSERT(mesh->texcoords.empty(), "Either all or none vertices have uv.");
                    }

                    if (idx.normal_index >= 0) {
                        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

                        mesh->vertex_normals.push_back(nx);
                        mesh->vertex_normals.push_back(ny);
                        mesh->vertex_normals.push_back(nz);
                    } else {
                        ASSERT(mesh->vertex_normals.empty(), "Either all or none vertices have vertex normal.");
                    }
                }
                mesh->indices.push_back(insert.first->second);
            }
        }

        // Add a dummy value to vertex buffer for embree padding.
        mesh->vertices.push_back(std::numeric_limits<float>::quiet_NaN());
        mesh->vertices.shrink_to_fit();
        // Add two dummy values to vertex buffer for embree padding.
        if (!mesh->texcoords.empty()) {
            mesh->texcoords.push_back(std::numeric_limits<float>::quiet_NaN());
            mesh->texcoords.push_back(std::numeric_limits<float>::quiet_NaN());
        }
        mesh->texcoords.shrink_to_fit();
        // Add a dummy values to vertex buffer for embree padding.
        if (!mesh->vertex_normals.empty()) {
            mesh->vertex_normals.push_back(std::numeric_limits<float>::quiet_NaN());
        }
        mesh->vertex_normals.shrink_to_fit();

        if (load_materials) {
            parse_tinyobj_material(materials[material_id], base_path, *this);
        }
        meshes.push_back(std::move(mesh));
        mesh_names.push_back(shape.name);
    }
}

constexpr const char *binary_mesh_asset_magic = "i_am_a_binary_mesh_asset";

void MeshAsset::load_from_binary(const fs::path &path)
{
    BinaryReader reader(path);
    std::array<char, std::string_view(binary_mesh_asset_magic).size() + 1> magic;
    reader.read_array<char>(magic.data(), magic.size() - 1);
    magic.back() = 0;
    if (strcmp(magic.data(), binary_mesh_asset_magic)) {
        ASSERT(false, "Invalid binary mesh asset.");
        return;
    }
    int n_meshes = reader.read<int>();
    meshes.resize(n_meshes);
    for (auto &mesh : meshes) {
        mesh = std::make_unique<MeshData>();
        mesh->twosided = reader.read<bool>();
        mesh->use_smooth_normal = reader.read<bool>();
        int n_verts = reader.read<int>();
        int n_tris = reader.read<int>();
        bool has_texcoord = reader.read<bool>();
        bool has_vertex_normal = reader.read<bool>();

        // Add a dummy value to vertex buffer for embree padding.
        mesh->vertices.resize(3 * n_verts + 1);
        mesh->indices.resize(3 * n_tris);
        reader.read_array<float>(mesh->vertices.data(), mesh->vertices.size());
        reader.read_array<uint32_t>(mesh->indices.data(), mesh->indices.size());
        if (has_texcoord) {
            // Add two dummy values to vertex buffer for embree padding.
            mesh->texcoords.resize(2 * n_verts + 2);
            reader.read_array<float>(mesh->texcoords.data(), mesh->texcoords.size());
        }
        if (has_vertex_normal) {
            mesh->vertex_normals.resize(3 * n_verts + 1);
            reader.read_array<float>(mesh->vertex_normals.data(), mesh->vertex_normals.size());
        }
    }
}

void MeshAsset::write_to_binary(const fs::path &path) const
{
    BinaryWriter writer(path);
    writer.write_array<char>(binary_mesh_asset_magic, strlen(binary_mesh_asset_magic));
    writer.write<int>((int)meshes.size());
    for (const auto &mesh : meshes) {
        writer.write<bool>(mesh->twosided);
        writer.write<bool>(mesh->use_smooth_normal);
        writer.write<int>(mesh->vertex_count());
        writer.write<int>(mesh->tri_count());
        writer.write<bool>(mesh->has_texcoord());
        writer.write<bool>(mesh->has_vertex_normal());
        writer.write_array<float>(mesh->vertices.data(), mesh->vertices.size());
        writer.write_array<uint32_t>(mesh->indices.data(), mesh->indices.size());
        if (mesh->has_texcoord()) {
            writer.write_array<float>(mesh->texcoords.data(), mesh->texcoords.size());
        }
        if (mesh->has_vertex_normal()) {
            writer.write_array<float>(mesh->vertex_normals.data(), mesh->vertex_normals.size());
        }
    }
}

std::unique_ptr<MeshAsset> create_mesh_asset(const ConfigArgs &args)
{
    std::unique_ptr<MeshAsset> mesh_asset = std::make_unique<MeshAsset>();
    fs::path path = args.load_path("path");
    std::string fmt = args.load_string("format", "obj");
    if (fmt == "obj") {
        bool load_materials = args.load_bool("load_materials");
        bool twosided = args.load_bool("twosided", false);
        bool use_smooth_normal = args.load_bool("use_smooth_normal", true);
        mesh_asset->load_from_obj(path, load_materials, twosided, use_smooth_normal);
    } else if (fmt == "bin") {
        mesh_asset->load_from_binary(path);
    } else {
        ASSERT(false, "Unsupported mesh asset format [%s].", fmt.c_str());
    }
    Transform to_world = args.load_transform("to_world", Transform());
    if (!to_world.m.isIdentity()) {
        for (const auto &m : mesh_asset->meshes) {
            m->transform(to_world);
        }
    }
    return mesh_asset;
}

Scene create_scene_from_mesh_asset(const MeshAsset &mesh_asset, const EmbreeDevice &device)
{
    SubScene subscene;
    for (const auto &m : mesh_asset.meshes) {
        subscene.geometries.push_back(std::make_unique<MeshGeometry>(*m));
    }
    for (const auto &m : mesh_asset.materials) {
        subscene.materials.push_back(&*m);
    }
    subscene.create_rtc_scene(device);

    Scene scene;
    scene.add_subscene(std::move(subscene));
    scene.add_instance(device, 0, Transform());
    scene.create_rtc_scene(device);

    return scene;
}

void assign_material_list(Scene &scene, const ConfigArgs &args_materials)
{
    ASSERT(args_materials.array_size() == scene.subscenes.size(), "Material list mismatch dimension.");
    for (int subscene_id = 0; subscene_id < scene.subscenes.size(); ++subscene_id) {
        SubScene &subscene = *scene.subscenes[subscene_id];
        ConfigArgs subscene_materials = args_materials[subscene_id];
        ASSERT(subscene_materials.array_size() == subscene.geometries.size(), "Material list mismatch dimension.");
        subscene.materials.resize(subscene.geometries.size());
        for (int geom_id = 0; geom_id < subscene.geometries.size(); ++geom_id) {
            subscene.materials[geom_id] =
                args_materials.asset_table().get<Material>(subscene_materials.load_string(geom_id));
        }
    }
}

void convert_mesh_asset_task(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    int n_assets = args["assets"].array_size();
    for (int i = 0; i < n_assets; ++i) {
        std::string asset_path = args["assets"].load_string(i);
        const MeshAsset *mesh_asset = args.asset_table().get<MeshAsset>(asset_path);
        std::string name = asset_path.substr(asset_path.rfind(".") + 1);
        mesh_asset->write_to_binary(task_dir / (name + ".bin"));
    }
}

static void copy_accessor_to_linear(const tinygltf::Buffer &buf, const tinygltf::BufferView &view,
                                    const tinygltf::Accessor &acc, uint8_t *dest)
{
    ASSERT(!buf.data.empty());
    const uint8_t *buf_data = buf.data.data();
    const uint8_t *src = buf_data + view.byteOffset + acc.byteOffset;

    int comp_size_in_bytes = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(acc.componentType));
    int num_comp = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(acc.type));
    int element_size_in_bytes = comp_size_in_bytes * num_comp;

    int stride = acc.ByteStride(view);
    for (int i = 0; i < acc.count; ++i) {
        std::copy(src, src + element_size_in_bytes, dest);
        dest += element_size_in_bytes;
        src += stride;
    }
}

template <typename CastFn>
static void copy_and_cast_accessor_to_linear(const tinygltf::Buffer &buf, const tinygltf::BufferView &view,
                                             const tinygltf::Accessor &acc, uint8_t *dest, const CastFn &fn = CastFn())
{
    ASSERT(!buf.data.empty());
    const uint8_t *buf_data = buf.data.data();
    const uint8_t *src = buf_data + view.byteOffset + acc.byteOffset;

    int comp_size_in_bytes = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(acc.componentType));
    int num_comp = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(acc.type));
    int element_size_in_bytes = comp_size_in_bytes * num_comp;

    int stride = acc.ByteStride(view);
    VLA(after_cast, uint8_t, fn.cast_element_size_in_bytes);
    for (int i = 0; i < acc.count; ++i) {
        fn(src, after_cast);

        std::copy(after_cast, after_cast + fn.cast_element_size_in_bytes, dest);
        dest += fn.cast_element_size_in_bytes;
        src += stride;
    }
}

struct CastU16ToU32
{
    void operator()(const uint8_t *src, uint8_t *dst) const
    {
        uint16_t u16 = *reinterpret_cast<const uint16_t *>(src);
        uint32_t u32 = static_cast<uint32_t>(u16);
        *reinterpret_cast<uint32_t *>(dst) = u32;
    }

    size_t cast_element_size_in_bytes = 4;
};

// TODO: support a lot of features such as:
// - more sampler types
// - avoid loading duplicate images

std::unique_ptr<Texture> create_texture_from_gltf(int img_idx, const tinygltf::Model &model, ColorSpace src_colorspace)
{
    const tinygltf::Image &src_img = model.images[img_idx];

    TextureDataType data_type;
    if (src_img.bits == 8 && src_img.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        data_type = TextureDataType::u8;
    } else if (src_img.bits == 16 && src_img.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        data_type = TextureDataType::u16;
    } else if (src_img.bits == 32 && src_img.pixel_type == TINYGLTF_COMPONENT_TYPE_FLOAT) {
        data_type = TextureDataType::f32;
    } else {
        fprintf(stderr, "Unsupported gltf texture data type!\n");
        std::abort();
    }

    const std::byte *bytes = reinterpret_cast<const std::byte *>(src_img.image.data());
    return std::make_unique<Texture>(bytes, src_img.width, src_img.height, src_img.component, data_type, src_colorspace,
                                     true);
}

template <typename TextureInfo>
std::pair<vec2, vec2> get_texture_transform(const TextureInfo &texture_info)
{
    vec2 scale = vec2::Ones();
    vec2 offset = vec2::Zero();
    if (auto it = texture_info.extensions.find("KHR_texture_transform"); it != texture_info.extensions.end()) {
        const auto &scale_ = it->second.Get("scale");
        if (scale_.IsArray()) {
            for (int i = 0; i < 2; ++i) {
                if (scale_.Get(i).IsNumber()) {
                    scale[i] = scale_.Get(i).GetNumberAsDouble();
                }
            }
        }
        const auto &offset_ = it->second.Get("offset");
        if (offset_.IsArray()) {
            for (int i = 0; i < 2; ++i) {
                if (offset_.Get(i).IsNumber()) {
                    offset[i] = offset_.Get(i).GetNumberAsDouble();
                }
            }
        }
    }
    return {scale, offset};
}

template <>
std::pair<vec2, vec2> get_texture_transform(const tinygltf::Value &texture_info)
{
    vec2 scale = vec2::Ones();
    vec2 offset = vec2::Zero();
    if (const auto &extensions = texture_info.Get("extensions"); extensions.IsObject()) {
        if (const auto &texture_transform = extensions.Get("KHR_texture_transform"); texture_transform.IsObject()) {
            const auto &scale_ = texture_transform.Get("scale");
            if (scale_.IsArray()) {
                for (int i = 0; i < 2; ++i) {
                    if (scale_.Get(i).IsNumber()) {
                        scale[i] = scale_.Get(i).GetNumberAsDouble();
                    }
                }
            }
            const auto &offset_ = texture_transform.Get("offset");
            if (offset_.IsArray()) {
                for (int i = 0; i < 2; ++i) {
                    if (offset_.Get(i).IsNumber()) {
                        offset[i] = offset_.Get(i).GetNumberAsDouble();
                    }
                }
            }
        }
    }
    return {scale, offset};
}

template <int N>
std::unique_ptr<TextureField<N>>
create_texture_shader_field(const Texture &texture, int sampler_idx, const tinygltf::Model &model,
                            color<N> scale = color<N>::Ones(), std::optional<arri<N>> swizzle = {},
                            vec2 uv_scale = vec2::Ones(), vec2 uv_offset = vec2::Zero())
{
    const tinygltf::Sampler &src_sampler = model.samplers[sampler_idx];

    std::unique_ptr<TextureSampler> dst_sampler;
    if (src_sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST ||
        src_sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST ||
        src_sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR ||
        src_sampler.minFilter == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
        dst_sampler = std::make_unique<EWASampler>();
    } else {
        dst_sampler = std::make_unique<NearestSampler>();
    }
    return std::make_unique<TextureField<N>>(texture, std::move(dst_sampler), false, swizzle, uv_scale, uv_offset,
                                             scale);
}

static void traverse_gltf_scene_graph(const tinygltf::Model &model,
                                      const std::function<bool(const tinygltf::Model &model, const tinygltf::Node &node,
                                                               const Transform &to_world)> &callback)
{
    auto dfs = [&](int node_idx, Transform transform, auto &self) -> bool {
        const auto &node = model.nodes[node_idx];

        Transform local_transform;
        if (!node.matrix.empty()) {
            mat4 m;
            std::copy(node.matrix.begin(), node.matrix.end(), m.data());
            local_transform = Transform(m);
        } else {
            vec3 scale = vec3::Ones();
            quat rotation = quat::Identity();
            vec3 translation = vec3::Zero();
            if (!node.scale.empty()) {
                scale[0] = node.scale[0];
                scale[1] = node.scale[1];
                scale[2] = node.scale[2];
            }
            if (!node.rotation.empty()) {
                // GLTF rotations are stored as XYZW quaternions
                rotation.w() = node.rotation[3];
                rotation.x() = node.rotation[0];
                rotation.y() = node.rotation[1];
                rotation.z() = node.rotation[2];
            }
            if (!node.translation.empty()) {
                translation[0] = node.translation[0];
                translation[1] = node.translation[1];
                translation[2] = node.translation[2];
            }
            local_transform = Transform(scale_rotate_translate(scale, rotation, translation));
        }

        transform = transform * local_transform;

        if (!callback(model, node, transform)) {
            return false;
        }

        for (int child_idx : node.children) {
            if (!self(child_idx, transform, self)) {
                return false;
            }
        }
        return true;
    };

    for (int root : model.scenes[model.defaultScene].nodes) {
        Transform transform;
        if (!dfs(root, transform, dfs)) {
            break;
        }
    }
}

static tinygltf::Model load_gltf_from_file(const fs::path &path)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool is_binary = path.extension() == ".glb";
    bool ret;
    if (is_binary) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    } else {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    }

    if (!err.empty()) {
        std::cout << "GLTF Loader Error: " << err << "\n";
    }
    if (!warn.empty()) {
        std::cout << "GLTF Loader Warning: " << warn << "\n";
    }
    return model;
}

void traverse_gltf_scene_graph(const fs::path &path,
                               const std::function<bool(const tinygltf::Model &model, const tinygltf::Node &node,
                                                        const Transform &to_world)> &callback)
{
    tinygltf::Model model = load_gltf_from_file(path);
    traverse_gltf_scene_graph(model, callback);
}

CompoundMeshAsset::LoadStats CompoundMeshAsset::load_from_gltf(const fs::path &path,
                                                               LoadMaterialOptions load_material_options)
{
    tinygltf::Model model = load_gltf_from_file(path);

    const std::vector<tinygltf::Mesh> &src_meshes = model.meshes;
    const std::vector<tinygltf::Material> &src_materials = model.materials;
    const std::vector<tinygltf::Texture> &src_textures = model.textures;
    const std::vector<tinygltf::Image> &src_images = model.images;
    const std::vector<tinygltf::Sampler> &src_samplers = model.samplers;
    const std::vector<tinygltf::Buffer> &buffers = model.buffers;
    const std::vector<tinygltf::BufferView> &bufferviews = model.bufferViews;
    const std::vector<tinygltf::Accessor> &accessors = model.accessors;

    if (load_material_options.enable) {
        // Determine color spaces for all texture images.
        std::vector<ColorSpace> colorspaces;
        colorspaces.resize(src_images.size(), ColorSpace::Linear);
        for (uint32_t i = 0; i < (uint32_t)src_materials.size(); ++i) {
            const auto &src = src_materials[i];
            const auto &pbr = src.pbrMetallicRoughness;
            if (pbr.baseColorTexture.index >= 0) {
                colorspaces[src_textures[pbr.baseColorTexture.index].source] = ColorSpace::sRGB;
            }
            if (src.emissiveTexture.index >= 0) {
                colorspaces[src_textures[src.emissiveTexture.index].source] = ColorSpace::sRGB;
            }
        }
        textures.resize(src_images.size());
        for (uint32_t i = 0; i < (uint32_t)src_images.size(); ++i) {
            textures[i] = create_texture_from_gltf(i, model, colorspaces[i]);
        }
    }

    prototypes.resize(src_meshes.size());
    for (int i = 0; i < src_meshes.size(); ++i) {
        MeshAsset mesh_asset;
        const auto &primtives = src_meshes[i].primitives;
        for (int j = 0; j < primtives.size(); ++j) {
            MeshData mesh_data;

            ASSERT(primtives[j].mode == TINYGLTF_MODE_TRIANGLES, "GLTF: Only support triangle primitives!");
            const auto &acc_idx = accessors[primtives[j].indices];
            ASSERT(acc_idx.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ||
                       acc_idx.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT,
                   "GLTF: Unsupported index buffer component type!");
            ASSERT(acc_idx.type == TINYGLTF_TYPE_SCALAR, "GLTF: Unsupport index buffer data type!");
            ASSERT(acc_idx.count % 3 == 0);
            mesh_data.indices.resize(acc_idx.count);
            if (acc_idx.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const auto &view = bufferviews[acc_idx.bufferView];
                const auto &buf = buffers[view.buffer];
                copy_accessor_to_linear(buf, view, acc_idx, reinterpret_cast<uint8_t *>(mesh_data.indices.data()));
            } else {
                const auto &view = bufferviews[acc_idx.bufferView];
                const auto &buf = buffers[view.buffer];
                copy_and_cast_accessor_to_linear<CastU16ToU32>(buf, view, acc_idx,
                                                               reinterpret_cast<uint8_t *>(mesh_data.indices.data()));
            }

            auto it_pos = primtives[j].attributes.find("POSITION");
            ASSERT(it_pos != primtives[j].attributes.end(), "GLTF primitive must have positions!");
            const auto &acc_pos = accessors[it_pos->second];
            ASSERT(acc_pos.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
                   "GLTF: Unsupported vertex position component type!");
            ASSERT(acc_pos.type == TINYGLTF_TYPE_VEC3, "GLTF: Unsupport vertex position data type!");
            // Add a dummy value to vertex buffer for embree padding.
            mesh_data.vertices.resize(acc_pos.count * 3 + 1);
            {
                const auto &view = bufferviews[acc_pos.bufferView];
                const auto &buf = buffers[view.buffer];
                copy_accessor_to_linear(buf, view, acc_pos, reinterpret_cast<uint8_t *>(mesh_data.vertices.data()));
            }

            auto it_normal = primtives[j].attributes.find("NORMAL");
            bool has_normal = it_normal != primtives[j].attributes.end();
            if (has_normal) {
                const auto &acc_normal = accessors[it_normal->second];
                ASSERT(acc_normal.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
                       "GLTF: Unsupported vertex normal component type!");
                ASSERT(acc_normal.type == TINYGLTF_TYPE_VEC3, "GLTF: Unsupport vertex normal data type!");

                // Add a dummy value to vertex buffer for embree padding.
                mesh_data.vertex_normals.resize(acc_normal.count * 3 + 1);
                {
                    const auto &view = bufferviews[acc_normal.bufferView];
                    const auto &buf = buffers[view.buffer];
                    copy_accessor_to_linear(buf, view, acc_normal,
                                            reinterpret_cast<uint8_t *>(mesh_data.vertex_normals.data()));
                }
            }

            auto it_tc0 = primtives[j].attributes.find("TEXCOORD_0");
            bool has_tc0 = it_tc0 != primtives[j].attributes.end();
            if (has_tc0) {
                const auto &acc_tc0 = accessors[it_tc0->second];
                ASSERT(acc_tc0.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT,
                       "GLTF: Unsupported texcoord component type!");
                ASSERT(acc_tc0.type == TINYGLTF_TYPE_VEC2, "GLTF: Unsupport texcoord data type!");

                // Add two dummy value to vertex buffer for embree padding.
                mesh_data.texcoords.resize(acc_tc0.count * 2 + 2);
                {
                    const auto &view = bufferviews[acc_tc0.bufferView];
                    const auto &buf = buffers[view.buffer];
                    copy_accessor_to_linear(buf, view, acc_tc0,
                                            reinterpret_cast<uint8_t *>(mesh_data.texcoords.data()));
                }
            }
            // TODO:
            // - we don't really care about backface culling. usually we just render twosided
            // - However, closed meshes with refraction or subsurface should be single sided so that they represent the
            // interface correctly.
            mesh_data.twosided = true;

            if (load_material_options.enable) {
                // Try our best to convert to PrincipledBSDF....some features are not supported yet or behave
                // differently.
                // TODO: support a lot of features such as:
                // - multiple texture coordinate sets
                // - tinted specular (specular color from KHR_materials_specular)
                // - sheen for cloth/fabric

                // TODO: check color space?
                const auto &src = src_materials[primtives[j].material];
                const auto &pbr = src.pbrMetallicRoughness;

                // Don't use this flag. Blender set it to true even with refraction/subsurface...
                // mesh_data.twosided = src.doubleSided;

                auto dst_mat = std::make_unique<Material>();
                // TODO: refactor loading different types of BSDFs...
                std::unique_ptr<PrincipledBSDF> dst_bsdf;
                std::unique_ptr<PrincipledBRDF> dst_brdf;
                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    dst_bsdf = std::make_unique<PrincipledBSDF>();
                    dst_mat->bsdf = dst_bsdf.get();
                } else {
                    dst_brdf = std::make_unique<PrincipledBRDF>();
                    dst_mat->bsdf = dst_brdf.get();
                }

                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_emissive_strength/README.md
                    float emissive_strength = 1.0f;
                    if (auto it = src.extensions.find("KHR_materials_emissive_strength"); it != src.extensions.end()) {
                        const auto &emissive_strength_ = it->second.Get("emissiveStrength");
                        if (emissive_strength_.Type() == tinygltf::REAL_TYPE ||
                            emissive_strength_.Type() == tinygltf::INT_TYPE) {
                            emissive_strength = emissive_strength_.GetNumberAsDouble();
                        } else {
                            if (emissive_strength_.Type() != tinygltf::NULL_TYPE) {
                                fprintf(stderr, "Unexpected ior value type from KHR_materials_ior!\n");
                            }
                        }
                    }
                    if (src.emissiveTexture.index >= 0) {
                        const auto &emissive_map = textures[src_textures[src.emissiveTexture.index].source];
                        auto [uv_scale, uv_offset] = get_texture_transform(src.emissiveTexture);
                        dst_bsdf->emissive = create_texture_shader_field<3>(
                            *emissive_map, src_textures[src.emissiveTexture.index].sampler, model,
                            emissive_strength *
                                color3(src.emissiveFactor[0], src.emissiveFactor[1], src.emissiveFactor[2]),
                            {}, uv_scale, uv_offset);
                    } else {
                        dst_bsdf->emissive = std::make_unique<ConstantField<color3>>(
                            emissive_strength *
                            color3(src.emissiveFactor[0], src.emissiveFactor[1], src.emissiveFactor[2]));
                    }
                    // TODO: check black texture?
                    if (emissive_strength > 0.0f && (src.emissiveFactor[0] > 0.0f || src.emissiveFactor[1] > 0.0f ||
                                                     src.emissiveFactor[2] > 0.0f)) {
                        dst_mat->emission = dst_bsdf->emissive.get();
                    }
                }

                std::unique_ptr<OpacityMap> om;
                // We will use stochastic test for all non-opaque mode
                if (src.alphaMode != "OPAQUE") {
                    if (!load_material_options.opacity_map_filter ||
                        (load_material_options.opacity_map_filter &&
                         load_material_options.opacity_map_filter(src.name))) {
                        om = std::make_unique<OpacityMap>();
                    }
                }
                if (pbr.baseColorTexture.index >= 0) {
                    const auto &basecolor_map = textures[src_textures[pbr.baseColorTexture.index].source];
                    auto [uv_scale, uv_offset] = get_texture_transform(pbr.baseColorTexture);
                    auto bc = create_texture_shader_field<3>(
                        *basecolor_map, src_textures[pbr.baseColorTexture.index].sampler, model,
                        color3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]), {}, uv_scale,
                        uv_offset);
                    if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                        dst_bsdf->basecolor = std::move(bc);
                    } else {
                        dst_brdf->basecolor = std::move(bc);
                    }
                    if (om) {
                        ASSERT(basecolor_map->num_channels == 4); // Must be rgba
                        om->map = create_texture_shader_field<1>(
                            *basecolor_map, src_textures[pbr.baseColorTexture.index].sampler, model,
                            color<1>(pbr.baseColorFactor[3]), arri<1>(3), uv_scale, uv_offset);
                    }
                } else {
                    auto bc = std::make_unique<ConstantField<color3>>(
                        color3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]));
                    if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                        dst_bsdf->basecolor = std::move(bc);
                    } else {
                        dst_brdf->basecolor = std::move(bc);
                    }
                    if (om) {
                        om->map = std::make_unique<ConstantField<color<1>>>(color<1>(pbr.baseColorFactor[3]));
                    }
                }
                if (om) {
                    dst_mat->opacity_map = om.get();
                    mesh_asset.opacity_maps.push_back(std::move(om));
                }

                if (pbr.metallicRoughnessTexture.index >= 0) {
                    // glTF expects the metallic values to be encoded in the blue (B) channel, and
                    // roughness to be encoded in the green (G) channel of the same image.
                    const auto &mr_map = textures[src_textures[pbr.metallicRoughnessTexture.index].source];
                    auto [uv_scale, uv_offset] = get_texture_transform(pbr.metallicRoughnessTexture);
                    auto rough = create_texture_shader_field<1>(
                        *mr_map, src_textures[pbr.metallicRoughnessTexture.index].sampler, model,
                        color<1>(pbr.roughnessFactor), arri<1>(1), uv_scale, uv_offset);
                    auto metal = create_texture_shader_field<1>(
                        *mr_map, src_textures[pbr.metallicRoughnessTexture.index].sampler, model,
                        color<1>(pbr.metallicFactor), arri<1>(2), uv_scale, uv_offset);
                    if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                        dst_bsdf->roughness = std::move(rough);
                        dst_bsdf->metallic = std::move(metal);
                    } else {
                        dst_brdf->roughness = std::move(rough);
                        dst_brdf->metallic = std::move(metal);
                    }
                } else {
                    auto rough = std::make_unique<ConstantField<color<1>>>(color<1>(pbr.roughnessFactor));
                    auto metal = std::make_unique<ConstantField<color<1>>>(color<1>(pbr.metallicFactor));
                    if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                        dst_bsdf->roughness = std::move(rough);
                        dst_bsdf->metallic = std::move(metal);
                    } else {
                        dst_brdf->roughness = std::move(rough);
                        dst_brdf->metallic = std::move(metal);
                    }
                }

                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_ior/README.md
                    if (auto it = src.extensions.find("KHR_materials_ior"); it != src.extensions.end()) {
                        const auto &ior_ = it->second.Get("ior");
                        if (ior_.Type() == tinygltf::REAL_TYPE || ior_.Type() == tinygltf::INT_TYPE) {
                            dst_bsdf->ior =
                                std::make_unique<ConstantField<color<1>>>(color<1>((float)ior_.GetNumberAsDouble()));
                        } else {
                            if (ior_.Type() != tinygltf::NULL_TYPE) {
                                fprintf(stderr, "Unexpected ior value type from KHR_materials_ior!\n");
                            }
                            dst_bsdf->ior = std::make_unique<ConstantField<color<1>>>(color<1>(1.5f));
                        }
                    } else {
                        dst_bsdf->ior = std::make_unique<ConstantField<color<1>>>(color<1>(1.5f));
                    }
                }
                // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md
                // But we don't really want to support colored specular, so we will multiply "specularFactor" by the
                // mean of "specularColorFactor" and completely ignore "specularColorTexture".
                if (auto it = src.extensions.find("KHR_materials_specular"); it != src.extensions.end()) {
                    const auto &specular_factor = it->second.Get("specularFactor");
                    if (!(specular_factor.IsNumber() || specular_factor.Type() == tinygltf::NULL_TYPE)) {
                        fprintf(stderr, "Unexpected specularFactor value type from KHR_materials_specular!\n");
                    }
                    const auto &specular_texture = it->second.Get("specularTexture");
                    if (!(specular_texture.IsObject() || specular_texture.Type() == tinygltf::NULL_TYPE)) {
                        fprintf(stderr, "Unexpected specularTexture value type from KHR_materials_specular!\n");
                    }
                    const auto &specular_color_factor = it->second.Get("specularColorFactor");
                    if (!(specular_color_factor.IsArray() || specular_color_factor.Type() == tinygltf::NULL_TYPE)) {
                        fprintf(stderr, "Unexpected specularColorFactor value type from KHR_materials_specular!\n");
                    }
                    // const auto &specular_color_texture = it->second.Get("specularColorTexture");
                    float sf = 0.5f;
                    if (specular_factor.IsNumber() || specular_color_factor.IsArray()) {
                        sf = 1.0f;
                        if (specular_factor.IsNumber()) {
                            sf = (float)specular_factor.GetNumberAsDouble();
                        }
                        if (specular_color_factor.IsArray()) {
                            sf *= ((specular_color_factor.Get(0).GetNumberAsDouble() +
                                    specular_color_factor.Get(1).GetNumberAsDouble() +
                                    specular_color_factor.Get(2).GetNumberAsDouble()) /
                                   3.0f);
                        }
                    }
                    if (specular_texture.Type() == tinygltf::OBJECT_TYPE) {
                        const auto &spec_tex_idx_ = specular_texture.Get("index");
                        ASSERT(spec_tex_idx_.Type() == tinygltf::INT_TYPE);
                        int spec_tex_idx = spec_tex_idx_.GetNumberAsInt();
                        const auto &spec_map = textures[src_textures[spec_tex_idx].source];

                        auto [uv_scale, uv_offset] = get_texture_transform(specular_texture);

                        auto spec = create_texture_shader_field<1>(*spec_map, src_textures[spec_tex_idx].sampler, model,
                                                                   color<1>(sf), {}, uv_scale, uv_offset);
                        if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                            dst_bsdf->specular_r0_mul = std::move(spec);
                        } else {
                            dst_brdf->specular = std::move(spec);
                        }
                    } else {
                        auto spec = std::make_unique<ConstantField<color<1>>>(color<1>(sf));
                        if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                            dst_bsdf->specular_r0_mul = std::move(spec);
                        } else {
                            dst_brdf->specular = std::move(spec);
                        }
                    }
                } else {
                    auto spec = std::make_unique<ConstantField<color<1>>>(color<1>(0.5f));
                    if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                        dst_bsdf->specular_r0_mul = std::move(spec);
                    } else {
                        dst_brdf->specular = std::move(spec);
                    }
                }
                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md
                    if (auto it = src.extensions.find("KHR_materials_transmission"); it != src.extensions.end()) {
                        // NOTE: set to single sided per discussion earlier.
                        mesh_data.twosided = false;

                        const auto &trans_factor = it->second.Get("transmissionFactor");
                        const auto &trans_texture = it->second.Get("transmissionTexture");
                        if (trans_texture.Type() == tinygltf::OBJECT_TYPE) {
                            const auto &trans_tex_idx_ = trans_texture.Get("index");
                            ASSERT(trans_tex_idx_.Type() == tinygltf::INT_TYPE);
                            int trans_tex_idx = trans_tex_idx_.GetNumberAsInt();
                            const auto &trans_map = textures[src_textures[trans_tex_idx].source];

                            float t = 0.0f;
                            if (trans_factor.Type() == tinygltf::REAL_TYPE ||
                                trans_factor.Type() == tinygltf::INT_TYPE) {
                                t = (float)trans_factor.GetNumberAsDouble();
                            }
                            auto [uv_scale, uv_offset] = get_texture_transform(trans_texture);

                            dst_bsdf->specular_trans =
                                create_texture_shader_field<1>(*trans_map, src_textures[trans_tex_idx].sampler, model,
                                                               color<1>(t), {}, uv_scale, uv_offset);
                        } else if (trans_factor.Type() == tinygltf::REAL_TYPE ||
                                   trans_factor.Type() == tinygltf::INT_TYPE) {
                            dst_bsdf->specular_trans = std::make_unique<ConstantField<color<1>>>(
                                color<1>((float)trans_factor.GetNumberAsDouble()));
                        } else {
                            if (trans_texture.Type() != tinygltf::NULL_TYPE) {
                                fprintf(stderr,
                                        "Unexpected transmissionTexture value type from KHR_materials_transmission!\n");
                            }
                            if (trans_factor.Type() != tinygltf::NULL_TYPE) {
                                fprintf(stderr,
                                        "Unexpected transmissionFactor value type from KHR_materials_transmission!\n");
                            }
                            dst_bsdf->specular_trans = std::make_unique<ConstantField<color<1>>>(color<1>(0.0f));
                        }
                    } else {
                        dst_bsdf->specular_trans = std::make_unique<ConstantField<color<1>>>(color<1>(0.0f));
                    }
                }
                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    dst_bsdf->microfacet = MicrofacetType::GGX;
                } else {
                    dst_brdf->microfacet = MicrofacetType::GGX;
                }
                if (src.normalTexture.index >= 0) {
                    const auto &normal_texture = textures[src_textures[src.normalTexture.index].source];
                    auto nm = std::make_unique<NormalMap>();
                    nm->strength = src.normalTexture.scale;

                    auto [uv_scale, uv_offset] = get_texture_transform(src.normalTexture);

                    nm->map =
                        create_texture_shader_field<3>(*normal_texture, src_textures[src.normalTexture.index].sampler,
                                                       model, color3::Ones(), {}, uv_scale, uv_offset);
                    dst_mat->normal_map = nm.get();
                    mesh_asset.normal_maps.push_back(std::move(nm));
                }

                if (load_material_options.bsdf_type == LoadMaterialOptions::BSDFType::PrincipledBSDF) {
                    mesh_asset.bsdfs.push_back(std::move(dst_bsdf));
                } else {
                    mesh_asset.bsdfs.push_back(std::move(dst_brdf));
                }
                mesh_asset.materials.push_back(std::move(dst_mat));
                mesh_asset.material_names.push_back(src.name);
            }

            mesh_asset.meshes.emplace_back(std::make_unique<MeshData>(std::move(mesh_data)));
            mesh_asset.mesh_names.push_back(src_meshes[i].name);
        }
        prototypes[i] = std::move(mesh_asset);
    }

    traverse_gltf_scene_graph(model,
                              [&](const tinygltf::Model &model, const tinygltf::Node &node, const Transform &to_world) {
                                  if (node.mesh >= 0) {
                                      uint32_t prototype = node.mesh;
                                      instances.push_back({prototype, to_world});
                                  }
                                  return true;
                              });

    LoadStats stats;
    stats.unique_tri_count = 0;
    stats.instanced_tri_count = 0;
    stats.size_bytes = 0;
    for (const auto &prototype : prototypes) {
        for (const auto &mesh : prototype.meshes) {
            stats.unique_tri_count += mesh->tri_count();
            stats.size_bytes += mesh->vertex_count() * sizeof(float[3]);
            stats.size_bytes += mesh->tri_count() * sizeof(uint32_t);
            if (mesh->has_texcoord()) {
                stats.size_bytes += mesh->vertex_count() * sizeof(float[2]);
            }
            if (mesh->has_vertex_normal()) {
                stats.size_bytes += mesh->vertex_count() * sizeof(float[3]);
            }
        }
    }
    for (const auto &texture : textures) {
        for (const auto &mip : texture->mips) {
            stats.size_bytes += mip.ures * mip.vres * texture->num_channels * byte_stride(texture->data_type);
        }
    }
    for (const auto &inst : instances) {
        for (const auto &mesh : prototypes[inst.first].meshes) {
            stats.instanced_tri_count += mesh->tri_count();
        }
    }

    return stats;
}

std::unique_ptr<CompoundMeshAsset> create_compound_mesh_asset(const ConfigArgs &args)
{
    std::unique_ptr<CompoundMeshAsset> compound = std::make_unique<CompoundMeshAsset>();

    fs::path path = args.load_path("path");
    std::string fmt = args.load_string("format", "glb");
    if (fmt == "glb" || fmt == "gltf") {
        CompoundMeshAsset::LoadMaterialOptions options;
        options.enable = args.load_bool("load_materials");
        options.bsdf_type = args.load_string("bsdf_type", "principled_bsdf") == "principled_bsdf"
                                ? CompoundMeshAsset::LoadMaterialOptions::BSDFType::PrincipledBSDF
                                : CompoundMeshAsset::LoadMaterialOptions::BSDFType::PrincipledBRDF;
        compound->load_from_gltf(path, options);
    } else {
        ASSERT(false, "Unsupported mesh asset format [%s].", fmt.c_str());
    }

    return compound;
}

Scene create_scene_from_compound_mesh_asset(const CompoundMeshAsset &compound, const EmbreeDevice &device)
{
    Scene scene;
    for (const auto &p : compound.prototypes) {
        SubScene subscene;
        subscene.geometries.reserve(p.meshes.size());
        for (const auto &m : p.meshes) {
            subscene.geometries.push_back(std::make_unique<MeshGeometry>(*m));
        }
        subscene.materials.reserve(p.meshes.size());
        for (const auto &m : p.materials) {
            subscene.materials.push_back(&*m);
        }
        subscene.create_rtc_scene(device);
        scene.add_subscene(std::move(subscene));
    }
    scene.instances.reserve(compound.instances.size());
    for (const auto &[prototype, transform] : compound.instances) {
        scene.add_instance(device, prototype, transform);
    }
    scene.create_rtc_scene(device);
    return scene;
}

} // namespace ks