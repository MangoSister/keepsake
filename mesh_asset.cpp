#include "mesh_asset.h"
#include "bsdf.h"
#include "config.h"
#include "file_util.h"
#include "hash.h"
#include "maths.h"
#include "parallel.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <array>
#include <unordered_map>

static void parse_tinyobj_material(const tinyobj::material_t &mat, const fs::path &base_path, MeshAsset &asset)
{
    std::unique_ptr<Lambertian> lambert;
    if (!mat.diffuse_texname.empty()) {
        fs::path path = base_path / mat.diffuse_texname;
        std::unique_ptr<Texture> albedo_map = create_texture_from_file(3, true, path);
        std::unique_ptr<LinearSampler> sampler = std::make_unique<LinearSampler>();
        std::unique_ptr<TextureField<3>> albedo = std::make_unique<TextureField<3>>(*albedo_map, std::move(sampler));
        asset.textures.push_back(std::move(albedo_map));
        asset.texture_names.push_back(mat.diffuse_texname);
        lambert = std::make_unique<Lambertian>(std::move(albedo));
    } else {
        color3 albedo(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
        lambert = std::make_unique<Lambertian>(albedo);
    }
    std::unique_ptr<BlendedMaterial> material = std::make_unique<BlendedMaterial>();
    material->bsdf = &*lambert;
    asset.materials.push_back(std::move(material));
    asset.bsdfs.push_back(std::move(lambert));
    asset.material_names.push_back(mat.name);
}

struct IndexHash
{
    std::size_t operator()(const tinyobj::index_t &t) const
    {
        std::size_t ret = 0;
        hash_combine(ret, t.vertex_index, t.normal_index, t.texcoord_index);
        return ret;
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
    Scene scene;
    for (const auto &m : mesh_asset.meshes) {
        scene.geometries.push_back(std::make_unique<MeshGeometry>(*m));
    }
    for (const auto &m : mesh_asset.materials) {
        scene.materials.push_back(&*m);
    }
    scene.create_rtc_scene(device);
    return scene;
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
