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

void MeshAsset::load_from_obj(const fs::path &path, bool load_materials, bool twosided)
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
        mesh->indices.reserve(shape.mesh.num_face_vertices.size() * 3);
        // Need to reconstruct index buffer per shape.
        std::unordered_map<tinyobj::index_t, uint32_t, IndexHash, IndexEqual> index_remap;
        mesh->vertices.reserve(shape.mesh.num_face_vertices.size() * 3 * 3);
        mesh->texcoords.reserve(shape.mesh.num_face_vertices.size() * 3 * 2);

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

        if (load_materials) {
            parse_tinyobj_material(materials[material_id], base_path, *this);
        }
        meshes.push_back(std::move(mesh));
        mesh_names.push_back(shape.name);
    }
}

std::unique_ptr<MeshAsset> create_mesh_asset(const ConfigArgs &args)
{
    fs::path path = args.load_path("path");
    bool load_materials = args.load_bool("load_materials");
    bool twosided = args.load_bool("twosided", false);
    std::unique_ptr<MeshAsset> mesh_asset = std::make_unique<MeshAsset>();
    mesh_asset->load_from_obj(path, load_materials, twosided);
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
