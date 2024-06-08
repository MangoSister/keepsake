#pragma once
#include "config.h"
#include "geometry.h"
#include "material.h"
#include "scene.h"
#include "texture.h"
#include <filesystem>
#include <memory>
#include <span>
namespace fs = std::filesystem;

namespace ks
{

struct MeshAsset : public Configurable
{
    // TODO: a smarter way to specify twosided
    void load_from_obj(const fs::path &path, bool load_materials, bool twosided, bool use_smooth_normal);
    void load_from_binary(const fs::path &path);
    void write_to_binary(const fs::path &path) const;

    std::vector<std::unique_ptr<MeshData>> meshes;
    std::vector<std::unique_ptr<Material>> materials;
    std::vector<std::unique_ptr<NormalMap>> normal_maps;
    std::vector<std::unique_ptr<OpacityMap>> opacity_maps;
    std::vector<std::unique_ptr<BSDF>> bsdfs;
    std::vector<std::unique_ptr<Texture>> textures;

    std::vector<std::string> mesh_names;
    std::vector<std::string> material_names;
    std::vector<std::string> texture_names;
};

std::unique_ptr<MeshAsset> create_mesh_asset(const ConfigArgs &args);

// Convenient function: create a scene from a single mesh asset.
Scene create_scene_from_mesh_asset(const MeshAsset &mesh_asset, const EmbreeDevice &device);
void assign_material_list(Scene &scene, const ConfigArgs &args_materials);

struct CompoundMeshAsset : public Configurable
{
    void load_from_gltf(const fs::path &path, bool load_materials, bool twosided);

    std::vector<MeshAsset> prototypes;
    std::vector<std::pair<uint32_t, Transform>> instances;
    // Textures are owned here instead of in each child prototype mesh asset.
    std::vector<std::unique_ptr<Texture>> textures;
};

std::unique_ptr<CompoundMeshAsset> create_compound_mesh_asset(const ConfigArgs &args);
Scene create_scene_from_compound_mesh_asset(const CompoundMeshAsset &compound, const EmbreeDevice &device);

} // namespace ks