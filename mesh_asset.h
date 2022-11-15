#pragma once
#include "geometry.h"
#include "config.h"
#include "material.h"
#include "texture.h"
#include "scene.h"
#include <filesystem>
#include <memory>
namespace fs = std::filesystem;

struct MeshAsset : public Configurable
{
    void load_from_obj(const fs::path &path, bool load_materials);

    std::vector<std::unique_ptr<MeshData>> meshes;
    std::vector<std::unique_ptr<Material>> materials;
    std::vector<std::unique_ptr<BSDF>> bsdfs;
    std::vector<std::unique_ptr<Texture>> textures;

    std::vector<std::string> mesh_names;
    std::vector<std::string> material_names;
    std::vector<std::string> texture_names;
};

std::unique_ptr<MeshAsset> create_mesh_asset(const ConfigArgs &args);

// Convenient function: create a scene from a single mesh asset.
Scene create_scene_from_mesh_asset(const MeshAsset &mesh_asset, const EmbreeDevice &device);