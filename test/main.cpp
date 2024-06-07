#include "../camera.h"
#include "../config.h"
#include "../log_util.h"
#include "../material.h"
#include "../mesh_asset.h"
#include "../normal_map.h"
#include "../opacity_map.h"
#include "../parallel.h"
#include "../subsurface.h"
#include "../texture.h"
#include <filesystem>
namespace fs = std::filesystem;
#include <cxxopts.hpp>

using namespace ks;

#define DECLARE_TASK(ENTRY_POINT) void ENTRY_POINT(const ks::ConfigArgs &args, const fs::path &task_dir, int task_id)

namespace ks
{
DECLARE_TASK(small_pt);
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("ks_test", "Keepsake Test");
    // clang-format off
    options.add_options()
        ("n,nthreads", "number of cpu threads", cxxopts::value<int>()->default_value("0"))
        ("d,device", "gpu device index", cxxopts::value<int>()->default_value("0"))
        ("c,config", "config file", cxxopts::value<std::string>()->default_value(std::string(DATA_DIR) + "config.toml"))
        ("r,asset_root_dir", "asset root dir", cxxopts::value<std::string>()->default_value(DATA_DIR));
    // clang-format on
    auto args = options.parse(argc, argv);

    fs::path cfg_path(args["config"].as<std::string>());
    ConfigService cfg;
    std::string asset_root_dir = args["asset_root_dir"].as<std::string>();
    if (!asset_root_dir.empty()) {
        cfg.set_asset_root_dir(fs::path(asset_root_dir));
    }
    cfg.parse_file(cfg_path);

    create_default_logger(cfg.output_directory() / "log.txt");

    int nthreads = args["nthreads"].as<int>();
    // DEBUG
    // nthreads = 1;
    init_parallel(nthreads);

    // These need to be ordered.
    // Shader fields are defined in-place most of the time...but need to retrieve the parser.
    cfg.register_asset("shader_field_1", create_shader_field_color<1>);
    cfg.register_asset("shader_field_2", create_shader_field_color<2>);
    cfg.register_asset("shader_field_3", create_shader_field_color<3>);
    cfg.register_asset("shader_field_4", create_shader_field_color<4>);
    cfg.register_asset("texture", create_texture);
    cfg.register_asset("normal_map", create_normal_map);
    cfg.register_asset("opacity_map", create_opacity_map);
    cfg.register_asset("bsdf", create_bsdf);
    cfg.register_asset("bssrdf", create_bssrdf);
    cfg.register_asset("material", create_material);
    cfg.register_asset("mesh_asset", create_mesh_asset);
    cfg.register_asset("compound_mesh_asset", create_compound_mesh_asset);
    cfg.register_asset("camera_animation", create_camera_animation);
    cfg.load_assets();

    // These does not need to be ordered.
    cfg.register_task("small_pt", small_pt);
    cfg.run_all_tasks();

    return 0;
}