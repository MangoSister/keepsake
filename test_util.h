#pragma once
#include "fur_renderer.h"
#include "renderer.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
namespace fs = std::filesystem;

static inline std::string current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S");
    return ss.str();
}

template <typename Func>
void test_case(const fs::path &root_dir, const std::string &test_name, const Func &test_func)
{
    printf("Start test [%s]...\n", test_name.c_str());
    fs::path output_dir = root_dir / test_name;
    if (!fs::create_directory(output_dir)) {
        printf("Cannot create output directory for [%s]. Skipping...\n", test_name.c_str());
        return;
    }

    auto start = std::chrono::system_clock::now();
    test_func(output_dir);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> duration = end - start;
    printf("Finished [%s]. Took %.3f sec.\n", test_name.c_str(), duration.count());
}

// {bcsdf, light, camera, bounces, (integrator)}
struct TestRenderConfig
{
    FurRendererInput make_render_input_old(const Scene &scene) const
    {
        FurRendererInput render_input;
        render_input.geometry = &scene;
        render_input.material = material;
        render_input.light = *light;
        render_input.camera = *camera;
        render_input.image_width = image_width;
        render_input.image_height = image_height;
        render_input.spp = geom_spp;
        render_input.switch_bounces = -1;
        render_input.max_bounces = bounces;
        return render_input;
    }
    RendererInput make_render_input(const CorrelatedMedium *media, const Scene *geom) const
    {
        RendererInput render_input;
        render_input.media = media;
        render_input.geom = geom;
        render_input.light = *light;
        render_input.camera = *camera;
        render_input.image_width = image_width;
        render_input.image_height = image_height;
        render_input.spp = media_spp;
        render_input.bounces = bounces;
        return render_input;
    }

    const Material *material = nullptr;
    const Light *light = nullptr;
    const Camera *camera = nullptr;
    int bounces;

    int image_width;
    int image_height;
    int geom_spp;
    int media_spp;

    std::string str;
};

struct TestRenderConfigCollection
{
    struct Iterator
    {
        Iterator(const TestRenderConfigCollection &collection) : collection(&collection), index(0) {}

        void next() { ++index; }
        bool end() const
        {
            int num_mat = (int)collection->materials.size();
            int num_lights = (int)collection->lights.size();
            int num_cameras = (int)collection->cameras.size();
            int num_bounces = (int)collection->bounces.size();
            return index >= num_mat * num_lights * num_cameras * num_bounces;
        }
        TestRenderConfig operator*() const
        {
            int num_mat = (int)collection->materials.size();
            int num_lights = (int)collection->lights.size();
            int num_cameras = (int)collection->cameras.size();
            int num_bounces = (int)collection->bounces.size();

            int i = index;
            int mat_index = i / (num_lights * num_cameras * num_bounces);
            i -= mat_index * (num_lights * num_cameras * num_bounces);
            int light_index = i / (num_cameras * num_bounces);
            i -= light_index * (num_cameras * num_bounces);
            int camera_index = i / num_bounces;
            i -= camera_index * (num_bounces);
            int bounce_index = i;

            TestRenderConfig config;
            config.material = collection->materials[mat_index].second.get();
            config.light = &collection->lights[light_index].second;
            config.camera = &collection->cameras[camera_index].second;
            config.bounces = collection->bounces[bounce_index];

            config.image_width = collection->image_width;
            config.image_height = collection->image_height;
            config.geom_spp = collection->geom_spp;
            config.media_spp = collection->media_spp;

            config.str = std::string();
            config.str += collection->materials[mat_index].first + '_';
            config.str += collection->lights[light_index].first + '_';
            config.str += collection->cameras[camera_index].first + '_';
            config.str += std::to_string(collection->bounces[bounce_index]) + '_';
            config.str += std::to_string(collection->image_width) + '_';
            config.str += std::to_string(collection->image_height) + '_';
            config.str += std::to_string(collection->ensemble_size) + '_';
            config.str += std::to_string(collection->geom_spp) + '_';
            config.str += std::to_string(collection->media_spp);

            return config;
        }

        const TestRenderConfigCollection *collection = nullptr;
        int index = 0;
    };

    Iterator iterator() const { return Iterator(*this); }

    std::vector<std::pair<std::string, std::unique_ptr<Material>>> materials;
    std::vector<std::pair<std::string, Light>> lights;
    std::vector<std::pair<std::string, Camera>> cameras;
    std::vector<int> bounces;
    int image_width;
    int image_height;
    int ensemble_size;
    int geom_spp;
    int media_spp;
};
