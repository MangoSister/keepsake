#pragma once
#include "bsdf.h"
#include "camera.h"
#include "config.h"
#include "light.h"
#include "parallel.h"
#include "render_target.h"
#include "rng.h"
#include "scene.h"

#include <functional>
#include <utility>

namespace ks
{

struct SmallPTInput
{
    const ks::Scene *scene = nullptr;
    const ks::Camera *camera = nullptr;
    const ks::PixelFilter *pixel_filter = nullptr;
    const ks::LightSampler *light_sampler = nullptr;
    bool include_background = true;
    ks::color3 backdrop = ks::color3::Zero();

    int bounces = 1;
    int render_width = 256;
    int render_height = 256;
    int crop_start_x = 0;
    int crop_start_y = 0;
    int crop_width = 0;
    int crop_height = 0;
    int spp = 1;
    int rng_seed = 0;
    bool scale_ray_diff = true;
    float clamp_indirect = 10.0f;

    int spp_prog_interval = 32;
    std::function<void(const ks::RenderTarget2 &rt, int)> prog_interval_callback;
};

struct PTRenderSampler;

struct SmallPT
{
    void run(const SmallPTInput &in) const;
    std::pair<bool, ks::color3> trace(const SmallPTInput &in, ks::Ray ray, ks::PTRenderSampler &sampler) const;
};

} // namespace ks
