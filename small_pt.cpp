#include "small_pt.h"
#include "file_util.h"
#include "hash.h"
#include "light.h"
#include "log_util.h"
#include "material.h"
#include "mesh_asset.h"
#include "nee.h"
#include "sobol.h"
#include "subsurface.h"

namespace ks
{

void SmallPT::run(const SmallPTInput &in) const
{
    RenderTargetArgs rt_args;
    rt_args.width = in.render_width;
    rt_args.height = in.render_height;
    rt_args.backdrop = in.backdrop;
    RenderTarget2 rt(rt_args);

    int full_render_width = in.render_width;
    int full_render_height = in.render_height;
    int crop_render_start_x = 0;
    int crop_render_start_y = 0;
    int crop_render_width = full_render_width;
    int crop_render_height = full_render_height;
    if (in.crop_width > 0 && in.crop_height > 0) {
        crop_render_start_x = std::max(in.crop_start_x, 0);
        crop_render_start_y = std::max(in.crop_start_y, 0);
        crop_render_width = std::min(in.crop_width, full_render_width - crop_render_start_x);
        crop_render_height = std::min(in.crop_height, full_render_height - crop_render_start_y);
    }

    for (int s_interval_start = 0; s_interval_start < in.spp; s_interval_start += in.spp_prog_interval) {
        int spp_batch = std::min(in.spp_prog_interval, in.spp - s_interval_start);
        int s_interval_end = s_interval_start + spp_batch;
        parallel_tile_2d(crop_render_width, crop_render_height, [&](int offset_x, int offset_y) {
            int pixel_x = crop_render_start_x + offset_x;
            int pixel_y = crop_render_start_y + offset_y;
            cpu_renderer_thread_monitor.record_pixel(pixel_x, pixel_y);

            for (int s = s_interval_start; s < s_interval_end; ++s) {
                cpu_renderer_thread_monitor.record_sample_idx(s);

                PTRenderSampler sampler(arr2u(full_render_width, full_render_height), arr2u(pixel_x, pixel_y), in.spp,
                                        s, in.rng_seed);

                vec2 pixel_sample_offset = in.spp == 1 ? vec2::Zero() : in.pixel_filter->sample(sampler.sobol.next2d());
                vec2 pixel_sample_pos = (vec2(pixel_x + 0.5f, pixel_y + 0.5f) + pixel_sample_offset)
                                            .cwiseQuotient(vec2(full_render_width, full_render_height));
                Ray ray =
                    in.camera->spawn_ray(pixel_sample_pos, vec2i(rt.width, rt.height), in.scale_ray_diff ? in.spp : 1);
                auto [hit, L] = trace(in, ray, sampler);
                if (hit) {
                    RenderTargetPixel pixel;
                    pixel.main = L;
                    rt.add(pixel_x, pixel_y, pixel);
                } else {
                    rt.add_miss(pixel_x, pixel_y);
                }
            }
        });

        if (in.prog_interval_callback) {
            in.prog_interval_callback(rt, s_interval_end);
        }
    }
}

std::pair<bool, color3> SmallPT::trace(const SmallPTInput &in, Ray ray, PTRenderSampler &sampler) const
{
    color3 L = color3::Zero();
    color3 beta = color3::Ones();
    float pdf_wi_bsdf = 0.0f;
    for (int bounce = 0; bounce < in.bounces + 1; ++bounce) {
        cpu_renderer_thread_monitor.record_bounce(bounce);

        SceneHit hit;
        bool isect_geom = in.scene->intersect1(ray, hit);

        if (!isect_geom) {
            // Unidirectional (bsdf sampling) strategy
            for (auto [light_index, sky] : in.light_sampler->get_sky_lights()) {
                vec3 wi = ray.dir.normalized();
                color3 Le = sky->eval(ray.origin, wi);
                if (bounce == 0) {
                    if (in.include_background) {
                        L += beta * Le;
                    }
                } else {
                    float pr_light = in.light_sampler->probability(light_index);
                    float pdf_wi_light = sky->pdf(ray.origin, wi, inf);
                    float mis_weight = power_heur(pdf_wi_bsdf, pdf_wi_light * pr_light);
                    color3 beta_uni = Le * mis_weight;
                    thread_monitor_check(beta_uni.allFinite());
                    if (bounce >= 2 && in.clamp_indirect > 0.0f) {
                        beta_uni = beta_uni.cwiseMin(color3::Constant(in.clamp_indirect));
                    }
                    L += beta * beta_uni;
                }
            }
            if (bounce == 0 && !in.include_background) {
                return {false, color3::Zero()};
            }
            break;
        }
        // Area lights / self emission
        if (hit.material->emission) {
            auto [mesh_light, pr_light] =
                in.light_sampler->get_mesh_light(MeshTriIndex{hit.inst_id, hit.geom_id, hit.prim_id});
            if (mesh_light) {
                vec3 wi = ray.dir.normalized();
                float wi_dist;
                color3 Le = mesh_light->eval(hit.it);
                if (bounce == 0) {
                    L += beta * Le;
                } else {
                    // A light can be too small or too un-important to cause this to underflow.
                    // In this case we treat it as a delta light and ignore the unidirectional strategy.
                    if (pr_light > 0.0f) {
                        float pdf_wi_light = mesh_light->pdf(ray.origin, wi, hit.it.thit);
                        float mis_weight = power_heur(pdf_wi_bsdf, pdf_wi_light * pr_light);
                        color3 beta_uni = Le * mis_weight;
                        thread_monitor_check(beta_uni.allFinite());
                        if (bounce >= 2 && in.clamp_indirect > 0.0f) {
                            beta_uni = beta_uni.cwiseMin(color3::Constant(in.clamp_indirect));
                        }
                        L += beta * beta_uni;
                    }
                }
            }
        }

        if (bounce == in.bounces) {
            break;
        }

        vec3 wo = -ray.dir.normalized();

        LocalGeometry local_geom{in.scene, hit.geom_id};
        vec3 wi;
        Intersection exit;
        // NEE (light sampling) strategy
        MaterialSample s =
            hit.material->sample_with_nee(wo, hit.it, *in.scene, local_geom, *in.light_sampler, sampler, wi, exit);
        thread_monitor_check(s.beta.allFinite());

        L += beta * s.Ld;
        beta *= s.beta;
        if (beta.maxCoeff() == 0.0f) {
            break;
        }
        pdf_wi_bsdf = s.pdf_wi;
        ray = spawn_ray<OffsetType::NextBounce>(exit.p, wi, exit.frame.n, 0.0f, inf);

        // Russian roulette
        if (beta.maxCoeff() < 1.0f && bounce >= 1) {
            float q = std::max(0.0f, 1.0f - beta.maxCoeff());
            if (sampler.sobol.next() < q)
                break;
            beta /= 1.0f - q;
            thread_monitor_check(beta.allFinite());
        }
    }

    return {true, L};
}

void small_pt(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    EmbreeDevice device;
    Scene scene = create_scene(args, device);
    AABB3 scene_bound = scene.bound();
    LightPointers light_ptrs;
    std::unique_ptr<SkyLight> sky;
    if (args.contains("sky")) {
        sky = create_sky_light(args["sky"]);
        light_ptrs.lights.push_back(sky.get());
    }
    std::vector<std::unique_ptr<Light>> lights;
    if (args.contains("light")) {
        int n_lights = args["light"].array_size();
        lights.resize(n_lights);
        for (int i = 0; i < n_lights; ++i) {
            lights[i] = create_light(args["light"][i]);
            light_ptrs.lights.push_back(lights[i].get());
        }
    }
    for (const auto &ml : scene.mesh_lights) {
        light_ptrs.mesh_lights.push_back(ml.get());
    }
    std::unique_ptr<LightSampler> light_sampler = std::make_unique<PowerLightSampler>(scene_bound);
    light_sampler->build(light_ptrs);

    SmallPTInput input;
    input.scene = &scene;
    input.light_sampler = light_sampler.get();

    std::unique_ptr<PixelFilter> pixel_filter;
    if (args.contains("pixel_filter")) {
        pixel_filter = create_pixel_filter(args["pixel_filter"]);
    } else {
        pixel_filter = std::make_unique<BoxPixelFilter>();
    }
    input.pixel_filter = pixel_filter.get();

    std::unique_ptr<ToneMapper> tone_mapper;
    if (args.contains("tone_mapper")) {
        tone_mapper = create_tone_mapper(args["tone_mapper"]);
    }

    if (args.contains("backdrop")) {
        input.backdrop = args.load_vec3("backdrop").array();
        input.include_background = false;
    } else {
        input.include_background = true;
    }
    input.bounces = args.load_integer("bounces");
    input.render_width = args.load_integer("render_width");
    input.render_height = args.load_integer("render_height");
    input.crop_start_x = args.load_integer("crop_start_x", 0);
    input.crop_start_y = args.load_integer("crop_start_y", 0);
    input.crop_width = args.load_integer("crop_width", 0);
    input.crop_height = args.load_integer("crop_height", 0);
    input.spp = args.load_integer("spp");
    input.scale_ray_diff = args.load_bool("scale_ray_diff", true);
    input.rng_seed = args.load_integer("rng_seed", 0);
    input.spp_prog_interval = args.load_integer("spp_prog_interval", 32);

    const CameraAnimation *camera_anim = nullptr;
    std::unique_ptr<Camera> camera_static;
    uint32_t n_frames = 0;
    uint32_t frame_offset = 0;
    uint32_t frame_count = 0;
    if (args.contains("camera_animation")) {
        camera_anim = args.asset_table().get<CameraAnimation>(args.load_string("camera_animation"));
        n_frames = camera_anim->n_frames();
        frame_offset = (uint32_t)args.load_integer("camera_animation_offset", 0);
        frame_count = (uint32_t)args.load_integer("camera_animation_count", 0);
    } else {
        camera_static = create_camera(args["camera"]);
        n_frames = 1;
    }

    SmallPT small_pt;

    uint32_t frame_start = frame_offset;
    uint32_t frame_end = frame_count == 0 ? n_frames : frame_offset + frame_count;
    for (uint32_t frame_idx = frame_start; frame_idx < frame_end; ++frame_idx) {
        get_default_logger().info("Frame [{}/{}] | Start", frame_idx + 1, n_frames);
        get_default_logger().flush();

        Camera camera_frame;
        if (camera_anim) {
            float anim_time = camera_anim->duration() * (frame_idx + 0.5f) / (float)n_frames;
            camera_frame = camera_anim ? camera_anim->eval(anim_time) : *camera_static;
        } else {
            camera_frame = *camera_static;
        }
        input.camera = &camera_frame;

        auto render_start = std::chrono::steady_clock::now();
        auto interval_render_start = render_start;

        input.prog_interval_callback = [&](const RenderTarget2 &rt, int spp_finished) {
            auto interval_render_end = std::chrono::steady_clock::now();
            float interval_render_time =
                std::chrono::duration<float>(interval_render_end - interval_render_start).count();

            get_default_logger().info("Frame [{}/{}] | [{}/{}] spp interval took: {:.1f} sec", frame_idx + 1, n_frames,
                                      spp_finished, input.spp, interval_render_time);
            get_default_logger().flush();
            fs::path save_path_prefix = task_dir / string_format("small_pt_spp%d", spp_finished);
            std::string save_path_postfix = n_frames > 1 ? string_format("%06u", frame_idx) : std::string();
            rt.composite_and_save_to_exr(save_path_prefix, save_path_postfix);
            if (tone_mapper) {
                rt.composite_and_save_to_png(save_path_prefix, save_path_postfix, tone_mapper.get());
            }

            interval_render_start = interval_render_end;
        };

        small_pt.run(input);
        auto render_end = std::chrono::steady_clock::now();
        std::chrono::duration<float> render_time_sec = render_end - render_start;
        get_default_logger().info("Frame [{}/{}] | Rendering time: {:.1f} sec", frame_idx + 1, n_frames,
                                  render_time_sec.count());
        get_default_logger().flush();
    }
}

} // namespace ks
