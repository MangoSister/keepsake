// shaders.slang

import ks;

struct GlobalParams
{
    PerspectiveCamera camera;
    uint bounces;
    uint full_render_width;
    uint full_render_height;
    uint crop_start_x;
    uint crop_start_y;
    uint spp;
    uint spp_interval_start;
    uint spp_interval_end;
    uint rng_seed;
    uint padding[7]; // pad to vec4 boundary for now...

    RWTexture2D image;
};

/*
`ParameterBlock` is a Slang generic type for binding uniform parameters.
It is similar to `ConstantBuffer` in HLSL, and `ParameterBlock` can include not only constant parameters but also
descriptors such as Texture2D or StructuredBuffer. `ParameterBlock` is designed specifically for d3d/vulkan/metal, so
that parameters are laid out more naturally on these platforms. For Vulkan, when a ParameterBlock doesn't contain nested
parameter block fields, it always maps to a single descriptor set, with a dedicated set number and every resources is
placed into the set with binding index starting from 0.

When both ordinary data fields and resource typed fields exist in a parameter block, all ordinary data fields will be
grouped together into a uniform buffer and appear as a binding 0 of the resulting descriptor set.
*/
ParameterBlock<SceneParams> scene_params;
ParameterBlock<GlobalParams> global_params;

struct RayPayload
{
    __init()
    {
        sampler = PTRenderSampler();
        L = float3(0.0);
        hit = 0.0;
        beta = float3(1.0);
        pdf_wi_bsdf = 0.0;
        next_ray = RayDesc();
    }
    __init(uint2 full_resolution, uint2 pixel, uint samples_per_pixel, uint sample_index, uint32_t seed)
    {
        sampler = PTRenderSampler(full_resolution, pixel, samples_per_pixel, sample_index, seed);
        L = float3(0.0);
        hit = 0.0;
        beta = float3(1.0);
        pdf_wi_bsdf = 0.0;
        next_ray = RayDesc();
    }

    PTRenderSampler sampler;

    float3 L;
    float hit;
    float3 beta;
    float pdf_wi_bsdf;
    RayDesc next_ray;
};

struct ShadowRayPayload
{
    float v;
};

[shader("raygeneration")]
void ray_gen_shader()
{
    uint bounces = global_params.bounces;
    uint full_render_width = global_params.full_render_width;
    uint full_render_height = global_params.full_render_height;
    uint crop_start_x = global_params.crop_start_x;
    uint crop_start_y = global_params.crop_start_y;
    uint spp = global_params.spp;
    uint spp_interval_start = global_params.spp_interval_start;
    uint spp_interval_end = global_params.spp_interval_end;
    uint rng_seed = global_params.rng_seed;

    uint2 pixel = uint2(crop_start_x + DispatchRaysIndex().x, crop_start_y + DispatchRaysIndex().y);
    uint2 full_render_size = uint2(full_render_width, full_render_height);

    BoxPixelFilter pixel_filter;
    float2 pixel_center = (float2)(pixel) + float2(0.5, 0.5);

    float3 pixel_value = float3(0.0);
    for (uint s = spp_interval_start; s < spp_interval_end; ++s) {
        RayPayload payload;
        payload.sampler = PTRenderSampler(full_render_size, pixel, spp, s, rng_seed);
        float2 pixel_sample_offset = spp <= 1 ? float2(0.0) : pixel_filter.sample(payload.sampler.rng.next2d());
        float2 pixel_sample_pos = (pixel_center + pixel_sample_offset) / (float2)(full_render_size);

        RayDesc ray = global_params.camera.spawn_ray(pixel_sample_pos, 0.0, float.maxValue);

        for (int bounce = 0; bounce < bounces + 1; ++bounce) {
            uint ray_flags = RAY_FLAG_NONE;
            if (bounce == bounces) {
                ray_flags |= RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
            }
            TraceRay(scene_params.scene_bvh, // acceleration structure
                     ray_flags,              // rayFlags
                     ~0,                     // cullMask
                     0,                      // sbtRecordOffset
                     0,                      // sbtRecordStride
                     0,                      // missIndex
                     ray,                    // ray
                     payload                 // payload
            );

            // Reaches max bounce || Miss || BSDF sampling failed
            if ((bounce == bounces) || (payload.hit == 0) || (payload.hit > 0 && payload.pdf_wi_bsdf == 0.0)) {
                break;
            }
            // Russian roulette
            float max_beta = max3(payload.beta.r, payload.beta.g, payload.beta.b);
            if (max_beta < 1.0f && bounce >= 1) {
                float q = max(0.0f, 1.0f - max_beta);
                if (payload.sampler.rng.next() < q)
                    break;
                payload.beta /= 1.0f - q;
                // thread_monitor_check(beta.allFinite());
            }
            ray = payload.next_ray;
        }

        // DEBUG. catch weird values...
        // if (!all(isfinite(payload.L)) || !all(payload.L >= 0)) {
        //     pixel_value = float3((float)samples_per_pixel, 0, (float)samples_per_pixel);
        //     break;
        // }
        pixel_value += payload.L.rgb;
    }

    //  (old * spp_interval_start + new ) / spp_interval_end
    pixel_value *= (1.0f / (float)spp_interval_end);
    if (spp_interval_start > 0) {
        float3 old_pixel_value = global_params.image[pixel].rgb;
        pixel_value += old_pixel_value * (float)spp_interval_start / (float)spp_interval_end;
    }
    global_params.image[pixel] = float4(pixel_value, 1.0);
}

[shader("miss")]
void miss_shader(inout RayPayload payload)
{
    // payload.L = float3(1, 0, 1);

    // TODO: Unidirectional (bsdf sampling) strategy
    // if (bounce == 0 && include_background)
    float3 Le = float3(0);
    payload.L += payload.beta * Le;
}

[shader("miss")]
void shadow_miss_shader(inout ShadowRayPayload payload) { payload.v = 1.0; }

[shader("closesthit")]
void closest_hit_shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    GPUMeshDataAddresses mesh_data = scene_params.mesh_data_addresses[InstanceID()];

    Intersection it = compute_intersection(mesh_data, PrimitiveIndex(), attr, WorldRayDirection(), ObjectToWorld(),
                                           WorldToObject(), false);
    // TODO: Area lights / self emission

    uint mat_id = scene_params.subscene_offsets[InstanceID()] + GeometryIndex();
    GPUMaterialPrincipledBSDFDataBlock mat = scene_params.material_blocks[mat_id];

    mat.apply_normal_map(it, scene_params.material_textures_2d);

    IBSDFClosure closure = mat.eval_closure(it.uv, scene_params.material_textures_2d);

    // Vector toward the light
    float3 wi = normalize(float3(0, 0, 1));
    float3 L = float3(1.0, 1.0, 1.0);
    float light_distance = 100000.0;

    float3 wo = -WorldRayDirection();
    float3 wo_local = it.sh_vector_to_local(wo);
    float3 wi_local = it.sh_vector_to_local(wi);
    float3 f = closure.eval(wo_local, wi_local);

    float3 Ld = float3(0.0f, 0.0f, 0.0f);
    // TODO: NEE
    if (any(f > 0.0)) {
        RayDesc ray;
        ray.Origin = it.p;
        ray.Direction = wi;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        ShadowRayPayload shadowPayload = { 0.0 };

        uint shadow_ray_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        // uint flags =
        //     RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        TraceRay(scene_params.scene_bvh, // acceleration structure
                 shadow_ray_flags,       // rayFlags
                 ~0,                     // cullMask
                 0,                      // sbtRecordOffset
                 0,                      // sbtRecordStride
                 1,                      // missIndex
                 ray,                    // ray
                 shadowPayload           // payload
        );
        Ld += f * L * shadowPayload.v;
    }

    payload.hit = 1.0;
    payload.L += payload.beta * Ld;

    float u_bsdf_lobe = payload.sampler.rng.next();
    float2 u_bsdf_wi = payload.sampler.rng.next2d();
    BSDFSample bsdf_sample = closure.sample(wo_local, u_bsdf_lobe, u_bsdf_wi);
    payload.pdf_wi_bsdf = bsdf_sample.pdf_wi;
    if (bsdf_sample.success()) {
        payload.beta *= bsdf_sample.beta;
        // TODO: robust offset
        payload.next_ray.Direction = it.sh_vector_to_world(bsdf_sample.wi);
        payload.next_ray.Origin = it.p;
        payload.next_ray.TMin = 0.001;
        payload.next_ray.TMax = float.maxValue;
    }
}

[shader("anyhit")]
void any_hit_shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // Now only for handling opacity map.

    GPUMeshDataAddresses mesh_data = scene_params.mesh_data_addresses[InstanceID()];

    float2 uv = compute_intersection_lite(mesh_data, PrimitiveIndex(), attr);

    uint mat_id = scene_params.subscene_offsets[InstanceID()] + GeometryIndex();
    GPUMaterialPrincipledBSDFDataBlock mat = scene_params.material_blocks[mat_id];
    if (!mat.apply_opacity_map(uv, WorldRayOrigin(), WorldRayDirection(), scene_params.material_textures_2d)) {
        IgnoreHit();
    }
}
