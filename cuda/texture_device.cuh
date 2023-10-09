#pragma once

#include "texture.cuh"
#include "vecmath.cuh"
#include <cuda_fp16.h>

namespace ksc
{

__device__ inline void write_to_half_render_target(const Surface2D &rt, ksc::vec2i pixel, ksc::color3 color)
{
    size_t texel_bytes = (rt.format_desc.x + rt.format_desc.y + rt.format_desc.z + rt.format_desc.w) / 8;
    ushort4 rgba;
    rgba.x = __half_as_ushort(__float2half(color.x));
    rgba.y = __half_as_ushort(__float2half(color.y));
    rgba.z = __half_as_ushort(__float2half(color.z));
    rgba.w = __half_as_ushort(__half(1.0));
    surf2Dwrite(rgba, rt.surf_obj, pixel.x * texel_bytes, pixel.y);
}

} // namespace ksc