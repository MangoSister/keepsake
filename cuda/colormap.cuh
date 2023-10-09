#pragma once
#include "vecmath.cuh"

namespace ksc
{

enum class Colormap
{
    Plasma,
    Inferno,
    Viridis,
};

CUDA_HOST_DEVICE color3 apply_colormap(float x, Colormap map);

} // namespace ksc