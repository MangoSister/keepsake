#pragma once
#include "assertion.h"
#include "maths.h"
#include "parallel.h"
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

namespace ks
{

struct RenderTarget
{
    RenderTarget() = default;
    RenderTarget(int width, int height, const color3 &clear) : width(width), height(height)
    {
        pixels.resize(width * height, clear);
    }

    color3 &operator()(int x, int y) { return pixels[y * width + x]; }
    const color3 &operator()(int x, int y) const { return pixels[y * width + x]; }

    template <bool parallel = false>
    RenderTarget &operator+=(const RenderTarget &other)
    {
        ASSERT(width == other.width && height == other.height);

        pixelwise_op<parallel>([&](int index, color3 pixel) { return pixel + other.pixels[index]; });

        return *this;
    }

    template <bool parallel = false>
    RenderTarget &operator*=(const color3 &scalar)
    {
        pixelwise_op<parallel>([&](int index, color3 pixel) { return pixel.cwiseProduct(scalar); });

        return *this;
    }

    template <bool parallel, typename Op>
    void pixelwise_op(const Op &op)
    {
        if constexpr (parallel) {
            parallel_for((int)pixels.size(), [&](int i) { pixels[i] = op(i, pixels[i]); });
        } else {
            for (int i = 0; i < (int)pixels.size(); ++i) {
                pixels[i] = op(i, pixels[i]);
            }
        }
    }

    void save_to_png(const fs::path &path) const;
    void save_to_hdr(const fs::path &path) const;
    void save_to_exr(const fs::path &path) const;

    int width, height;
    std::vector<color3> pixels;
};

} // namespace ks