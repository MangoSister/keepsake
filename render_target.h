#pragma once
#include "assertion.h"
#include "maths.h"
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

KS_NAMESPACE_BEGIN

struct RenderTarget
{
    RenderTarget() = default;
    RenderTarget(int width, int height, const color3 &clear) : width(width), height(height)
    {
        pixels.resize(width * height, clear);
    }

    color3 &operator()(int x, int y) { return pixels[y * width + x]; }
    const color3 &operator()(int x, int y) const { return pixels[y * width + x]; }

    RenderTarget &operator+=(const RenderTarget &other)
    {
        ASSERT(width == other.width && height == other.height);
        for (int i = 0; i < (int)pixels.size(); ++i) {
            pixels[i] += other.pixels[i];
        }
        return *this;
    }

    RenderTarget &operator*=(const color3 &scalar)
    {
        for (int i = 0; i < (int)pixels.size(); ++i) {
            pixels[i] = pixels[i].cwiseProduct(scalar);
        }
        return *this;
    }

    void save_to_png(const fs::path &path) const;
    void save_to_hdr(const fs::path &path) const;
    void save_to_exr(const fs::path &path) const;

    int width, height;
    std::vector<color3> pixels;
};

KS_NAMESPACE_END