#pragma once
#include "maths.h"
#include <cstddef>
#include <filesystem>
namespace fs = std::filesystem;

namespace ks
{

enum class ColorSpace
{
    sRGB,
    Linear,
};

std::unique_ptr<std::byte[]> load_from_ldr(const fs::path &path, int c, int &w, int &h, ColorSpace src_colorspace);

std::unique_ptr<float[]> load_from_ldr_to_float(const fs::path &path, int c, int &w, int &h, ColorSpace src_colorspace);

template <int C>
std::unique_ptr<color<C>[]> load_from_ldr_to_float(const fs::path &path, int &w, int &h, ColorSpace src_colorspace) {
    static_assert(C > 0);
    std::unique_ptr<float[]> float_data = load_from_ldr_to_float(path, C, w, h, src_colorspace);
    float *float_ptr = float_data.release();
    color<C> *color_ptr = reinterpret_cast<color<C> *>(float_ptr);
    return std::unique_ptr<color<C>[]>(color_ptr);
}

void save_to_png(const std::byte *data, int w, int h, int c, const fs::path &path);

std::unique_ptr<float[]> load_from_hdr(const fs::path &path, int c, int &w, int &h);

template <int C>
std::unique_ptr<color<C>[]> load_from_hdr(const fs::path &path, int &w, int &h) {
    static_assert(C > 0);
    std::unique_ptr<float[]> float_data = load_from_hdr(path, C, w, h);
    float *float_ptr = float_data.release();
    color<C> *color_ptr = reinterpret_cast<color<C> *>(float_ptr);
    return std::unique_ptr<color<C>[]>(color_ptr);
}

void save_to_hdr(const float *data, int w, int h, int c, const fs::path &path);

std::unique_ptr<float[]> load_from_exr(const fs::path &path, int c, int &w, int &h);

template <int C>
std::unique_ptr<color<C>[]> load_from_exr(const fs::path &path, int &w, int &h) {
    static_assert(C > 0);
    std::unique_ptr<float[]> float_data = load_from_exr(path, C, w, h);
    float *float_ptr = float_data.release();
    color<C> *color_ptr = reinterpret_cast<color<C> *>(float_ptr);
    return std::unique_ptr<color<C>[]>(color_ptr);
}

void save_to_exr(const std::byte *data, bool half, int w, int h, int c, const fs::path &path);

} // namespace ks