#include "texture.h"
#include "assertion.h"
#include "file_util.h"
#include "image_util.h"
#include "parallel.h"
#include <array>
#include <lz4.h>

namespace ks
{

constexpr int byte_stride(TextureDataType data_type)
{
    switch (data_type) {
    case TextureDataType::u8:
        return 1;
    case TextureDataType::u16:
        return 2;
    case TextureDataType::f32:
    default:
        return 4;
    }
}

Texture::Texture(int width, int height, int num_channels, TextureDataType data_type)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    int levels = 1;
    mips.resize(levels);
    mips[0] = BlockedArray<std::byte>(width, height, stride);
}

Texture::Texture(const std::byte *bytes, int width, int height, int num_channels, TextureDataType data_type,
                 bool build_mipmaps)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    // https://www.nvidia.com/en-us/drivers/np2-mipmapping/
    // NPOT mipmapping with rounding-up.
    int levels = build_mipmaps ? (1 + (int)std::ceil(std::log2(std::max(width, height)))) : 1;
    mips.resize(levels);
    constexpr int min_parallel_res = 256 * 256;
    for (int l = 0; l < levels; ++l) {
        if (l == 0) {
            bool parallel = width * height >= min_parallel_res;
            mips[l] = BlockedArray(width, height, stride, bytes, 2, parallel);
        } else {
            mips[l] = BlockedArray<std::byte>(width, height, stride);
            int last_width = mips[l - 1].ures;
            int last_height = mips[l - 1].vres;
            bool round_up_width = (last_width == width * 2 - 1);
            bool round_up_height = (last_height == height * 2 - 1);
            if (last_width > 1 && round_up_width && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0 && y0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                    }
                    if (y0 > 0) {
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    if (x2 < last_width && y0 > 0) {
                        fetch_as_float(x2, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w0y * v_span[c];
                    }
                    if (x0 > 0) {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w1y * v_span[c];
                    }
                    if (x0 > 0 && y2 < last_height) {
                        fetch_as_float(x0, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w2y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x1, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w2y * v_span[c];
                    }
                    if (x2 < last_width && y2 < last_height) {
                        fetch_as_float(x2, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    constexpr float w0x = 0.5f;
                    constexpr float w1x = 0.5f;
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (y0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x0, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w2y * v_span[c];
                        fetch_as_float(x1, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && round_up_width && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;
                    constexpr float w0y = 0.5f;
                    constexpr float w1y = 0.5f;
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w0y * v_span[c];
                    }
                    if (x0 > 0) {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w1y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x0, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x1, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x0, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x1, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width == 1 && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (y0 > 0) {
                        fetch_as_float(x, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width == 1 && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    fetch_as_float(x, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && round_up_width && last_height == 1) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0) {
                        fetch_as_float(x0, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height == 1) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x0, y, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    fetch_as_float(x1, y, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            }
        }
        width = std::max(1, (width + 1) / 2);
        height = std::max(1, (height + 1) / 2);
    }
}

Texture::Texture(std::span<const std::byte *> mip_bytes, int width, int height, int num_channels,
                 TextureDataType data_type)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    int levels = (int)mip_bytes.size();
    mips.resize(levels);
    for (int i = 0; i < levels; ++i) {
        constexpr int min_parallel_res = 256 * 256;
        bool parallel = width * height >= min_parallel_res;
        mips[i] = BlockedArray(width, height, stride, mip_bytes[i], 2, parallel);
        // Assuming rounding up.
        width = std::max(1, (width + 1) / 2);
        height = std::max(1, (height + 1) / 2);
    }
}

// TODO: fixed-point math for 8bit textures
// TODO: SIMD
void Texture::fetch_as_float(int x, int y, int level, std::span<float> out) const
{
    const std::byte *bytes = fetch_raw(x, y, level);
    switch (data_type) {
    case TextureDataType::u8: {
        const uint8_t *u8_data = reinterpret_cast<const uint8_t *>(bytes);
        int nc = std::min(num_channels, (int)out.size());
        for (int c = 0; c < nc; ++c) {
            out[c] = (float)u8_data[c] / 255.0f;
        }
        break;
    }
    case TextureDataType::u16: {
        const uint16_t *u16_data = reinterpret_cast<const uint16_t *>(bytes);
        int nc = std::min(num_channels, (int)out.size());
        for (int c = 0; c < nc; ++c) {
            out[c] = (float)u16_data[c] / 65535.0f;
        }
        break;
    }
    case TextureDataType::f32:
    default: {
        const float *f32_data = reinterpret_cast<const float *>(bytes);
        int nc = std::min(num_channels, (int)out.size());
        std::copy(f32_data, f32_data + nc, out.data());
        break;
    }
    }
}

void Texture::set_from_float(int x, int y, int level, std::span<const float> in)
{
    ASSERT(in.size() == num_channels);
    std::byte *bytes = mips[level].fetch_multi(x, y);
    switch (data_type) {
    case TextureDataType::u8: {
        uint8_t *u8_data = reinterpret_cast<uint8_t *>(bytes);
        for (int c = 0; c < num_channels; ++c) {
            u8_data[c] = (uint8_t)std::floor(in[c] * 255.0f);
        }
        break;
    }
    case TextureDataType::u16: {
        uint16_t *u16_data = reinterpret_cast<uint16_t *>(bytes);
        for (int c = 0; c < num_channels; ++c) {
            u16_data[c] = (uint16_t)std::floor(in[c] * 65535.0f);
        }
        break;
    }
    case TextureDataType::f32:
    default: {
        float *f32_data = reinterpret_cast<float *>(bytes);
        std::copy(in.data(), in.data() + num_channels, f32_data);
        break;
    }
    }
}

static inline int wrap(int x, int dim, TextureWrapMode mode)
{
    switch (mode) {
    case TextureWrapMode::Repeat:
        return mod(x, dim);
    case TextureWrapMode::Clamp:
        return std::clamp(x, 0, dim - 1);
    default:
        ASSERT(false, "Invalid texture wrap mode.");
        return 0;
    }
}

color4 TextureSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy) const
{
    ASSERT(texture.num_channels <= 4, "Texture has more than 4 channels");
    color4 out = color4::Zero();
    this->operator()(texture, uv, duvdxy, {out.data(), 4});
    return out;
}

void NearestSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float u = uv[0] * texture.mips[0].ures - 0.5f;
    float v = uv[1] * texture.mips[0].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    u0 = wrap(u0, texture.mips[0].ures, wrap_mode_u);
    v0 = wrap(v0, texture.mips[0].vres, wrap_mode_v);

    texture.fetch_as_float(u0, v0, 0, out);
}

void LinearSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float width = duvdxy.cwiseAbs().maxCoeff();
    float level = texture.levels() - 1 + std::log2(std::max(width, (float)1e-8));
    if (level < 0 || texture.levels() == 1) {
        bilinear(texture, 0, uv, out);
    } else if (level >= texture.levels() - 1) {
        texture.fetch_as_float(0, 0, texture.levels() - 1, out);
    } else {
        int ilevel = (int)std::floor(level);
        float delta = level - ilevel;
        size_t nc = std::min((size_t)texture.num_channels, out.size());
        VLA(out0, float, nc);
        bilinear(texture, ilevel, uv, {out0, nc});
        VLA(out1, float, nc);
        bilinear(texture, ilevel + 1, uv, {out1, nc});
        for (int i = 0; i < nc; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

void LinearSampler::bilinear(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const
{
    float u = uv[0] * texture.mips[level].ures - 0.5f;
    float v = uv[1] * texture.mips[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    u0 = wrap(u0, texture.mips[level].ures, wrap_mode_u);
    v0 = wrap(v0, texture.mips[level].vres, wrap_mode_v);
    int u1 = wrap(u0 + 1, texture.mips[level].ures, wrap_mode_u);
    int v1 = wrap(v0 + 1, texture.mips[level].vres, wrap_mode_v);

    float w00 = (1 - du) * (1 - dv);
    float w10 = du * (1 - dv);
    float w01 = (1 - du) * dv;
    float w11 = du * dv;

    size_t nc = std::min((size_t)texture.num_channels, out.size());

    VLA(out00, float, nc);
    texture.fetch_as_float(u0, v0, level, {out00, nc});
    VLA(out10, float, nc);
    texture.fetch_as_float(u1, v0, level, {out10, nc});
    VLA(out01, float, nc);
    texture.fetch_as_float(u0, v1, level, {out01, nc});
    VLA(out11, float, nc);
    texture.fetch_as_float(u1, v1, level, {out11, nc});

    for (int i = 0; i < nc; ++i)
        out[i] = w00 * out00[i] + w10 * out10[i] + w01 * out01[i] + w11 * out11[i];
}

void CubicSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float width = duvdxy.cwiseAbs().maxCoeff();
    float level = texture.levels() - 1 + std::log2(std::max(width, (float)1e-8));
    if (level < 0 || texture.levels() == 1) {
        bicubic(texture, 0, uv, out);
    } else if (level >= texture.levels() - 1) {
        texture.fetch_as_float(0, 0, texture.levels() - 1, out);
    } else {
        int ilevel = (int)std::floor(level);
        float delta = level - ilevel;
        size_t nc = std::min((size_t)texture.num_channels, out.size());
        VLA(out0, float, nc);
        bicubic(texture, ilevel, uv, {out0, nc});
        VLA(out1, float, nc);
        bicubic(texture, ilevel + 1, uv, {out1, nc});
        for (int i = 0; i < nc; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

inline vec4 powers(float x) { return vec4(x * x * x, x * x, x, 1.0f); }

inline void spline(float x, int nc, const float *c0, const float *c1, const float *c2, const float *c3, const vec4 &ca,
                   const vec4 &cb, float *out)
{
    float a0 = cb.dot(powers(x + 1.0));
    float a1 = ca.dot(powers(x));
    float a2 = ca.dot(powers(1.0 - x));
    float a3 = cb.dot(powers(2.0 - x));
    for (int i = 0; i < nc; ++i) {
        out[i] = c0[i] * a0 + c1[i] * a1 + c2[i] * a2 + c3[i] * a3;
    }
}

void CubicSampler::bicubic(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const
{
    float u = uv[0] * texture.mips[level].ures - 0.5f;
    float v = uv[1] * texture.mips[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    vec4i us;
    vec4i vs;
    for (int i = 0; i < 4; ++i) {
        us[i] = wrap(u0 + i - 1, texture.mips[level].ures, wrap_mode_u);
        vs[i] = wrap(v0 + i - 1, texture.mips[level].vres, wrap_mode_v);
    }

    size_t nc = std::min((size_t)texture.num_channels, out.size());
    VLA(rows, float, nc * 4);
    VLA(cols, float, nc * 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x)
            texture.fetch_as_float(us[x], vs[y], level, {&cols[x * nc], nc});
        const float *c0 = &cols[0];
        const float *c1 = &cols[1 * nc];
        const float *c2 = &cols[2 * nc];
        const float *c3 = &cols[3 * nc];
        spline(du, nc, c0, c1, c2, c3, ca, cb, &rows[y * nc]);
    }

    const float *c0 = &rows[0];
    const float *c1 = &rows[1 * nc];
    const float *c2 = &rows[2 * nc];
    const float *c3 = &rows[3 * nc];
    spline(dv, nc, c0, c1, c2, c3, ca, cb, out.data());
}

std::unique_ptr<Texture> create_texture_from_image(int ch, bool build_mipmaps, ColorSpace src_colorspace,
                                                   const fs::path &path)
{
    std::string ext = path.extension().string();
    int width, height;
    TextureDataType data_type;
    std::unique_ptr<float[]> float_data;
    std::unique_ptr<std::byte[]> byte_data;
    const std::byte *ptr = nullptr;
    if (ext == ".exr") {
        float_data = load_from_exr(path, ch, width, height);
        data_type = TextureDataType::f32;
    } else if (ext == ".hdr") {
        float_data = load_from_hdr(path, ch, width, height);
        data_type = TextureDataType::f32;
    } else {
        // TODO: support 16-bit pngs.
        byte_data = load_from_ldr(path, ch, width, height, src_colorspace);
        data_type = TextureDataType::u8;
    }
    ptr = float_data ? reinterpret_cast<const std::byte *>(float_data.get()) : byte_data.get();
    return std::make_unique<Texture>(ptr, width, height, ch, data_type, build_mipmaps);
}

// We use lz4 to compress serialized textures for smaller file size and fast decompress speed.

constexpr const char *serialized_texture_magic = "i_am_a_serialized_texture";

void write_texture_to_serialized(const Texture &texture, const fs::path &path)
{
    BinaryWriter writer(path);
    writer.write_array<char>(serialized_texture_magic, strlen(serialized_texture_magic));
    writer.write<int>(texture.width);
    writer.write<int>(texture.height);
    writer.write<int>(texture.num_channels);
    writer.write<TextureDataType>(texture.data_type);
    int levels = (int)texture.levels();
    writer.write<int>(levels);
    size_t total_size = 0;
    int w = texture.width;
    int h = texture.height;
    int stride = byte_stride(texture.data_type) * texture.num_channels;
    for (int l = 0; l < levels; ++l) {
        total_size += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }
    writer.write<size_t>(total_size);

    std::unique_ptr<std::byte[]> buf = std::make_unique<std::byte[]>(total_size);
    w = texture.width;
    h = texture.height;
    size_t offset = 0;
    for (int l = 0; l < levels; ++l) {
        texture.mips[l].copy_to_linear_array(buf.get() + offset);
        offset += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }

    // LZ4_MAX_INPUT_SIZE is ~2GB. For super high-res textures we need to split the data into blocks.
    int num_blocks = (int)std::ceil((double)total_size / double(LZ4_MAX_INPUT_SIZE));
    writer.write<int>(num_blocks);

    offset = 0;
    size_t total_compressed_capacity = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        total_compressed_capacity += LZ4_compressBound((int)block_size);
        offset += block_size;
    }
    std::unique_ptr<std::byte[]> compressed_buf = std::make_unique<std::byte[]>(total_compressed_capacity);
    offset = 0;
    size_t compressed_offset = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        int compressed_block_capacity = LZ4_compressBound((int)block_size);
        int compressed_block_size =
            LZ4_compress_default((const char *)(buf.get() + offset), (char *)(compressed_buf.get() + compressed_offset),
                                 (int)block_size, compressed_block_capacity);
        ASSERT(compressed_block_size > 0, "lz4 compression failed.");
        writer.write<int>(compressed_block_size);
        compressed_offset += compressed_block_size;
        offset += block_size;
    }
    size_t total_compressed_size = compressed_offset;
    writer.write_array<std::byte>(compressed_buf.get(), total_compressed_size);
}

std::unique_ptr<Texture> create_texture_from_serialized(const fs::path &path)
{
    BinaryReader reader(path);
    std::array<char, std::string_view(serialized_texture_magic).size() + 1> magic;
    reader.read_array<char>(magic.data(), magic.size() - 1);
    magic.back() = 0;
    if (strcmp(magic.data(), serialized_texture_magic)) {
        ASSERT(false, "Invalid serialized texture.");
        return nullptr;
    }
    int width = reader.read<int>();
    int height = reader.read<int>();
    int num_channels = reader.read<int>();
    TextureDataType data_type = reader.read<TextureDataType>();
    int levels = reader.read<int>();
    size_t total_size = reader.read<size_t>();
    int num_blocks = reader.read<int>();
    std::vector<int> compressed_block_sizes(num_blocks);
    size_t total_compressed_size = 0;
    for (int block = 0; block < num_blocks; ++block) {
        compressed_block_sizes[block] = reader.read<int>();
        total_compressed_size += compressed_block_sizes[block];
    }
    //
    std::unique_ptr<std::byte[]> compressed_buf = std::make_unique<std::byte[]>(total_compressed_size);
    reader.read_array<std::byte>(compressed_buf.get(), total_compressed_size);
    std::unique_ptr<std::byte[]> buf = std::make_unique<std::byte[]>(total_size);
    // LZ4_MAX_INPUT_SIZE is ~2GB. For super high-res textures we need to split the data into blocks.
    size_t offset = 0;
    size_t compressed_offset = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        int ret = LZ4_decompress_safe((const char *)(compressed_buf.get() + compressed_offset),
                                      (char *)(buf.get() + offset), compressed_block_sizes[block], (int)block_size);
        ASSERT(ret > 0, "lz4 decompression failed.");
        offset += block_size;
        compressed_offset += compressed_block_sizes[block];
    }
    // compressed_buf can be released now.
    compressed_buf.reset();

    std::vector<const std::byte *> mip_bytes(levels);
    int w = width;
    int h = height;
    int stride = byte_stride(data_type) * num_channels;
    offset = 0;
    for (int l = 0; l < levels; ++l) {
        mip_bytes[l] = buf.get() + offset;
        offset += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }
    return std::make_unique<Texture>(mip_bytes, width, height, num_channels, data_type);
}

std::unique_ptr<Texture> create_texture(const ConfigArgs &args)
{
    fs::path path = args.load_path("path");
    bool serialized = args.load_bool("serialized", false);
    if (serialized) {
        return create_texture_from_serialized(path);
    } else {
        int ch = args.load_integer("channels");
        bool build_mipmaps = args.load_bool("build_mipmaps");
        std::string src_colorspace_str = args.load_string("colorspace", "Linear");
        ColorSpace src_colorspace = ColorSpace::Linear;
        if (src_colorspace_str == "sRGB") {
            src_colorspace = ColorSpace::sRGB;
        }
        return create_texture_from_image(ch, build_mipmaps, src_colorspace, path);
    }
}

std::unique_ptr<TextureSampler> create_texture_sampler(const ConfigArgs &args)
{
    std::unique_ptr<TextureSampler> sampler;
    std::string type = args.load_string("type");
    if (type == "nearest") {
        sampler = std::make_unique<NearestSampler>();
    } else if (type == "linear") {
        sampler = std::make_unique<LinearSampler>();
    } else if (type == "cubic") {
        CubicSampler::Kernel kernel = CubicSampler::Kernel::MitchellNetravali;
        std::string k = args.load_string("kernel", "mitchell");
        if (k == "bspline") {
            kernel = CubicSampler::Kernel::BSpline;
        } else if (k == "catmull_rom") {
            kernel = CubicSampler::Kernel::CatmullRom;
        }
        sampler = std::make_unique<CubicSampler>(kernel);
    }

    std::string wu = args.load_string("wrap_mode_u", "repeat");
    if (wu == "repeat") {
        sampler->wrap_mode_u = TextureWrapMode::Repeat;
    } else if (wu == "clamp") {
        sampler->wrap_mode_u = TextureWrapMode::Clamp;
    }
    std::string wv = args.load_string("wrap_mode_v", "repeat");
    if (wv == "repeat") {
        sampler->wrap_mode_v = TextureWrapMode::Repeat;
    } else if (wv == "clamp") {
        sampler->wrap_mode_v = TextureWrapMode::Clamp;
    }
    return sampler;
}

void convert_texture_task(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    int n_textures = args["textures"].array_size();
    for (int i = 0; i < n_textures; ++i) {
        std::string asset_path = args["textures"].load_string(i);
        const Texture *texture = args.asset_table().get<Texture>(asset_path);
        std::string name = asset_path.substr(asset_path.rfind(".") + 1);
        write_texture_to_serialized(*texture, task_dir / (name + ".bin"));

        // int w = texture->width;
        // int h = texture->height;
        // for (int l = 0; l < texture->levels(); ++l) {
        //    RenderTarget rt(w, h, color3::Zero());
        //    for (int y = 0; y < h; ++y) {
        //        for (int x = 0; x < w; ++x) {
        //            color3 c;
        //            texture->fetch_as_float(x, y, l, {c.data(), 3});
        //            rt(x, y) = c;
        //        }
        //    }
        //    w = std::max(1, (w + 1) / 2);
        //    h = std::max(1, (h + 1) / 2);
        //    rt.save_to_png(task_dir / string_format("%s_%d.png", name.c_str(), l));
        //}
    }
}

} // namespace ks