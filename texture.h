#pragma once
#include "barray.h"
#include "config.h"
#include "maths.h"
#include <cstddef>

// TODO: building mipmaps
// TODO: EWA filtering
// TODO: f16 data type?

enum class TextureDataType
{
    u8,
    f32,
};

enum class TextureWrapMode
{
    Repeat,
    Clamp
};

struct Texture : public Configurable
{
    Texture() = default;
    Texture(const std::byte *bytes, int width, int height, int num_channels, TextureDataType data_type,
            bool build_mipmaps);
    Texture(const std::byte **pyramid_bytes, int width, int height, int num_channels, TextureDataType data_type,
            int levels);

    const std::byte *fetch_raw(int x, int y, int level) const { return pyramid[level].fetch_multi(x, y); }
    void fetch_as_float(int x, int y, int level, float *out) const;
    int levels() const { return (int)pyramid.size(); }

    std::vector<BlockedArray<std::byte>> pyramid;
    int width = 0;
    int height = 0;
    int num_channels = 0;
    TextureDataType data_type = TextureDataType::u8;
};

struct TextureSampler
{
    virtual ~TextureSampler() = default;
    virtual void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, float *out) const = 0;
    color4 operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy) const;

    TextureWrapMode wrap_mode_u = TextureWrapMode::Repeat;
    TextureWrapMode wrap_mode_v = TextureWrapMode::Repeat;
};

struct NearestSampler : public TextureSampler
{
    using TextureSampler::operator();
    void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, float *out) const;
};

struct LinearSampler : public TextureSampler
{
    using TextureSampler::operator();
    void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, float *out) const;
    void bilinear(const Texture &texture, int level, const vec2 &uv, float *out) const;
};

std::unique_ptr<Texture> create_texture(const ConfigArgs &args);