#pragma once
#include "barray.h"
#include "config.h"
#include "image_util.h"
#include "maths.h"
#include <cstddef>
#include <span>

namespace ks
{

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
    Texture(int width, int height, int num_channels, TextureDataType data_type);
    Texture(const std::byte *bytes, int width, int height, int num_channels, TextureDataType data_type,
            bool build_mipmaps);
    Texture(std::span<const std::byte *> mip_bytes, int width, int height, int num_channels, TextureDataType data_type);

    const std::byte *fetch_raw(int x, int y, int level) const { return mips[level].fetch_multi(x, y); }
    void fetch_as_float(int x, int y, int level, std::span<float> out) const;
    void set_from_float(int x, int y, int level, std::span<const float> in);
    int levels() const { return (int)mips.size(); }

    std::vector<BlockedArray<std::byte>> mips;
    int width = 0;
    int height = 0;
    int num_channels = 0;
    TextureDataType data_type = TextureDataType::u8;
};

struct TextureSampler
{
    virtual ~TextureSampler() = default;
    virtual void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const = 0;
    color4 operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy) const;

    TextureWrapMode wrap_mode_u = TextureWrapMode::Repeat;
    TextureWrapMode wrap_mode_v = TextureWrapMode::Repeat;
};

struct NearestSampler : public TextureSampler
{
    using TextureSampler::operator();
    void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const;
};

struct LinearSampler : public TextureSampler
{
    using TextureSampler::operator();
    void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const;
    void bilinear(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const;
};

struct CubicSampler : public TextureSampler
{
    enum class Kernel
    {
        MitchellNetravali,
        BSpline,
        CatmullRom,
    };
    CubicSampler(Kernel kernel = Kernel::MitchellNetravali)
    {
        switch (kernel) {
        case Kernel::MitchellNetravali: {
            ca = vec4(21.0, -36.0, 0.0, 16.0) / 18.0;
            cb = vec4(-7.0, 36.0, -60.0, 32.0) / 18.0;
            break;
        }
        case Kernel::BSpline: {
            ca = vec4(3.0, -6.0, 0.0, 4.0) / 6.0;
            cb = vec4(-1.0, 6.0, -12.0, 8.0) / 6.0;
            break;
        }
        case Kernel::CatmullRom:
        default: {
            ca = vec4(3.0, -5.0, 0.0, 2.0) / 2.0;
            cb = vec4(-1.0, 5.0, -8.0, 4.0) / 2.0;
            break;
        }
        }
    }
    using TextureSampler::operator();
    void operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const;
    void bicubic(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const;

    vec4 ca, cb;
};

std::unique_ptr<Texture> create_texture_from_image(int channels, bool build_mipmap, ColorSpace src_colorspace,
                                                   const fs::path &path);
std::unique_ptr<Texture> create_texture_from_serialized(const fs::path &path);
void write_texture_to_serialized(const Texture &texture, const fs::path &path);
std::unique_ptr<Texture> create_texture(const ConfigArgs &args);
std::unique_ptr<TextureSampler> create_texture_sampler(const ConfigArgs &args);

} // namespace ks