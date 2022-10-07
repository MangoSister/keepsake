#include "texture.h"
#include "assertion.h"
#include "image_util.h"

constexpr int byte_stride(TextureDataType data_type)
{
    switch (data_type) {
    case TextureDataType::u8:
        return 1;
    case TextureDataType::f32:
    default:
        return 4;
    }
}

Texture::Texture(const std::byte *bytes, int width, int height, int num_channels, TextureDataType data_type,
                 bool build_mipmaps)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    pyramid.resize(1);
    pyramid[0] = BlockedArray(width, height, stride, bytes);
}

Texture::Texture(const std::byte **pyramid_bytes, int width, int height, int num_channels, TextureDataType data_type,
                 int levels)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    pyramid.resize(levels);
    for (int i = 0; i < levels; ++i) {
        pyramid[i] = BlockedArray(width, height, stride, pyramid_bytes[i]);
    }
}

// TODO: fixed-point math for 8bit textures
// TODO: SIMD
void Texture::fetch_as_float(int x, int y, int level, float *out) const
{
    const std::byte *bytes = fetch_raw(x, y, 0);
    switch (data_type) {
    case TextureDataType::u8: {
        const uint8_t *u8_data = reinterpret_cast<const uint8_t *>(bytes);
        for (int c = 0; c < num_channels; ++c) {
            out[c] = (float)u8_data[c] / 255.0f;
        }
        break;
    }
    case TextureDataType::f32:
    default: {
        const float *f32_data = reinterpret_cast<const float *>(bytes);
        std::copy(f32_data, f32_data + num_channels, out);
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
    this->operator()(texture, uv, duvdxy, out.data());
    return out;
}

void NearestSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, float *out) const
{
    float u = uv[0] * texture.pyramid[0].ures - 0.5f;
    float v = uv[1] * texture.pyramid[0].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    u0 = wrap(u0, texture.pyramid[0].ures, wrap_mode_u);
    v0 = wrap(v0, texture.pyramid[0].vres, wrap_mode_v);

    texture.fetch_as_float(u0, v0, 0, out);
}

void LinearSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, float *out) const
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
        VLA(out0, float, texture.num_channels);
        bilinear(texture, ilevel, uv, out0);
        VLA(out1, float, texture.num_channels);
        bilinear(texture, ilevel + 1, uv, out1);
        for (int i = 0; i < texture.num_channels; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

void LinearSampler::bilinear(const Texture &texture, int level, const vec2 &uv, float *out) const
{
    float u = uv[0] * texture.pyramid[0].ures - 0.5f;
    float v = uv[1] * texture.pyramid[0].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    u0 = wrap(u0, texture.pyramid[0].ures, wrap_mode_u);
    v0 = wrap(v0, texture.pyramid[0].vres, wrap_mode_v);
    int u1 = wrap(u0 + 1, texture.pyramid[0].ures, wrap_mode_u);
    int v1 = wrap(v0 + 1, texture.pyramid[0].vres, wrap_mode_v);

    float w00 = (1 - du) * (1 - dv);
    float w10 = du * (1 - dv);
    float w01 = (1 - du) * dv;
    float w11 = du * dv;

    VLA(out00, float, texture.num_channels);
    texture.fetch_as_float(u0, v0, level, out00);
    VLA(out10, float, texture.num_channels);
    texture.fetch_as_float(u1, v0, level, out10);
    VLA(out01, float, texture.num_channels);
    texture.fetch_as_float(u0, v1, level, out01);
    VLA(out11, float, texture.num_channels);
    texture.fetch_as_float(u1, v1, level, out11);

    for (int i = 0; i < texture.num_channels; ++i)
        out[i] = w00 * out00[i] + w10 * out10[i] + w01 * out01[i] + w11 * out11[i];
}

std::unique_ptr<Texture> create_texture(const ConfigArgs &args)
{
    int ch = args.load_integer("channels");
    bool build_mipmaps = args.load_bool("build_mipmaps");
    fs::path path = args.load_path("path");
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
        byte_data = load_from_ldr(path, ch, width, height);
        data_type = TextureDataType::u8;
    }
    ptr = float_data ? reinterpret_cast<const std::byte *>(float_data.get()) : byte_data.get();
    return std::make_unique<Texture>(ptr, width, height, ch, data_type, build_mipmaps);
}
