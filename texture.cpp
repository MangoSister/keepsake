#include "texture.h"
#include "assertion.h"
#include "image_util.h"

KS_NAMESPACE_BEGIN

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
    constexpr int min_parallel_res = 256 * 256;
    bool parallel = width * height >= min_parallel_res;
    pyramid[0] = BlockedArray(width, height, stride, bytes, 2, parallel);
}

Texture::Texture(const std::byte **pyramid_bytes, int width, int height, int num_channels, TextureDataType data_type,
                 int levels)
    : width(width), height(height), num_channels(num_channels), data_type(data_type)
{
    int stride = byte_stride(data_type) * num_channels;
    pyramid.resize(levels);
    for (int i = 0; i < levels; ++i) {
        constexpr int min_parallel_res = 256 * 256;
        bool parallel = width * height >= min_parallel_res;
        pyramid[i] = BlockedArray(width, height, stride, pyramid_bytes[i], 2, parallel);
        width = std::max(1, (width / 2));
        height = std::max(1, (height / 2));
    }
}

// TODO: fixed-point math for 8bit textures
// TODO: SIMD
void Texture::fetch_as_float(int x, int y, int level, std::span<float> out) const
{
    const std::byte *bytes = fetch_raw(x, y, 0);
    switch (data_type) {
    case TextureDataType::u8: {
        const uint8_t *u8_data = reinterpret_cast<const uint8_t *>(bytes);
        int nc = std::min(num_channels, (int)out.size());
        for (int c = 0; c < nc; ++c) {
            out[c] = (float)u8_data[c] / 255.0f;
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
    float u = uv[0] * texture.pyramid[0].ures - 0.5f;
    float v = uv[1] * texture.pyramid[0].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    u0 = wrap(u0, texture.pyramid[0].ures, wrap_mode_u);
    v0 = wrap(v0, texture.pyramid[0].vres, wrap_mode_v);

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
    float u = uv[0] * texture.pyramid[level].ures - 0.5f;
    float v = uv[1] * texture.pyramid[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    u0 = wrap(u0, texture.pyramid[level].ures, wrap_mode_u);
    v0 = wrap(v0, texture.pyramid[level].vres, wrap_mode_v);
    int u1 = wrap(u0 + 1, texture.pyramid[level].ures, wrap_mode_u);
    int v1 = wrap(v0 + 1, texture.pyramid[level].vres, wrap_mode_v);

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
    float u = uv[0] * texture.pyramid[level].ures - 0.5f;
    float v = uv[1] * texture.pyramid[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    vec4i us;
    vec4i vs;
    for (int i = 0; i < 4; ++i) {
        us[i] = wrap(u0 + i - 1, texture.pyramid[level].ures, wrap_mode_u);
        vs[i] = wrap(v0 + i - 1, texture.pyramid[level].vres, wrap_mode_v);
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

std::unique_ptr<Texture> create_texture_from_file(int ch, bool build_mipmaps, const fs::path &path)
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
        byte_data = load_from_ldr(path, ch, width, height);
        data_type = TextureDataType::u8;
    }
    ptr = float_data ? reinterpret_cast<const std::byte *>(float_data.get()) : byte_data.get();
    return std::make_unique<Texture>(ptr, width, height, ch, data_type, build_mipmaps);
}

std::unique_ptr<Texture> create_texture(const ConfigArgs &args)
{
    int ch = args.load_integer("channels");
    bool build_mipmaps = args.load_bool("build_mipmaps");
    fs::path path = args.load_path("path");
    return create_texture_from_file(ch, build_mipmaps, path);
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

KS_NAMESPACE_END