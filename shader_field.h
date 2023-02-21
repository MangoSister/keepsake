#pragma once
#include "config.h"
#include "ray.h"
#include "texture.h"
#include <numeric>

namespace ks
{

template <typename T>
struct ShaderField : Configurable
{
    virtual ~ShaderField() = default;
    virtual T operator()(const vec2 &uv, const mat2 &duvdxy) const = 0;
    T operator()(const Intersection &it) const
    {
        mat2 duvdxy;
        duvdxy(0, 0) = it.dudx;
        duvdxy(0, 1) = it.dvdx;
        duvdxy(1, 0) = it.dudy;
        duvdxy(1, 1) = it.dvdy;
        return (*this)(it.uv, duvdxy);
    }
};

template <typename T>
struct ConstantField : public ShaderField<T>
{
    ConstantField() = default;
    ConstantField(const T &value) : value(value) {}
    T operator()(const vec2 &uv, const mat2 &duvdxy) const { return value; }

    T value;
};

template <int N>
struct TextureField : public ShaderField<color<N>>
{
    TextureField() = default;
    TextureField(const Texture &texture, std::unique_ptr<TextureSampler> &&sampler, bool flip_v = true,
                 const arri<N> *swizzle = nullptr)
        : texture(&texture), sampler(std::move(sampler)), flip_v(flip_v)
    {
        if (!swizzle)
            std::iota(this->swizzle.data(), this->swizzle.data() + N, 0);
        else {
            this->swizzle = *swizzle;
        }
    }

    color<N> operator()(const vec2 &uv, const mat2 &duvdxy) const
    {
        vec2 flip_uv = uv;
        mat2 flip_duvdxy = duvdxy;
        if (flip_v) {
            flip_uv[1] = 1.0f - flip_uv[1];
            flip_duvdxy(0, 1) *= -1.0f;
            flip_duvdxy(1, 1) *= -1.0f;
        }

        VLA(out, float, texture->num_channels);
        (*sampler)(*texture, flip_uv, flip_duvdxy, {out, (size_t)texture->num_channels});
        color<N> c;
        for (int i = 0; i < N; ++i)
            c[i] = swizzle[i] >= 0 ? out[swizzle[i]] : 0.0f;
        return c;
    }

    const Texture *texture = nullptr;
    std::unique_ptr<TextureSampler> sampler;
    arri<N> swizzle;
    bool flip_v = true; // This is very common for some reasons...
};

template <int N>
std::unique_ptr<ConstantField<color<N>>> create_constant_shader_field_color(const ConfigArgs &args)
{
    static_assert(N >= 1 && N <= 4);
    if constexpr (N == 1) {
        float value = args.load_float("value");
        return std::make_unique<ConstantField<color<N>>>(color<N>(value));
    } else if constexpr (N == 2) {
        color<N> value = args.load_vec2("value").array();
        return std::make_unique<ConstantField<color<N>>>(value);
    } else if constexpr (N == 3) {
        color<N> value = args.load_vec3("value").array();
        return std::make_unique<ConstantField<color<N>>>(color<N>(value));
    } else {
        color<N> value = args.load_vec4("value").array();
        return std::make_unique<ConstantField<color<N>>>(color<N>(value));
    }
}

template <int N>
std::unique_ptr<TextureField<N>> create_texture_shader_field_color(const ConfigArgs &args)
{
    static_assert(N >= 1 && N <= 4);
    const Texture *map = args.asset_table().get<Texture>(args.load_string("map"));
    std::unique_ptr<TextureSampler> sampler = create_texture_sampler(args["sampler"]);
    bool flip_v = args.load_bool("flip_v", true);
    arri<N> swizzle;
    std::iota(swizzle.data(), swizzle.data() + N, 0);
    if (args.contains("swizzle")) {
        if constexpr (N == 1) {
            swizzle = arri<N>(args.load_integer("swizzle"));
        } else if constexpr (N == 2) {
            swizzle = args.load_vec2("swizzle").cast<int>().array();
        } else if constexpr (N == 3) {
            swizzle = args.load_vec3("swizzle").cast<int>().array();
        } else {
            swizzle = args.load_vec4("swizzle").cast<int>().array();
        }
    }
    return std::make_unique<TextureField<N>>(*map, std::move(sampler), flip_v, &swizzle);
}

template <int N>
std::unique_ptr<ShaderField<color<N>>> create_shader_field_color(const ConfigArgs &args)
{
    static_assert(N >= 1 && N <= 4);
    std::string field_type = args.load_string("type");
    std::unique_ptr<ShaderField<color<N>>> field;
    if (field_type == "constant") {
        return create_constant_shader_field_color<N>(args);
    } else if (field_type == "texture") {
        return create_texture_shader_field_color<N>(args);
    }
    return field;
}

using ShaderField1 = ShaderField<color<1>>;
using ShaderField2 = ShaderField<color<2>>;
using ShaderField3 = ShaderField<color<3>>;
using ShaderField4 = ShaderField<color<4>>;

} // namespace ks