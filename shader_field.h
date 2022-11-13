#pragma once
#include "config.h"
#include "ray.h"
#include "texture.h"
#include <numeric>

template <typename T>
struct ShaderField
{
    virtual ~ShaderField() = default;
    virtual T operator()(const Intersection &it) const = 0;
};

template <typename T>
struct ConstantField : public ShaderField<T>
{
    ConstantField() = default;
    ConstantField(const T &value) : value(value) {}
    T operator()(const Intersection &it) const { return value; }

    T value;
};

template <int N>
struct TextureField : public ShaderField<color<N>>
{
    TextureField() = default;
    TextureField(const Texture &texture, std::unique_ptr<TextureSampler> &&sampler)
        : texture(&texture), sampler(std::move(sampler))
    {
        std::iota(swizzle.data(), swizzle.data() + N, 0);
    }
    TextureField(const Texture &texture, std::unique_ptr<TextureSampler> &&sampler, const arri<N> &swizzle)
        : texture(&texture), sampler(std::move(sampler)), swizzle(swizzle)
    {}

    color<N> operator()(const Intersection &it) const
    {
        mat2 duvdxy;
        duvdxy(0, 0) = it.dudx;
        duvdxy(0, 1) = it.dvdx;
        duvdxy(1, 0) = it.dudy;
        duvdxy(1, 1) = it.dvdy;
        VLA(out, float, texture->num_channels);
        (*sampler)(*texture, it.uv, duvdxy, {out, (size_t)texture->num_channels});
        color<N> c;
        for (int i = 0; i < N; ++i)
            c[i] = swizzle[i] >= 0 ? out[swizzle[i]] : 0.0f;
        return c;
    }

    const Texture *texture = nullptr;
    std::unique_ptr<TextureSampler> sampler;
    arri<N> swizzle;
};

template <int N>
std::unique_ptr<ShaderField<color<N>>> create_shader_field_color(const ConfigArgs &args)
{
    static_assert(N >= 1 && N <= 4);

    std::string field_type = args.load_string("type");
    std::unique_ptr<ShaderField<color<N>>> field;
    if (field_type == "constant") {
        if constexpr (N == 1) {
            float value = args.load_float("value");
            field = std::make_unique<ConstantField<color<N>>>(color<N>(value));
        } else if constexpr (N == 2) {
            color<N> value = args.load_vec2("value").array();
            field = std::make_unique<ConstantField<color<N>>>(value);
        } else if constexpr (N == 3) {
            color<N> value = args.load_vec3("value").array();
            field = std::make_unique<ConstantField<color<N>>>(color<N>(value));
        } else {
            color<N> value = args.load_vec4("value").array();
            field = std::make_unique<ConstantField<color<N>>>(color<N>(value));
        }
    } else if (field_type == "texture") {
        const Texture *map = args.asset_table().get<Texture>(args.load_string("map"));
        std::unique_ptr<TextureSampler> sampler = create_texture_sampler(args["sampler"]);
        if (args.contains("swizzle")) {
            arri<N> swizzle;
            if constexpr (N == 1) {
                swizzle = arri<N>(args.load_integer("swizzle"));
            } else if constexpr (N == 2) {
                swizzle = args.load_vec2("swizzle").cast<int>().array();
            } else if constexpr (N == 3) {
                swizzle = args.load_vec3("swizzle").cast<int>().array();
            } else {
                swizzle = args.load_vec4("swizzle").cast<int>().array();
            }
            field = std::make_unique<TextureField<N>>(*map, std::move(sampler), swizzle);
        } else {
            field = std::make_unique<TextureField<N>>(*map, std::move(sampler));
        }
    }
    return field;
}