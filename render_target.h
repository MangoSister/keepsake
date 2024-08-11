#pragma once
#include "assertion.h"
#include "config.h"
#include "distrib.h"
#include "maths.h"
#include "parallel.h"
#include "tonemap.h"

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

// Assume:
// - the filter can either be sampled exactly or its discretized version can be sampled exactly;
// - and the filter is non-negative (don't use filters with negative lobes).
// Then there is no need to track the sample weight anymore (it's always 1.0 even if the filter is unnormalized).

struct PixelFilter
{
    virtual ~PixelFilter() = default;
    virtual float eval(vec2 offset) const = 0;
    // return the offset w.r.t. pixel center
    virtual vec2 sample(vec2 u) const = 0;
    virtual float radius() const = 0;
};

struct PixelFilterSampleTable
{
    PixelFilterSampleTable() = default;
    PixelFilterSampleTable(const PixelFilter &filter, int res)
        : radius(filter.radius()), n((int)std::ceil(res * radius))
    {
        std::vector<float> f(n * n);
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                vec2 offset(std::lerp(-radius, radius, (x + 0.5f) / n), std::lerp(-radius, radius, (y + 0.5f) / n));
                f[y * n + x] = filter.eval(offset);
            }
        distrib = DistribTable2D(f.data(), n, n);
    }

    vec2 sample(vec2 u) const
    {
        vec2i index;
        float pdf;
        vec2 offset = distrib.sample_linear(u, pdf, &index);
        offset = lerp(arr2::Constant(-radius), arr2::Constant(radius), offset.array());
        return offset;
    }

    float radius;
    int n;
    DistribTable2D distrib;
};

struct BoxPixelFilter : public PixelFilter
{
    explicit BoxPixelFilter(float width = 1.0f) : r(0.5f * width) {}

    float eval(vec2 offset) const final
    {
        if (offset.x() < -r || offset.x() > r || offset.y() < -r || offset.y() > r) {
            return 0.0f;
        }
        return 1.0f;
    }

    vec2 sample(vec2 u) const final { return lerp(arr2::Constant(-r), arr2::Constant(r), u.array()); }

    float radius() const final { return r; }

    float r;
};

struct GaussianPixelFilter : public PixelFilter
{
    GaussianPixelFilter() = default;
    explicit GaussianPixelFilter(float width = 1.5f, float sigma = 0.5f, int sample_table_res = 32)
        : r(0.5f * width), sigma(sigma), threshold(gaussian(r, 0.0f, sigma)), sampler(*this, sample_table_res)
    {}

    float eval(vec2 offset) const final
    {
        if (offset.x() < -r || offset.x() > r || offset.y() < -r || offset.y() > r) {
            return 0.0f;
        }

        return (gaussian(offset.x(), 0.0f, sigma) - threshold) * (gaussian(offset.y(), 0.0f, sigma) - threshold);
    }

    vec2 sample(vec2 u) const final { return sampler.sample(u); }

    float radius() const final { return r; }

    float r;
    float sigma;
    float threshold;
    PixelFilterSampleTable sampler;
};

struct BlackmanHarrisPixelFilter : public PixelFilter
{
    BlackmanHarrisPixelFilter() = default;
    explicit BlackmanHarrisPixelFilter(float width = 1.5f, int sample_table_res = 32)
        : r(0.5f * width), sampler(*this, sample_table_res)
    {}

    float eval(vec2 offset) const final
    {
        if (offset.x() < -r || offset.x() > r || offset.y() < -r || offset.y() > r) {
            return 0.0f;
        }

        constexpr float a0 = 0.35875f;
        constexpr float a1 = 0.48829f;
        constexpr float a2 = 0.14128f;
        constexpr float a3 = 0.01168f;

        float x = (offset.x() + r) / (2.0f * r);
        float fx = a0 - a1 * std::cos(2.0f * pi * x) + a2 * std::cos(4.0f * pi * x) - a3 * std::cos(6.0f * pi * x);
        float y = (offset.y() + r) / (2.0f * r);
        float fy = a0 - a1 * std::cos(2.0f * pi * y) + a2 * std::cos(4.0f * pi * y) - a3 * std::cos(6.0f * pi * y);
        return fx * fy;
    }

    vec2 sample(vec2 u) const final { return sampler.sample(u); }

    float radius() const final { return r; }

    float r;
    PixelFilterSampleTable sampler;
};

std::unique_ptr<PixelFilter> create_pixel_filter(const ConfigArgs &args);

struct RenderTargetArgs
{
    template <int N>
    void add_aov(std::string_view name, color<N> backdrop = color<N>::Zero())
    {
        static_assert(N >= 1 && N <= 4);
        if constexpr (N == 1) {
            aov1.emplace_back(name, backdrop);
        } else if constexpr (N == 2) {
            aov2.emplace_back(name, backdrop);
        } else if constexpr (N == 3) {
            aov3.emplace_back(name, backdrop);
        } else {
            aov4.emplace_back(name, backdrop);
        }
    }

    uint32_t width, height;
    color3 backdrop;
    std::vector<std::pair<std::string_view, color<1>>> aov1;
    std::vector<std::pair<std::string_view, color<2>>> aov2;
    std::vector<std::pair<std::string_view, color<3>>> aov3;
    std::vector<std::pair<std::string_view, color<4>>> aov4;
};

struct RenderTargetPixel
{
    color3 main;
    std::span<const color<1>> aov1;
    std::span<const color<2>> aov2;
    std::span<const color<3>> aov3;
    std::span<const color<4>> aov4;
};

// TODO: support transparent background
// TODO: maybe more composite features
struct RenderTarget2
{
    template <typename T, int N>
        requires std::is_floating_point_v<T>
    struct AOVPlane
    {
        AOVPlane() = default;
        AOVPlane(std::string_view name, color<N> backdrop, size_t n)
            : name(name), backdrop(backdrop), pixels(n, Eigen::Array<T, N, 1>::Zero())
        {}

        std::string name;
        color<N> backdrop;
        std::vector<Eigen::Array<T, N, 1>> pixels;
    };

    struct AOVPtr
    {
        std::byte *ptr;
        size_t pixel_bytes;
    };

    RenderTarget2() = default;
    RenderTarget2(const RenderTargetArgs &args);

    void add(uint32_t x, uint32_t y, float alpha, const RenderTargetPixel &pixel);

    void add_miss(uint32_t x, uint32_t y);

    void clear();

    // void Func(int x, int y, std::byte *pixel_ptr);
    // Caller should know how to cast the data.
    template <typename Func, bool parallel>
    void pixelwise_op(std::string_view name, const Func &op)
    {
        auto it = aov_ptr_table.find(name);
        if (it == aov_ptr_table.end()) {
            return;
        }
        AOVPtr p = it->second;
        if constexpr (parallel) {
            parallel_tile_2d(width, height, [&](int x, int y) {
                std::byte *pixel_ptr = p.ptr + (y * width + x) * p.pixel_bytes;
                op(x, y, pixel_ptr);
            });
        } else {
            for (uint32_t y = 0; y < height; ++y)
                for (uint32_t x = 0; x < width; ++x) {
                    std::byte *pixel_ptr = p.ptr + (y * width + x) * p.pixel_bytes;
                    op(x, y, pixel_ptr);
                }
        }
    }

    // postfix useful for outputing sequences.
    void composite_and_save_to_png(const fs::path &path_prefix, const std::string &path_postfix = {},
                                   const ToneMapper *tonemapper = nullptr) const;
    void composite_and_save_to_exr(const fs::path &path_prefix, const std::string &path_postfix = {},
                                   bool write_fp16 = false) const;

    uint32_t width, height;
    std::vector<arr2d> pixel_weights;
    AOVPlane<double, 3> main;
    std::vector<AOVPlane<float, 1>> aovs1;
    std::vector<AOVPlane<float, 2>> aovs2;
    std::vector<AOVPlane<float, 3>> aovs3;
    std::vector<AOVPlane<float, 4>> aovs4;
    StringHashTable<AOVPtr> aov_ptr_table;
};

} // namespace ks