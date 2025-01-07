#include "render_target.h"
#include "file_util.h"
#include "image_util.h"

namespace ks
{

void RenderTarget::save_to_png(const fs::path &path) const
{
    auto buf = std::make_unique<std::uint8_t[]>(width * height * 3);
    for (int i = 0; i < (int)pixels.size(); ++i) {
        for (int c = 0; c < 3; ++c)
            buf[3 * i + c] = (uint8_t)std::floor(pixels[i][c] * 255.0f);
    }
    ks::save_to_png((const std::byte *)buf.get(), width, height, 3, path);
}

void RenderTarget::save_to_hdr(const fs::path &path) const
{
    ks::save_to_hdr(reinterpret_cast<const float *>(pixels.data()), width, height, 3, path);
}

void RenderTarget::save_to_exr(const fs::path &path) const
{
    ks::save_to_exr(reinterpret_cast<const std::byte *>(pixels.data()), false, width, height, 3, path);
}

std::unique_ptr<PixelFilter> create_pixel_filter(const ConfigArgs &args)
{
    std::string type = args.load_string("type");
    if (type == "box") {
        float width = args.load_float("width", 1.0f);
        return std::make_unique<BoxPixelFilter>(width);
    } else if (type == "gaussian") {
        float width = args.load_float("width", 3.0f);
        float sigma = args.load_float("sigma", 0.5f);
        return std::make_unique<GaussianPixelFilter>(width, sigma);
    } else if (type == "blackman_harris") {
        float width = args.load_float("width", 3.0f);
        return std::make_unique<BlackmanHarrisPixelFilter>(width);
    }
    fprintf(stderr, "Invalid pixel filter type [%s]", type.c_str());
    std::abort();
}

RenderTarget2::RenderTarget2(const RenderTargetArgs &args) : width(args.width), height(args.height)
{
    size_t n = width * height;
    pixel_weights.resize(n, arr2d::Zero());
    main = AOVPlane<double, 3>("", args.backdrop, n);
    for (auto [name, bd] : args.aov1) {
        this->aovs1.emplace_back(name, bd, n);
    }
    for (auto [name, bd] : args.aov2) {
        this->aovs2.emplace_back(name, bd, n);
    }
    for (auto [name, bd] : args.aov3) {
        this->aovs3.emplace_back(name, bd, n);
    }
    for (auto [name, bd] : args.aov4) {
        this->aovs4.emplace_back(name, bd, n);
    }

    for (auto &plane : aovs1) {
        aov_ptr_table.insert(
            {plane.name, AOVPtr{reinterpret_cast<std::byte *>(plane.pixels.data()), sizeof(float) * 1}});
    }
    for (auto &plane : aovs2) {
        aov_ptr_table.insert(
            {plane.name, AOVPtr{reinterpret_cast<std::byte *>(plane.pixels.data()), sizeof(float) * 2}});
    }
    for (auto &plane : aovs3) {
        aov_ptr_table.insert(
            {plane.name, AOVPtr{reinterpret_cast<std::byte *>(plane.pixels.data()), sizeof(float) * 3}});
    }
    for (auto &plane : aovs4) {
        aov_ptr_table.insert(
            {plane.name, AOVPtr{reinterpret_cast<std::byte *>(plane.pixels.data()), sizeof(float) * 4}});
    }
}

void RenderTarget2::add(uint32_t x, uint32_t y, float alpha, const RenderTargetPixel &pixel)
{
    uint32_t idx = y * width + x;
    alpha = saturate(alpha);
    pixel_weights[idx][0] += alpha;
    pixel_weights[idx][1] += 1.0f - alpha;
    main.pixels[idx] += pixel.main.cast<double>() * alpha;
    for (uint32_t j = 0; j < (uint32_t)pixel.aov1.size(); ++j) {
        aovs1[j].pixels[idx] += pixel.aov1[j] * alpha;
    }
    for (uint32_t j = 0; j < (uint32_t)pixel.aov2.size(); ++j) {
        aovs2[j].pixels[idx] += pixel.aov2[j] * alpha;
    }
    for (uint32_t j = 0; j < (uint32_t)pixel.aov3.size(); ++j) {
        aovs3[j].pixels[idx] += pixel.aov3[j] * alpha;
    }
    for (uint32_t j = 0; j < (uint32_t)pixel.aov4.size(); ++j) {
        aovs4[j].pixels[idx] += pixel.aov4[j] * alpha;
    }
}

void RenderTarget2::add_miss(uint32_t x, uint32_t y)
{
    uint32_t idx = y * width + x;
    ++pixel_weights[idx][1];
}

void RenderTarget2::clear()
{
    std::fill(main.pixels.begin(), main.pixels.end(), color3d::Zero());
    std::fill(pixel_weights.begin(), pixel_weights.end(), arr2d::Zero());
    for (auto &plane : aovs1) {
        std::fill(plane.pixels.begin(), plane.pixels.end(), color<1>::Zero());
    }
    for (auto &plane : aovs2) {
        std::fill(plane.pixels.begin(), plane.pixels.end(), color<2>::Zero());
    }
    for (auto &plane : aovs3) {
        std::fill(plane.pixels.begin(), plane.pixels.end(), color<3>::Zero());
    }
    for (auto &plane : aovs4) {
        std::fill(plane.pixels.begin(), plane.pixels.end(), color<4>::Zero());
    }
}

void RenderTarget2::composite_and_save_to_png(const fs::path &path_prefix, const std::string &path_postfix,
                                              const ToneMapper *tonemapper) const
{
    auto buf = std::make_unique<std::uint8_t[]>(width * height * 4); // allocate for max 4 components.
    parallel_for((uint32_t)main.pixels.size(), [&](uint32_t i) {
        double w_sum = (pixel_weights[i][0] + pixel_weights[i][1]);
        color3 c = ((main.pixels[i] + pixel_weights[i][1] * main.backdrop.cast<double>()) / w_sum).cast<float>();
        if (tonemapper) {
            c = (*tonemapper)(c);
        }
        for (int d = 0; d < 3; ++d) {
            c[d] = linear_to_srgb(c[d]);
        }
        for (int d = 0; d < 3; ++d) {
            buf[3 * i + d] = (uint8_t)std::clamp(std::floor(c[d] * 255.0f), 0.0f, 255.0f);
        }
        // TODO: linear to sRGB?
    });

    fs::path save_path = path_prefix;
    save_path += path_postfix.empty() ? ".png" : string_format("_%s.png", path_postfix.c_str());
    ks::save_to_png((const std::byte *)buf.get(), width, height, 3, save_path);

    auto save_plane = [&]<int N>(const AOVPlane<float, N> &plane) {
        parallel_for((uint32_t)plane.pixels.size(), [&](uint32_t i) {
            double w_sum = (pixel_weights[i][0] + pixel_weights[i][1]);
            color<N> c = ((plane.pixels[i] + pixel_weights[i][1] * plane.backdrop) / w_sum).template cast<float>();
            for (int d = 0; d < N; ++d) {
                c[d] = linear_to_srgb(c[d]);
            }
            for (int d = 0; d < N; ++d) {
                buf[N * i + d] = (uint8_t)std::clamp(std::floor(c[d] * 255.0f), 0.0f, 255.0f);
            }
        });
        fs::path save_path = path_prefix;
        save_path += path_postfix.empty() ? string_format("_%s.png", plane.name.c_str())
                                          : string_format("_%s_%s.png", plane.name.c_str(), path_postfix.c_str());
        ks::save_to_png((const std::byte *)buf.get(), width, height, N, save_path);
    };

    for (const auto &plane : aovs1) {
        save_plane(plane);
    }
    for (const auto &plane : aovs2) {
        save_plane(plane);
    }
    for (const auto &plane : aovs3) {
        save_plane(plane);
    }
    for (const auto &plane : aovs4) {
        save_plane(plane);
    }
}

void RenderTarget2::composite_and_save_to_exr(const fs::path &path_prefix, const std::string &path_postfix,
                                              bool write_fp16) const
{
    size_t component_size = (write_fp16 ? sizeof(short) : sizeof(float));
    auto buf = std::make_unique<std::byte[]>(width * height * 4 * component_size); // allocate for max 4 components.

    parallel_for((uint32_t)main.pixels.size(), [&](uint32_t i) {
        uint32_t w_sum = (pixel_weights[i][0] + pixel_weights[i][1]);
        double w = w_sum == 0 ? 0.0f : (double)pixel_weights[i][0] / (double)(w_sum);
        color3d fg =
            pixel_weights[i][0] == 0 ? color3d::Zero() : color3d(main.pixels[i] / (double)(pixel_weights[i][0]));
        color3 c = lerp(main.backdrop.cast<double>(), fg, w).cast<float>();

        std::byte *ptr = &buf.get()[i * 3 * component_size];
        for (int d = 0; d < 3; ++d) {
            if (write_fp16) {
                short h = convert_float_to_half(c[d]);
                (*(short *)ptr) = h;
            } else {
                (*(float *)ptr) = c[d];
            }
            ptr += component_size;
        }
    });

    fs::path save_path = path_prefix;
    save_path += path_postfix.empty() ? ".exr" : string_format("_%s.exr", path_postfix.c_str());
    ks::save_to_exr((const std::byte *)buf.get(), write_fp16, width, height, 3, save_path);

    auto save_plane = [&]<int N>(const AOVPlane<float, N> &plane) {
        parallel_for((uint32_t)plane.pixels.size(), [&](uint32_t i) {
            uint32_t w_sum = (pixel_weights[i][0] + pixel_weights[i][1]);
            double w = w_sum == 0 ? 0.0f : (double)pixel_weights[i][0] / (double)(w_sum);
            color<N> fg =
                pixel_weights[i][0] == 0 ? color<N>::Zero() : color<N>(plane.pixels[i] / (double)(pixel_weights[i][0]));
            color<N> c = lerp(plane.backdrop, fg, (float)w);

            std::byte *ptr = &buf.get()[i * N * component_size];
            for (int d = 0; d < N; ++d) {
                if (write_fp16) {
                    short h = convert_float_to_half(c[d]);
                    (*(short *)ptr) = h;
                } else {
                    (*(float *)ptr) = c[d];
                }
                ptr += component_size;
            }
        });
        fs::path save_path = path_prefix;
        save_path += path_postfix.empty() ? string_format("_%s.exr", plane.name.c_str())
                                          : string_format("_%s_%s.exr", plane.name.c_str(), path_postfix.c_str());
        ks::save_to_exr((const std::byte *)buf.get(), write_fp16, width, height, N, save_path);
    };

    for (const auto &plane : aovs1) {
        save_plane(plane);
    }
    for (const auto &plane : aovs2) {
        save_plane(plane);
    }
    for (const auto &plane : aovs3) {
        save_plane(plane);
    }
    for (const auto &plane : aovs4) {
        save_plane(plane);
    }
}

} // namespace ks