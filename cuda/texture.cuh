#pragma once

#include "basic.cuh"
#include "memory_low_level.cuh"
#include "span.cuh"
#include "vecmath.cuh"
#include <cstddef>
#include <vector>

namespace ksc
{

struct Texture2D
{
    Texture2D() = default;
    Texture2D(size_t width, size_t height, const cudaChannelFormatDesc &format_desc, const cudaTextureDesc &tex_desc,
              ksc::span<const std::byte> data);
    ~Texture2D();

    Texture2D(const Texture2D &) = delete;
    Texture2D &operator=(const Texture2D &) = delete;

    Texture2D(Texture2D &&);
    Texture2D &operator=(Texture2D &&);

    CUDA_HOST_DEVICE operator cudaTextureObject_t() const { return tex_obj; }

    size_t width = 0;
    size_t height = 0;
    cudaChannelFormatDesc format_desc{};
    cudaTextureDesc tex_desc{};

    cudaArray_t arr = nullptr;
    cudaTextureObject_t tex_obj = 0;
};

struct Texture3D
{
    Texture3D() = default;
    Texture3D(size_t width, size_t height, size_t depth, const cudaChannelFormatDesc &format_desc,
              const cudaTextureDesc &tex_desc, ksc::span<const std::byte> data);
    ~Texture3D();

    Texture3D(const Texture3D &) = delete;
    Texture3D &operator=(const Texture3D &) = delete;

    Texture3D(Texture3D &&);
    Texture3D &operator=(Texture3D &&);

    CUDA_HOST_DEVICE operator cudaTextureObject_t() const { return tex_obj; }

    std::vector<std::byte> write_to_host_linear() const;

    size_t width = 0;
    size_t height = 0;
    size_t depth = 0;
    cudaChannelFormatDesc format_desc{};
    cudaTextureDesc tex_desc{};

    cudaArray_t arr = nullptr;
    cudaTextureObject_t tex_obj = 0;
};

struct Surface2D
{
    Surface2D() = default;
    Surface2D(size_t width, size_t height, const cudaChannelFormatDesc &format_desc,
              ksc::span<const std::byte> data = {});
    ~Surface2D();

    Surface2D(const Surface2D &) = delete;
    Surface2D &operator=(const Surface2D &) = delete;

    Surface2D(Surface2D &&);
    Surface2D &operator=(Surface2D &&);

    CUDA_HOST_DEVICE operator cudaSurfaceObject_t() const { return surf_obj; }

    std::vector<std::byte> write_to_host_linear() const;

    size_t width = 0;
    size_t height = 0;
    cudaChannelFormatDesc format_desc{};

    cudaArray_t arr = nullptr;
    cudaSurfaceObject_t surf_obj = 0;
};

struct LowLevelLinearImage2D
{
    LowLevelLinearImage2D() = default;
    LowLevelLinearImage2D(size_t width, size_t height, const cudaChannelFormatDesc &format_desc,
                          ksc::span<const std::byte> data = {});
    ~LowLevelLinearImage2D();

    LowLevelLinearImage2D(const LowLevelLinearImage2D &) = delete;
    LowLevelLinearImage2D &operator=(const LowLevelLinearImage2D &) = delete;

    LowLevelLinearImage2D(LowLevelLinearImage2D &&);
    LowLevelLinearImage2D &operator=(LowLevelLinearImage2D &&);

    CUDA_DEVICE ksc::color3 read_rgb(ksc::vec2i pixel) const;
    CUDA_DEVICE void write_rgb(ksc::vec2i pixel, ksc::color3 color);

    CUDA_DEVICE ksc::vec2 read_vec2(ksc::vec2i pixel) const;
    CUDA_DEVICE void write_vec2(ksc::vec2i pixel, ksc::vec2 v);

    std::vector<std::byte> write_to_host_linear() const;

    size_t width = 0;
    size_t height = 0;
    cudaChannelFormatDesc format_desc{};
    ksc::CudaShareableLowLevelMemory m;
};

} // namespace ksc