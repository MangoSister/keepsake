#include "image_util.h"
#include "assertion.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace ks
{

std::unique_ptr<std::byte[]> load_from_ldr(const fs::path &path, int c, int &w, int &h, ColorSpace src_colorspace)
{
    int comp;
    stbi_uc *loaded = stbi_load(path.string().c_str(), &w, &h, &comp, c);
    ASSERT(loaded);
    std::unique_ptr<std::byte[]> copy = std::make_unique<std::byte[]>(w * h * c);
    std::copy(reinterpret_cast<const std::byte *>(loaded), reinterpret_cast<const std::byte *>(loaded) + (w * h * c),
              copy.get());
    stbi_image_free(loaded);
    if (src_colorspace == ColorSpace::sRGB) {
        // TODO: converting srgb to linear for in u8 may have precision problem on the dark end...
        // https://blog.demofox.org/2018/03/10/dont-convert-srgb-u8-to-linear-u8/
        int n = w * h * c;
        for (int i = 0; i < n; ++i) {
            float f32 = (float)(*reinterpret_cast<uint8_t *>(&copy[i])) / 255.0f;
            float linear_f32 = srgb_to_linear(f32);
            uint8_t linear_u8 = (uint8_t)std::floor(linear_f32 * 255.0f);
            (*reinterpret_cast<uint8_t *>(&copy[i])) = linear_u8;
        }
    }
    return copy;
}

std::unique_ptr<float[]> load_from_ldr_to_float(const fs::path &path, int c, int &w, int &h, ColorSpace src_colorspace)
{
    int comp;
    stbi_uc *loaded = stbi_load(path.string().c_str(), &w, &h, &comp, c);
    ASSERT(loaded);
    std::unique_ptr<float[]> float_data = std::make_unique<float[]>(w * h * c);
    int n = w * h * c;
    for (int i = 0; i < n; ++i) {
        float_data[i] = (float)loaded[i] / 255.0f;
    }
    stbi_image_free(loaded);
    if (src_colorspace == ColorSpace::sRGB) {
        for (int i = 0; i < n; ++i) {
            float_data[i] = srgb_to_linear(float_data[i]);
        }
    }
    return float_data;
}

void save_to_png(const std::byte *data, int w, int h, int c, const fs::path &path)
{
    int stride_bytes = w * c;
    if (!stbi_write_png(path.string().c_str(), w, h, c, data, stride_bytes)) {
        printf("save_to_png failed (%s)\n", path.string().c_str());
    }
}

std::unique_ptr<float[]> load_from_hdr(const fs::path &path, int c, int &w, int &h)
{
    int comp;
    float *loaded = stbi_loadf(path.string().c_str(), &w, &h, &comp, c);
    ASSERT(loaded);
    std::unique_ptr<float[]> float_data = std::make_unique<float[]>(w * h * c);
    memcpy(float_data.get(), loaded, sizeof(float) * w * h * c);
    stbi_image_free(loaded);
    return float_data;
}

void save_to_hdr(const float *data, int w, int h, int c, const fs::path &path)
{
    if (!stbi_write_hdr(path.string().c_str(), w, h, c, data)) {
        printf("save_to_hdr failed (%s)\n", path.string().c_str());
    }
}

std::unique_ptr<float[]> load_from_exr(const fs::path &path, int c, int &w, int &h)
{
    // width * height * RGBA
    float *loaded;
    const char *err = nullptr;

    int ret = LoadEXR(&loaded, &w, &h, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "load_from_exr error: %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }
        return nullptr;
    }

    std::unique_ptr<float[]> float_data = std::make_unique<float[]>(w * h * c);
    if (c == 4) {
        memcpy(float_data.get(), loaded, sizeof(float) * w * h * c);
    } else {
        int n_pixels = w * h;
        for (int i = 0; i < n_pixels; ++i) {
            for (int j = 0; j < c; ++j)
                float_data[c * i + j] = loaded[4 * i + j];
        }
    }
    free(loaded);
    return float_data;
}

void save_to_exr(const std::byte *data, bool half, int w, int h, int c, const fs::path &path)
{
    ASSERT(c == 3 || c == 4, "save_to_exr only supports rgb or rgba");
    EXRHeader header;
    InitEXRHeader(&header);
    header.num_channels = c;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);

    EXRImage image;
    InitEXRImage(&image);
    image.width = w;
    image.height = h;
    image.num_channels = c;

    // Split channels
    int channel_size = half ? 2 : 4;
    std::vector<std::byte> images[4];
    for (int i = 0; i < c; ++i)
        images[i].resize(w * h * c * channel_size);
    for (int i = 0; i < w * h; i++)
        for (int j = 0; j < c; ++j)
            for (int k = 0; k < channel_size; ++k)
                images[j][i * channel_size + k] = data[channel_size * (c * i + j) + k];

    std::byte *image_ptr[4];
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    if (c == 4) {
        image_ptr[0] = images[3].data(); // A
        image_ptr[1] = images[2].data(); // B
        image_ptr[2] = images[1].data(); // G
        image_ptr[3] = images[0].data(); // R

        strncpy(header.channels[0].name, "A", 255);
        header.channels[0].name[strlen("A")] = '\0';
        strncpy(header.channels[1].name, "B", 255);
        header.channels[1].name[strlen("B")] = '\0';
        strncpy(header.channels[2].name, "G", 255);
        header.channels[2].name[strlen("G")] = '\0';
        strncpy(header.channels[3].name, "R", 255);
        header.channels[3].name[strlen("R")] = '\0';
    } else {
        image_ptr[0] = images[2].data(); // B
        image_ptr[1] = images[1].data(); // G
        image_ptr[2] = images[0].data(); // R

        strncpy(header.channels[0].name, "B", 255);
        header.channels[0].name[strlen("B")] = '\0';
        strncpy(header.channels[1].name, "G", 255);
        header.channels[1].name[strlen("G")] = '\0';
        strncpy(header.channels[2].name, "R", 255);
        header.channels[2].name[strlen("R")] = '\0';
    }
    image.images = (unsigned char **)image_ptr;

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
        header.pixel_types[i] = (half ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT); // pixel type of input image
        header.requested_pixel_types[i] =
            (half ? TINYEXR_PIXELTYPE_HALF
                  : TINYEXR_PIXELTYPE_FLOAT); // pixel type of output image to be stored in .EXR
        // TODO: maybe also support saving to half
    }

    const char *err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, path.string().c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "save_to_exr error: %s\n", err);
        FreeEXRErrorMessage(err); // free's buffer for an error message
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

} // namespace ks