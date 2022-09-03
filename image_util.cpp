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

std::unique_ptr<float[]> load_from_ldr(const fs::path &path, int c, int &w, int &h, ColorSpace color_space)
{
    int comp;
    stbi_uc *loaded = stbi_load(path.string().c_str(), &w, &h, &comp, c);
    ASSERT(loaded);
    std::unique_ptr<float[]> float_data = std::make_unique<float[]>(w * h * c);
    int n = w * h * c;
    for (int i = 0; i < n; ++i) {
        float_data[i] = (float)loaded[i] / 255.0f;
    }
    if (color_space == ColorSpace::sRGB) {
        for (int i = 0; i < n; ++i) {
            float_data[i] = srgb_to_linear(float_data[i]);
        }
    }
    stbi_image_free(loaded);
    return float_data;
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