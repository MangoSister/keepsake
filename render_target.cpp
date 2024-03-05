#include "render_target.h"
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

} // namespace ks