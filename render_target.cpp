#include "render_target.h"
#include "image_util.h"

void RenderTarget::save_to_hdr(const fs::path &path) const
{
    ::save_to_hdr((const float *)pixels.data(), width, height, 3, path);
}

void RenderTarget::save_to_exr(const fs::path &path) const
{
    ::save_to_exr((const float *)pixels.data(), width, height, 3, path);
}
