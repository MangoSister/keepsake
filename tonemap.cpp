#include "tonemap.h"
#include "image_util.h"
#include "log_util.h"
#include "render_target.h"

namespace ks
{

void tonemap(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    fs::path input_path = args.load_path("input_path");
    fs::path output_path = args.load_path("output_path");
    std::string ext = input_path.extension().string();
    int width = 0, height = 0;
    std::unique_ptr<float[]> float_data;
    if (ext == ".exr") {
        float_data = load_from_exr(input_path, 3, width, height);
    } else if (ext == ".hdr") {
        float_data = load_from_hdr(input_path, 3, width, height);
    } else {
        get_default_logger().error("Invalid image extension {}", ext.c_str());
        std::abort();
    }
    std::unique_ptr<ToneMapper> tone_mapper = create_tone_mapper(args["tone_mapper"]);
    parallel_tile_2d(width, height, [&](int x, int y) {
        color3 *color_ptr = reinterpret_cast<color3 *>(float_data.get());
        color_ptr[y * width + x] = (*tone_mapper)(color_ptr[y * width + x]);
    });

    save_to_exr((const std::byte *)float_data.get(), false, width, height, 3, output_path);
}

} // namespace ks