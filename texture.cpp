#include "texture.h"
#include "assertion.h"
#include "file_util.h"
#include "image_util.h"
#include "parallel.h"
#include <array>
#include <lz4.h>

namespace ks
{

constexpr int byte_stride(TextureDataType data_type)
{
    switch (data_type) {
    case TextureDataType::u8:
        return 1;
    case TextureDataType::u16:
        return 2;
    case TextureDataType::f32:
    default:
        return 4;
    }
}

Texture::Texture(int width, int height, int num_channels, TextureDataType data_type, ColorSpace color_space)
    : width(width), height(height), num_channels(num_channels), data_type(data_type), color_space(color_space)
{
    int stride = byte_stride(data_type) * num_channels;
    int levels = 1;
    mips.resize(levels);
    mips[0] = BlockedArray<std::byte>(width, height, stride);
}

Texture::Texture(const std::byte *bytes, int width, int height, int num_channels, TextureDataType data_type,
                 ColorSpace color_space, bool build_mipmaps)
    : width(width), height(height), num_channels(num_channels), data_type(data_type), color_space(color_space)
{
    int stride = byte_stride(data_type) * num_channels;
    // https://www.nvidia.com/en-us/drivers/np2-mipmapping/
    // NPOT mipmapping with rounding-up.
    int levels = build_mipmaps ? (1 + (int)std::ceil(std::log2(std::max(width, height)))) : 1;
    mips.resize(levels);
    constexpr int min_parallel_res = 256 * 256;
    for (int l = 0; l < levels; ++l) {
        if (l == 0) {
            bool parallel = width * height >= min_parallel_res;
            mips[l] = BlockedArray(width, height, stride, bytes, 2, parallel);
        } else {
            mips[l] = BlockedArray<std::byte>(width, height, stride);
            int last_width = mips[l - 1].ures;
            int last_height = mips[l - 1].vres;
            bool round_up_width = (last_width == width * 2 - 1);
            bool round_up_height = (last_height == height * 2 - 1);
            if (last_width > 1 && round_up_width && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0 && y0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                    }
                    if (y0 > 0) {
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    if (x2 < last_width && y0 > 0) {
                        fetch_as_float(x2, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w0y * v_span[c];
                    }
                    if (x0 > 0) {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w1y * v_span[c];
                    }
                    if (x0 > 0 && y2 < last_height) {
                        fetch_as_float(x0, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w2y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x1, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w2y * v_span[c];
                    }
                    if (x2 < last_width && y2 < last_height) {
                        fetch_as_float(x2, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    constexpr float w0x = 0.5f;
                    constexpr float w1x = 0.5f;
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (y0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x0, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w2y * v_span[c];
                        fetch_as_float(x1, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && round_up_width && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;
                    constexpr float w0y = 0.5f;
                    constexpr float w1y = 0.5f;
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0) {
                        fetch_as_float(x0, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w0y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w0y * v_span[c];
                    }
                    if (x0 > 0) {
                        fetch_as_float(x0, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * w1y * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * w1y * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * w1y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x0, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x1, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x0, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    fetch_as_float(x1, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.25f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width == 1 && last_height > 1 && round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0y = (float)y / (float)(last_height);
                    float w1y = (float)height / (float)(last_height);
                    float w2y = (float)(height - y - 1) / (float)(last_height);
                    int y0 = 2 * y - 1;
                    int y1 = 2 * y;
                    int y2 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (y0 > 0) {
                        fetch_as_float(x, y0, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0y * v_span[c];
                    }
                    {
                        fetch_as_float(x, y1, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1y * v_span[c];
                    }
                    if (y2 < last_height) {
                        fetch_as_float(x, y2, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2y * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width == 1 && last_height > 1 && !round_up_height) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int y0 = 2 * y;
                    int y1 = 2 * y + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x, y0, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    fetch_as_float(x, y1, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && round_up_width && last_height == 1) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    float w0x = (float)x / (float)(last_width);
                    float w1x = (float)width / (float)(last_width);
                    float w2x = (float)(width - x - 1) / (float)(last_width);
                    int x0 = 2 * x - 1;
                    int x1 = 2 * x;
                    int x2 = 2 * x + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    if (x0 > 0) {
                        fetch_as_float(x0, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w0x * v_span[c];
                    }
                    {
                        fetch_as_float(x1, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w1x * v_span[c];
                    }
                    if (x2 < last_width) {
                        fetch_as_float(x2, y, l - 1, v_span);
                        for (int c = 0; c < num_channels; ++c)
                            v_sum_span[c] += w2x * v_span[c];
                    }
                    set_from_float(x, y, l, v_sum_span);
                });
            } else if (last_width > 1 && !round_up_width && last_height == 1) {
                parallel_tile_2d(width, height, [&](int x, int y) {
                    int x0 = 2 * x;
                    int x1 = 2 * x + 1;

                    VLA(v_sum, float, num_channels);
                    std::span<float> v_sum_span(v_sum, v_sum + num_channels);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] = 0.0f;
                    VLA(v, float, num_channels);
                    std::span<float> v_span(v, v + num_channels);

                    fetch_as_float(x0, y, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    fetch_as_float(x1, y, l - 1, v_span);
                    for (int c = 0; c < num_channels; ++c)
                        v_sum_span[c] += 0.5f * v_span[c];
                    set_from_float(x, y, l, v_sum_span);
                });
            }
        }
        width = std::max(1, (width + 1) / 2);
        height = std::max(1, (height + 1) / 2);
    }
}

Texture::Texture(std::span<const std::byte *> mip_bytes, int width, int height, int num_channels,
                 TextureDataType data_type, ColorSpace color_space)
    : width(width), height(height), num_channels(num_channels), data_type(data_type), color_space(color_space)
{
    int stride = byte_stride(data_type) * num_channels;
    int levels = (int)mip_bytes.size();
    mips.resize(levels);
    for (int i = 0; i < levels; ++i) {
        constexpr int min_parallel_res = 256 * 256;
        bool parallel = width * height >= min_parallel_res;
        mips[i] = BlockedArray(width, height, stride, mip_bytes[i], 2, parallel);
        // Assuming rounding up.
        width = std::max(1, (width + 1) / 2);
        height = std::max(1, (height + 1) / 2);
    }
}

// TODO: fixed-point math for 8bit textures
// TODO: SIMD

static constexpr float srgb_u8_to_linear_f32_lut[256] = {
    0.0000000000, 0.0003035270, 0.0006070540, 0.0009105810, 0.0012141080, 0.0015176350, 0.0018211619, 0.0021246888,
    0.0024282159, 0.0027317430, 0.0030352699, 0.0033465356, 0.0036765069, 0.0040247170, 0.0043914421, 0.0047769533,
    0.0051815170, 0.0056053917, 0.0060488326, 0.0065120910, 0.0069954102, 0.0074990317, 0.0080231922, 0.0085681248,
    0.0091340570, 0.0097212177, 0.0103298230, 0.0109600937, 0.0116122449, 0.0122864870, 0.0129830306, 0.0137020806,
    0.0144438436, 0.0152085144, 0.0159962922, 0.0168073755, 0.0176419523, 0.0185002182, 0.0193823613, 0.0202885624,
    0.0212190095, 0.0221738834, 0.0231533647, 0.0241576303, 0.0251868572, 0.0262412224, 0.0273208916, 0.0284260381,
    0.0295568332, 0.0307134409, 0.0318960287, 0.0331047624, 0.0343398079, 0.0356013142, 0.0368894450, 0.0382043645,
    0.0395462364, 0.0409151986, 0.0423114114, 0.0437350273, 0.0451862030, 0.0466650836, 0.0481718220, 0.0497065634,
    0.0512694679, 0.0528606549, 0.0544802807, 0.0561284944, 0.0578054339, 0.0595112406, 0.0612460710, 0.0630100295,
    0.0648032799, 0.0666259527, 0.0684781820, 0.0703601092, 0.0722718611, 0.0742135793, 0.0761853904, 0.0781874284,
    0.0802198276, 0.0822827145, 0.0843762159, 0.0865004659, 0.0886556059, 0.0908417329, 0.0930589810, 0.0953074843,
    0.0975873619, 0.0998987406, 0.1022417471, 0.1046164930, 0.1070231125, 0.1094617173, 0.1119324341, 0.1144353822,
    0.1169706732, 0.1195384338, 0.1221387982, 0.1247718409, 0.1274376959, 0.1301364899, 0.1328683347, 0.1356333494,
    0.1384316236, 0.1412633061, 0.1441284865, 0.1470272839, 0.1499598026, 0.1529261619, 0.1559264660, 0.1589608639,
    0.1620294005, 0.1651322246, 0.1682693958, 0.1714410931, 0.1746473908, 0.1778884083, 0.1811642349, 0.1844749898,
    0.1878207624, 0.1912016720, 0.1946178079, 0.1980693042, 0.2015562356, 0.2050787061, 0.2086368501, 0.2122307271,
    0.2158605307, 0.2195262313, 0.2232279778, 0.2269658893, 0.2307400703, 0.2345506549, 0.2383976579, 0.2422811985,
    0.2462013960, 0.2501583695, 0.2541521788, 0.2581829131, 0.2622507215, 0.2663556635, 0.2704978585, 0.2746773660,
    0.2788943350, 0.2831487954, 0.2874408960, 0.2917706966, 0.2961383164, 0.3005438447, 0.3049873710, 0.3094689548,
    0.3139887452, 0.3185468316, 0.3231432438, 0.3277781308, 0.3324515820, 0.3371636569, 0.3419144452, 0.3467040956,
    0.3515326977, 0.3564002514, 0.3613068759, 0.3662526906, 0.3712377846, 0.3762622178, 0.3813261092, 0.3864295185,
    0.3915725648, 0.3967553079, 0.4019778669, 0.4072403014, 0.4125427008, 0.4178851545, 0.4232677519, 0.4286905527,
    0.4341537058, 0.4396572411, 0.4452012479, 0.4507858455, 0.4564110637, 0.4620770514, 0.4677838385, 0.4735315442,
    0.4793202281, 0.4851499796, 0.4910208881, 0.4969330430, 0.5028865933, 0.5088814497, 0.5149177909, 0.5209956765,
    0.5271152258, 0.5332764983, 0.5394796133, 0.5457245708, 0.5520114899, 0.5583404899, 0.5647116303, 0.5711249113,
    0.5775805116, 0.5840784907, 0.5906189084, 0.5972018838, 0.6038274169, 0.6104956269, 0.6172066331, 0.6239604354,
    0.6307572126, 0.6375969648, 0.6444797516, 0.6514056921, 0.6583748460, 0.6653873324, 0.6724432111, 0.6795425415,
    0.6866854429, 0.6938719153, 0.7011020184, 0.7083759308, 0.7156936526, 0.7230552435, 0.7304608822, 0.7379105687,
    0.7454043627, 0.7529423237, 0.7605246305, 0.7681512833, 0.7758223414, 0.7835379243, 0.7912980318, 0.7991028428,
    0.8069523573, 0.8148466945, 0.8227858543, 0.8307699561, 0.8387991190, 0.8468732834, 0.8549926877, 0.8631572723,
    0.8713672161, 0.8796223402, 0.8879231811, 0.8962693810, 0.9046613574, 0.9130986929, 0.9215820432, 0.9301108718,
    0.9386858940, 0.9473065734, 0.9559735060, 0.9646862745, 0.9734454751, 0.9822505713, 0.9911022186, 1.0000000000};

void Texture::fetch_as_float(int x, int y, int level, std::span<float> out) const
{
    const std::byte *bytes = fetch_raw(x, y, level);
    int nc = std::min(num_channels, (int)out.size());
    switch (data_type) {
    case TextureDataType::u8: {
        const uint8_t *u8_data = reinterpret_cast<const uint8_t *>(bytes);
        switch (color_space) {
        case ColorSpace::sRGB:
            for (int c = 0; c < nc; ++c) {
                out[c] = srgb_u8_to_linear_f32_lut[u8_data[c]];
            }
            break;
        case ColorSpace::Linear:
        default: {
            for (int c = 0; c < nc; ++c) {
                out[c] = (float)u8_data[c] / 255.0f;
            }
            break;
        }
        }
        break;
    }
    case TextureDataType::u16: {
        const uint16_t *u16_data = reinterpret_cast<const uint16_t *>(bytes);
        for (int c = 0; c < nc; ++c) {
            out[c] = (float)u16_data[c] / 65535.0f;
        }
        switch (color_space) {
        case ColorSpace::sRGB:
            for (int c = 0; c < nc; ++c) {
                out[c] = srgb_to_linear(out[c]);
            }
            break;
        case ColorSpace::Linear:
        default:
            break;
        }
        break;
    }
    case TextureDataType::f32:
    default: {
        const float *f32_data = reinterpret_cast<const float *>(bytes);
        std::copy(f32_data, f32_data + nc, out.data());
        switch (color_space) {
        case ColorSpace::sRGB:
            for (int c = 0; c < nc; ++c) {
                out[c] = srgb_to_linear(out[c]);
            }
            break;
        case ColorSpace::Linear:
        default:
            break;
        }
        break;
    }
    }
}

void Texture::set_from_float(int x, int y, int level, std::span<const float> in)
{
    ASSERT(in.size() == num_channels);
    std::byte *bytes = mips[level].fetch_multi(x, y);
    switch (data_type) {
    case TextureDataType::u8: {
        uint8_t *u8_data = reinterpret_cast<uint8_t *>(bytes);
        for (int c = 0; c < num_channels; ++c) {
            float f32_value;
            switch (color_space) {
            case ColorSpace::sRGB:
                f32_value = linear_to_srgb(in[c]);
                break;
            case ColorSpace::Linear:
            default:
                f32_value = in[c];
                break;
            }
            u8_data[c] = (uint8_t)std::floor(f32_value * 255.0f);
        }
        break;
    }
    case TextureDataType::u16: {
        uint16_t *u16_data = reinterpret_cast<uint16_t *>(bytes);
        for (int c = 0; c < num_channels; ++c) {
            float f32_value;
            switch (color_space) {
            case ColorSpace::sRGB:
                f32_value = linear_to_srgb(in[c]);
                break;
            case ColorSpace::Linear:
            default:
                f32_value = in[c];
                break;
            }
            u16_data[c] = (uint16_t)std::floor(f32_value * 65535.0f);
        }
        break;
    }
    case TextureDataType::f32:
    default: {
        float *f32_data = reinterpret_cast<float *>(bytes);
        std::copy(in.data(), in.data() + num_channels, f32_data);
        switch (color_space) {
        case ColorSpace::sRGB:
            for (int c = 0; c < num_channels; ++c) {
                f32_data[c] = linear_to_srgb(f32_data[c]);
            }
            break;
        case ColorSpace::Linear:
        default:
            break;
        }
        break;
    }
    }
}

static inline int wrap(int x, int dim, TextureWrapMode mode)
{
    switch (mode) {
    case TextureWrapMode::Repeat:
        return mod(x, dim);
    case TextureWrapMode::Clamp:
        return std::clamp(x, 0, dim - 1);
    default:
        ASSERT(false, "Invalid texture wrap mode.");
        return 0;
    }
}

color4 TextureSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy) const
{
    ASSERT(texture.num_channels <= 4, "Texture has more than 4 channels");
    color4 out = color4::Zero();
    this->operator()(texture, uv, duvdxy, {out.data(), 4});
    return out;
}

void TextureSampler::bilinear(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const
{
    float u = uv[0] * texture.mips[level].ures - 0.5f;
    float v = uv[1] * texture.mips[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    u0 = wrap(u0, texture.mips[level].ures, wrap_mode_u);
    v0 = wrap(v0, texture.mips[level].vres, wrap_mode_v);
    int u1 = wrap(u0 + 1, texture.mips[level].ures, wrap_mode_u);
    int v1 = wrap(v0 + 1, texture.mips[level].vres, wrap_mode_v);

    float w00 = (1 - du) * (1 - dv);
    float w10 = du * (1 - dv);
    float w01 = (1 - du) * dv;
    float w11 = du * dv;

    size_t nc = std::min((size_t)texture.num_channels, out.size());

    VLA(out00, float, nc);
    texture.fetch_as_float(u0, v0, level, {out00, nc});
    VLA(out10, float, nc);
    texture.fetch_as_float(u1, v0, level, {out10, nc});
    VLA(out01, float, nc);
    texture.fetch_as_float(u0, v1, level, {out01, nc});
    VLA(out11, float, nc);
    texture.fetch_as_float(u1, v1, level, {out11, nc});

    for (int i = 0; i < nc; ++i)
        out[i] = w00 * out00[i] + w10 * out10[i] + w01 * out01[i] + w11 * out11[i];
}

void NearestSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float u = uv[0] * texture.mips[0].ures - 0.5f;
    float v = uv[1] * texture.mips[0].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    u0 = wrap(u0, texture.mips[0].ures, wrap_mode_u);
    v0 = wrap(v0, texture.mips[0].vres, wrap_mode_v);

    texture.fetch_as_float(u0, v0, 0, out);
}

void LinearSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float width = duvdxy.cwiseAbs().maxCoeff();
    float level = texture.levels() - 1 + std::log2(std::max(width, (float)1e-8));
    if (level < 0 || texture.levels() == 1) {
        bilinear(texture, 0, uv, out);
    } else if (level >= texture.levels() - 1) {
        texture.fetch_as_float(0, 0, texture.levels() - 1, out);
    } else {
        int ilevel = (int)std::floor(level);
        float delta = level - ilevel;
        size_t nc = std::min((size_t)texture.num_channels, out.size());
        VLA(out0, float, nc);
        bilinear(texture, ilevel, uv, {out0, nc});
        VLA(out1, float, nc);
        bilinear(texture, ilevel + 1, uv, {out1, nc});
        for (int i = 0; i < nc; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

void CubicSampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    float width = duvdxy.cwiseAbs().maxCoeff();
    float level = texture.levels() - 1 + std::log2(std::max(width, (float)1e-8));
    if (level < 0 || texture.levels() == 1) {
        bicubic(texture, 0, uv, out);
    } else if (level >= texture.levels() - 1) {
        texture.fetch_as_float(0, 0, texture.levels() - 1, out);
    } else {
        int ilevel = (int)std::floor(level);
        float delta = level - ilevel;
        size_t nc = std::min((size_t)texture.num_channels, out.size());
        VLA(out0, float, nc);
        bicubic(texture, ilevel, uv, {out0, nc});
        VLA(out1, float, nc);
        bicubic(texture, ilevel + 1, uv, {out1, nc});
        for (int i = 0; i < nc; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

inline vec4 powers(float x) { return vec4(x * x * x, x * x, x, 1.0f); }

inline void spline(float x, int nc, const float *c0, const float *c1, const float *c2, const float *c3, const vec4 &ca,
                   const vec4 &cb, float *out)
{
    float a0 = cb.dot(powers(x + 1.0));
    float a1 = ca.dot(powers(x));
    float a2 = ca.dot(powers(1.0 - x));
    float a3 = cb.dot(powers(2.0 - x));
    for (int i = 0; i < nc; ++i) {
        out[i] = c0[i] * a0 + c1[i] * a1 + c2[i] * a2 + c3[i] * a3;
    }
}

void CubicSampler::bicubic(const Texture &texture, int level, const vec2 &uv, std::span<float> out) const
{
    float u = uv[0] * texture.mips[level].ures - 0.5f;
    float v = uv[1] * texture.mips[level].vres - 0.5f;
    int u0 = (int)std::floor(u);
    int v0 = (int)std::floor(v);
    float du = u - u0;
    float dv = v - v0;

    vec4i us;
    vec4i vs;
    for (int i = 0; i < 4; ++i) {
        us[i] = wrap(u0 + i - 1, texture.mips[level].ures, wrap_mode_u);
        vs[i] = wrap(v0 + i - 1, texture.mips[level].vres, wrap_mode_v);
    }

    size_t nc = std::min((size_t)texture.num_channels, out.size());
    VLA(rows, float, nc * 4);
    VLA(cols, float, nc * 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x)
            texture.fetch_as_float(us[x], vs[y], level, {&cols[x * nc], nc});
        const float *c0 = &cols[0];
        const float *c1 = &cols[1 * nc];
        const float *c2 = &cols[2 * nc];
        const float *c3 = &cols[3 * nc];
        spline(du, nc, c0, c1, c2, c3, ca, cb, &rows[y * nc]);
    }

    const float *c0 = &rows[0];
    const float *c1 = &rows[1 * nc];
    const float *c2 = &rows[2 * nc];
    const float *c3 = &rows[3 * nc];
    spline(dv, nc, c0, c1, c2, c3, ca, cb, out.data());
}

void EWASampler::operator()(const Texture &texture, const vec2 &uv, const mat2 &duvdxy, std::span<float> out) const
{
    vec2 duv_major = duvdxy.row(0);
    float len_major = duv_major.norm();
    vec2 duv_minor = duvdxy.row(1);
    float len_minor = duv_minor.norm();
    if (len_major < len_minor) {
        std::swap(duv_major, duv_minor);
        std::swap(len_major, len_minor);
    }
    // Clamp ellipse vector ratio if too large
    if (len_minor * anisotropy < len_major && len_minor > 0) {
        float scale = len_major / (len_minor * anisotropy);
        duv_minor *= scale;
        len_minor *= scale;
    }
    if (len_minor == 0) {
        bilinear(texture, 0, uv, out);
        return;
    }

    // Choose level of detail for EWA lookup and perform EWA filtering
    float level = texture.levels() - 1 + std::log2(std::max(len_minor, (float)1e-8));
    if (level < 0 || texture.levels() == 1) {
        ewa(texture, 0, uv, duv_major, duv_minor, out);
    } else if (level >= texture.levels() - 1) {
        texture.fetch_as_float(0, 0, texture.levels() - 1, out);
    } else {
        int ilevel = (int)std::floor(level);
        float delta = level - ilevel;
        size_t nc = std::min((size_t)texture.num_channels, out.size());
        VLA(out0, float, nc);
        ewa(texture, ilevel, uv, duv_major, duv_minor, {out0, nc});
        VLA(out1, float, nc);
        ewa(texture, ilevel + 1, uv, duv_major, duv_minor, {out1, nc});
        for (int i = 0; i < nc; ++i)
            out[i] = std::lerp(out0[i], out1[i], delta);
    }
}

/*
        for (int i = 0; i < WeightLUTSize; ++i) {
            float alpha = 2;
            float r2 = float(i) / float(WeightLUTSize - 1);
            weightLut[i] = std::exp(-alpha * r2) - std::exp(-alpha);
        }
*/
static constexpr int ewa_lut_size = 128;
static constexpr float ewa_lut[ewa_lut_size] = {0.864664733f,
                                                0.849040031f,
                                                0.83365953f,
                                                0.818519294f,
                                                0.80361563f,
                                                0.788944781f,
                                                0.774503231f,
                                                0.760287285f,
                                                0.746293485f,
                                                0.732518315f,
                                                0.718958378f,
                                                0.705610275f,
                                                0.692470789f,
                                                0.679536581f,
                                                0.666804492f,
                                                0.654271305f,
                                                0.641933978f,
                                                0.629789352f,
                                                0.617834508f,
                                                0.606066525f,
                                                0.594482362f,
                                                0.583079159f,
                                                0.571854174f,
                                                0.560804546f,
                                                0.549927592f,
                                                0.539220572f,
                                                0.528680861f,
                                                0.518305838f,
                                                0.50809288f,
                                                0.498039544f,
                                                0.488143265f,
                                                0.478401601f,
                                                0.468812168f,
                                                0.45937258f,
                                                0.450080454f,
                                                0.440933526f,
                                                0.431929469f,
                                                0.423066139f,
                                                0.414341331f,
                                                0.405752778f,
                                                0.397298455f,
                                                0.388976216f,
                                                0.380784035f,
                                                0.372719884f,
                                                0.364781618f,
                                                0.356967449f,
                                                0.34927541f,
                                                0.341703475f,
                                                0.334249914f,
                                                0.32691282f,
                                                0.319690347f,
                                                0.312580705f,
                                                0.305582166f,
                                                0.298692942f,
                                                0.291911423f,
                                                0.285235822f,
                                                0.278664529f,
                                                0.272195935f,
                                                0.265828371f,
                                                0.259560347f,
                                                0.253390193f,
                                                0.247316495f,
                                                0.241337672f,
                                                0.235452279f,
                                                0.229658857f,
                                                0.223955944f,
                                                0.21834214f,
                                                0.212816045f,
                                                0.207376286f,
                                                0.202021524f,
                                                0.196750447f,
                                                0.191561714f,
                                                0.186454013f,
                                                0.181426153f,
                                                0.176476851f,
                                                0.171604887f,
                                                0.166809067f,
                                                0.162088141f,
                                                0.157441005f,
                                                0.152866468f,
                                                0.148363426f,
                                                0.143930718f,
                                                0.139567271f,
                                                0.135272011f,
                                                0.131043866f,
                                                0.126881793f,
                                                0.122784719f,
                                                0.11875169f,
                                                0.114781633f,
                                                0.11087364f,
                                                0.107026696f,
                                                0.103239879f,
                                                0.0995122194f,
                                                0.0958427936f,
                                                0.0922307223f,
                                                0.0886750817f,
                                                0.0851749927f,
                                                0.0817295909f,
                                                0.0783380121f,
                                                0.0749994367f,
                                                0.0717130303f,
                                                0.0684779733f,
                                                0.0652934611f,
                                                0.0621587038f,
                                                0.0590728968f,
                                                0.0560353249f,
                                                0.0530452281f,
                                                0.0501018465f,
                                                0.0472044498f,
                                                0.0443523228f,
                                                0.0415447652f,
                                                0.0387810767f,
                                                0.0360605568f,
                                                0.0333825648f,
                                                0.0307464004f,
                                                0.0281514227f,
                                                0.0255970061f,
                                                0.0230824798f,
                                                0.0206072628f,
                                                0.0181707144f,
                                                0.0157722086f,
                                                0.013411209f,
                                                0.0110870898f,
                                                0.0087992847f,
                                                0.0065472275f,
                                                0.00433036685f,
                                                0.0021481365f,
                                                0.f

};

void EWASampler::ewa(const Texture &texture, int level, vec2 uv, vec2 duv_major, vec2 duv_minor,
                     std::span<float> out) const
{
    // Convert EWA coordinates to appropriate scale for level
    uv[0] = uv[0] * texture.mips[level].ures - 0.5f;
    uv[1] = uv[1] * texture.mips[level].vres - 0.5f;
    duv_major[0] *= texture.mips[level].ures;
    duv_major[1] *= texture.mips[level].vres;
    duv_minor[0] *= texture.mips[level].ures;
    duv_minor[1] *= texture.mips[level].vres;

    // Find ellipse coefficients that bound EWA filter region
    float A = sqr(duv_major[1]) + sqr(duv_minor[1]) + 1;
    float B = -2 * (duv_major[0] * duv_major[1] + duv_minor[0] * duv_minor[1]);
    float C = sqr(duv_major[0]) + sqr(duv_minor[0]) + 1;
    float inv_F = 1 / (A * C - sqr(B) * 0.25f);
    A *= inv_F;
    B *= inv_F;
    C *= inv_F;

    // Compute the ellipse's $(s,t)$ bounding box in texture space
    float det = -sqr(B) + 4 * A * C;
    float inv_det = 1 / det;
    float u_sqrt = safe_sqrt(det * C), vSqrt = safe_sqrt(A * det);
    int u0 = std::ceil(uv[0] - 2 * inv_det * u_sqrt);
    int u1 = std::floor(uv[0] + 2 * inv_det * u_sqrt);
    int v0 = std::ceil(uv[1] - 2 * inv_det * vSqrt);
    int v1 = std::floor(uv[1] + 2 * inv_det * vSqrt);

    // Scan over ellipse bound and evaluate quadratic equation to filter image
    size_t nc = std::min((size_t)texture.num_channels, out.size());
    for (int c = 0; c < nc; ++c) {
        out[c] = 0.0f;
    }
    float sum_weights = 0;
    VLA(texel, float, nc);
    for (int iv = v0; iv <= v1; ++iv) {
        float vv = iv - uv[1];
        int iv_wrap = wrap(iv, texture.mips[level].vres, wrap_mode_v);
        for (int iu = u0; iu <= u1; ++iu) {
            float uu = iu - uv[0];
            // Compute squared radius and filter texel if it is inside the ellipse
            float r2 = A * sqr(uu) + B * uu * vv + C * sqr(vv);
            if (r2 < 1) {
                int index = std::min<int>(r2 * ewa_lut_size, ewa_lut_size - 1);
                float weight = ewa_lut[index];
                int iu_wrap = wrap(iu, texture.mips[level].ures, wrap_mode_u);
                texture.fetch_as_float(iu_wrap, iv_wrap, level, {texel, nc});
                for (int c = 0; c < nc; ++c) {
                    out[c] += weight * texel[c];
                }
                sum_weights += weight;
            }
        }
    }

    if (sum_weights > 0.0f) {
        float inv_sum_weight = 1.0f / sum_weights;
        for (int c = 0; c < nc; ++c) {
            out[c] *= inv_sum_weight;
        }
    }
}

std::unique_ptr<Texture> create_texture_from_image(int ch, bool build_mipmaps, ColorSpace src_colorspace,
                                                   const fs::path &path)
{
    std::string ext = path.extension().string();
    int width, height;
    TextureDataType data_type;
    std::unique_ptr<float[]> float_data;
    std::unique_ptr<std::byte[]> byte_data;
    const std::byte *ptr = nullptr;
    if (ext == ".exr") {
        float_data = load_from_exr(path, ch, width, height);
        data_type = TextureDataType::f32;
    } else if (ext == ".hdr") {
        float_data = load_from_hdr(path, ch, width, height);
        data_type = TextureDataType::f32;
    } else {
        // TODO: support 16-bit pngs.
        byte_data = load_from_ldr(path, ch, width, height);
        data_type = TextureDataType::u8;
    }
    ptr = float_data ? reinterpret_cast<const std::byte *>(float_data.get()) : byte_data.get();
    return std::make_unique<Texture>(ptr, width, height, ch, data_type, src_colorspace, build_mipmaps);
}

// We use lz4 to compress serialized textures for smaller file size and fast decompress speed.

constexpr const char *serialized_texture_magic = "i_am_a_serialized_texture";

void write_texture_to_serialized(const Texture &texture, const fs::path &path)
{
    BinaryWriter writer(path);
    writer.write_array<char>(serialized_texture_magic, strlen(serialized_texture_magic));
    writer.write<int>(texture.width);
    writer.write<int>(texture.height);
    writer.write<int>(texture.num_channels);
    writer.write<TextureDataType>(texture.data_type);
    int levels = (int)texture.levels();
    writer.write<int>(levels);
    size_t total_size = 0;
    int w = texture.width;
    int h = texture.height;
    int stride = byte_stride(texture.data_type) * texture.num_channels;
    for (int l = 0; l < levels; ++l) {
        total_size += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }
    writer.write<size_t>(total_size);

    std::unique_ptr<std::byte[]> buf = std::make_unique<std::byte[]>(total_size);
    w = texture.width;
    h = texture.height;
    size_t offset = 0;
    for (int l = 0; l < levels; ++l) {
        texture.mips[l].copy_to_linear_array(buf.get() + offset);
        offset += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }

    // LZ4_MAX_INPUT_SIZE is ~2GB. For super high-res textures we need to split the data into blocks.
    int num_blocks = (int)std::ceil((double)total_size / double(LZ4_MAX_INPUT_SIZE));
    writer.write<int>(num_blocks);

    offset = 0;
    size_t total_compressed_capacity = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        total_compressed_capacity += LZ4_compressBound((int)block_size);
        offset += block_size;
    }
    std::unique_ptr<std::byte[]> compressed_buf = std::make_unique<std::byte[]>(total_compressed_capacity);
    offset = 0;
    size_t compressed_offset = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        int compressed_block_capacity = LZ4_compressBound((int)block_size);
        int compressed_block_size =
            LZ4_compress_default((const char *)(buf.get() + offset), (char *)(compressed_buf.get() + compressed_offset),
                                 (int)block_size, compressed_block_capacity);
        ASSERT(compressed_block_size > 0, "lz4 compression failed.");
        writer.write<int>(compressed_block_size);
        compressed_offset += compressed_block_size;
        offset += block_size;
    }
    size_t total_compressed_size = compressed_offset;
    writer.write_array<std::byte>(compressed_buf.get(), total_compressed_size);
}

std::unique_ptr<Texture> create_texture_from_serialized(const fs::path &path)
{
    BinaryReader reader(path);
    std::array<char, std::string_view(serialized_texture_magic).size() + 1> magic;
    reader.read_array<char>(magic.data(), magic.size() - 1);
    magic.back() = 0;
    if (strcmp(magic.data(), serialized_texture_magic)) {
        ASSERT(false, "Invalid serialized texture.");
        return nullptr;
    }
    int width = reader.read<int>();
    int height = reader.read<int>();
    int num_channels = reader.read<int>();
    TextureDataType data_type = reader.read<TextureDataType>();
    int levels = reader.read<int>();
    size_t total_size = reader.read<size_t>();
    int num_blocks = reader.read<int>();
    std::vector<int> compressed_block_sizes(num_blocks);
    size_t total_compressed_size = 0;
    for (int block = 0; block < num_blocks; ++block) {
        compressed_block_sizes[block] = reader.read<int>();
        total_compressed_size += compressed_block_sizes[block];
    }
    //
    std::unique_ptr<std::byte[]> compressed_buf = std::make_unique<std::byte[]>(total_compressed_size);
    reader.read_array<std::byte>(compressed_buf.get(), total_compressed_size);
    std::unique_ptr<std::byte[]> buf = std::make_unique<std::byte[]>(total_size);
    // LZ4_MAX_INPUT_SIZE is ~2GB. For super high-res textures we need to split the data into blocks.
    size_t offset = 0;
    size_t compressed_offset = 0;
    for (int block = 0; block < num_blocks; ++block) {
        size_t block_size = std::min(size_t(LZ4_MAX_INPUT_SIZE), total_size - offset);
        int ret = LZ4_decompress_safe((const char *)(compressed_buf.get() + compressed_offset),
                                      (char *)(buf.get() + offset), compressed_block_sizes[block], (int)block_size);
        ASSERT(ret > 0, "lz4 decompression failed.");
        offset += block_size;
        compressed_offset += compressed_block_sizes[block];
    }
    // compressed_buf can be released now.
    compressed_buf.reset();

    std::vector<const std::byte *> mip_bytes(levels);
    int w = width;
    int h = height;
    int stride = byte_stride(data_type) * num_channels;
    offset = 0;
    for (int l = 0; l < levels; ++l) {
        mip_bytes[l] = buf.get() + offset;
        offset += w * h * stride;
        w = std::max(1, (w + 1) / 2);
        h = std::max(1, (h + 1) / 2);
    }
    // TODO: fix me!
    ColorSpace color_space = ColorSpace::Linear;
    return std::make_unique<Texture>(mip_bytes, width, height, num_channels, data_type, color_space);
}

std::unique_ptr<Texture> create_texture(const ConfigArgs &args)
{
    fs::path path = args.load_path("path");
    bool serialized = args.load_bool("serialized", false);
    if (serialized) {
        return create_texture_from_serialized(path);
    } else {
        int ch = args.load_integer("channels");
        bool build_mipmaps = args.load_bool("build_mipmaps");
        std::string src_colorspace_str = args.load_string("colorspace", "Linear");
        ColorSpace src_colorspace = ColorSpace::Linear;
        if (src_colorspace_str == "sRGB") {
            src_colorspace = ColorSpace::sRGB;
        }
        return create_texture_from_image(ch, build_mipmaps, src_colorspace, path);
    }
}

std::unique_ptr<TextureSampler> create_texture_sampler(const ConfigArgs &args)
{
    std::unique_ptr<TextureSampler> sampler;
    std::string type = args.load_string("type");
    if (type == "nearest") {
        sampler = std::make_unique<NearestSampler>();
    } else if (type == "linear") {
        sampler = std::make_unique<LinearSampler>();
    } else if (type == "cubic") {
        CubicSampler::Kernel kernel = CubicSampler::Kernel::MitchellNetravali;
        std::string k = args.load_string("kernel", "mitchell");
        if (k == "bspline") {
            kernel = CubicSampler::Kernel::BSpline;
        } else if (k == "catmull_rom") {
            kernel = CubicSampler::Kernel::CatmullRom;
        }
        sampler = std::make_unique<CubicSampler>(kernel);
    } else if (type == "ewa") {
        float anisotropy = args.load_float("anisotropy", 8.0f);
        sampler = std::make_unique<EWASampler>(anisotropy);
    }

    std::string wu = args.load_string("wrap_mode_u", "repeat");
    if (wu == "repeat") {
        sampler->wrap_mode_u = TextureWrapMode::Repeat;
    } else if (wu == "clamp") {
        sampler->wrap_mode_u = TextureWrapMode::Clamp;
    }
    std::string wv = args.load_string("wrap_mode_v", "repeat");
    if (wv == "repeat") {
        sampler->wrap_mode_v = TextureWrapMode::Repeat;
    } else if (wv == "clamp") {
        sampler->wrap_mode_v = TextureWrapMode::Clamp;
    }
    return sampler;
}

void convert_texture_task(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    int n_textures = args["textures"].array_size();
    for (int i = 0; i < n_textures; ++i) {
        std::string asset_path = args["textures"].load_string(i);
        const Texture *texture = args.asset_table().get<Texture>(asset_path);
        std::string name = asset_path.substr(asset_path.rfind(".") + 1);
        write_texture_to_serialized(*texture, task_dir / (name + ".bin"));

        // int w = texture->width;
        // int h = texture->height;
        // for (int l = 0; l < texture->levels(); ++l) {
        //    RenderTarget rt(w, h, color3::Zero());
        //    for (int y = 0; y < h; ++y) {
        //        for (int x = 0; x < w; ++x) {
        //            color3 c;
        //            texture->fetch_as_float(x, y, l, {c.data(), 3});
        //            rt(x, y) = c;
        //        }
        //    }
        //    w = std::max(1, (w + 1) / 2);
        //    h = std::max(1, (h + 1) / 2);
        //    rt.save_to_png(task_dir / string_format("%s_%d.png", name.c_str(), l));
        //}
    }
}

} // namespace ks