#pragma once
#include "config.h"
#include "maths.h"

namespace ks
{

struct ToneMapper
{
    virtual ~ToneMapper() = default;
    virtual color3 operator()(color3 c) const = 0;
};

// ACES and AgX are mainly based on three.js implementation.
// TODO: maybe an actual spectral rendering/color production workflow someday...
// TODO: The default AgX look seems a bit flat? expected?
// https://github.com/google/filament/blob/main/filament/src/ToneMapper.cpp
// https://github.com/Apress/physically-based-shader-dev-for-unity-2017/blob/master/PostProcessingv2/Assets/PostProcessing/PostProcessing/Shaders/ACES.hlsl
// https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// https://github.com/godotengine/godot/blob/master/servers/rendering/renderer_rd/shaders/effects/tonemap.glsl
// https://dev.epicgames.com/documentation/en-us/unreal-engine/color-grading-and-the-filmic-ToneMapper-in-unreal-engine
// https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js

// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
// clang-format off
static const mat3 ACESInputMat((mat3() <<
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777)
.finished());
// clang-format on

// ODT_SAT => XYZ => D60_2_D65 => sRGB
// clang-format off
static const mat3 ACESOutputMat((mat3() <<
        1.60475, -0.53108, -0.07367,
        -0.10208, 1.10813, -0.00605,
        -0.00327, -0.07276, 1.07602)
.finished());
// clang-format on

// source: https://github.com/selfshadow/ltc_code/blob/master/webgl/shaders/ltc/ltc_blit.fs
inline color3 RRTAndODTFit(color3 v)
{
    color3 a = v * (v + 0.0245786) - 0.000090537;
    color3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

struct ACESToneMapper : public ToneMapper
{
    ACESToneMapper(float exposure) : exposure(exposure) {}

    color3 operator()(color3 v) const final
    {
        // This exposure boost follows three.js implementation.
        v *= exposure / 0.6;

        v = ACESInputMat * v.matrix();

        // Apply RRT and ODT
        v = RRTAndODTFit(v);

        v = ACESOutputMat * v.matrix();

        // Clamp to [0, 1]
        v = clamp(v, color3::Zero(), color3::Ones());

        return v;
    }

    float exposure;
};

enum class AgXLook
{
    None,
    Golden,
    Punchy,
};

// Adapted from https://iolite-engine.com/blog_posts/minimal_agx_implementation
inline color3 agxDefaultContrastApprox(color3 x)
{
    color3 x2 = x * x;
    color3 x4 = x2 * x2;

    return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// Adapted from https://iolite-engine.com/blog_posts/minimal_agx_implementation
inline color3 agxLook(color3 val, AgXLook look)
{
    if (look == AgXLook::None) {
        return val;
    }

    color3 lw(0.2126f, 0.7152f, 0.0722f);
    float luma = val.matrix().dot(lw.matrix());

    // Default
    color3 offset = color3(0.0f);
    color3 slope = color3(1.0f);
    color3 power = color3(1.0f);
    float sat = 1.0f;

    if (look == AgXLook::Golden) {
        slope = color3(1.0f, 0.9f, 0.5f);
        power = color3(0.8f);
        sat = 1.3;
    }
    if (look == AgXLook::Punchy) {
        slope = color3(1.0f);
        power = color3(1.35f, 1.35f, 1.35f);
        sat = 1.4;
    }

    // ASC CDL
    val = pow(val * slope + offset, power);
    return luma + sat * (val - luma);
}

// https://github.com/EaryChow/AgX_LUT_Gen/blob/main/AgXBaseRec2020.py

// clang-format off
static const mat3 AgXInsetMatrix((mat3() <<
        0.856627153315983, 0.0951212405381588, 0.0482516061458583,
        0.137318972929847, 0.761241990602591, 0.101439036467562,
        0.11189821299995, 0.0767994186031903, 0.811302368396859)
.finished());
// clang-format on

// clang-format off
static const mat3 AgXOutsetMatrixInv((mat3() <<
        0.899796955911611, 0.0871996192028351, 0.013003424885555,
        0.11142098895748, 0.875575586156966, 0.0130034248855548,
        0.11142098895748, 0.0871996192028349, 0.801379391839686)
.finished());
// clang-format on

static const mat3 AgXOutsetMatrix = AgXOutsetMatrixInv.inverse();

// clang-format off
static const mat3 LINEAR_REC2020_TO_LINEAR_SRGB ((mat3() <<
        1.6605, -0.5876, -0.0728,
        -0.1246, 1.1329, -0.0083,
        -0.0182, -0.1006, 1.1187)
.finished());
// clang-format on

// clang-format off
static const mat3 LINEAR_SRGB_TO_LINEAR_REC2020  ((mat3() <<
        0.6274, 0.3293, 0.0433,
        0.0691, 0.9195, 0.0113,
        0.0164, 0.0880, 0.8956)
.finished());
// clang-format on

// LOG2_MIN      = -10.0
// LOG2_MAX      =  +6.5
// MIDDLE_GRAY   =  0.18
const float AgxMinEv = -12.47393f; // log2(pow(2, LOG2_MIN) * MIDDLE_GRAY)
const float AgxMaxEv = 4.026069f;  // log2(pow(2, LOG2_MAX) * MIDDLE_GRAY)

struct AgXToneMapper : public ToneMapper
{
    AgXToneMapper(float exposure = 1.0f, AgXLook look = AgXLook::None) : exposure(exposure), look(look) {}

    color3 operator()(color3 v) const final
    {
        v *= exposure;

        // Ensure no negative values
        v = v.cwiseMax(0.0f);

        v = LINEAR_SRGB_TO_LINEAR_REC2020 * v.matrix();

        v = AgXInsetMatrix * v.matrix();

        // Log2 encoding
        v = v.cwiseMax(1E-10f); // avoid 0 or negative numbers for log2
        v = log2(v);
        v = (v - AgxMinEv) / (AgxMaxEv - AgxMinEv);

        v = clamp(v, color3::Zero(), color3::Ones());

        // Apply sigmoid
        v = agxDefaultContrastApprox(v);

        // Apply AgX look
        v = agxLook(v, look);

        v = AgXOutsetMatrix * v.matrix();

        // Linearize
        v = pow(v.cwiseMax(color3(0.0f)), 2.2f);

        v = LINEAR_REC2020_TO_LINEAR_SRGB * v.matrix();

        // Gamut mapping. Simple clamp for now.
        v = clamp(v, color3::Zero(), color3::Ones());

        return v;
    }

    float exposure;
    AgXLook look;
};

struct KhronosPBRNeutralToneMapper : public ToneMapper
{
    explicit KhronosPBRNeutralToneMapper(float exposure = 1.0f) : exposure(exposure) {}

    color3 operator()(color3 v) const final
    {
        v *= exposure;

        constexpr float F90 = 0.04f;
        constexpr float Ks = 0.8f - F90;
        constexpr float Kd = 0.15f;

        float x = v.minCoeff();
        float f = x <= 2.0f * F90 ? (x - sqr(x) / (4.0f * F90)) : F90;
        float p = std::max(std::max(v[0] - f, v[1] - f), v[2] - f);
        float pn = 1.0f - sqr(1.0f - Ks) / (p + 1.0f - 2.0f * Ks);
        float g = 1.0f / (Kd * (p - pn) + 1.0f);
        if (p <= Ks) {
            return v - f;
        } else {
            return (v - f) * (pn / p) * g + color3::Constant(pn) * (1.0f - g);
        }
    }

    float exposure;
};

inline std::unique_ptr<ToneMapper> create_tone_mapper(const ConfigArgs &args)
{
    float exposure = args.load_float("exposure", 1.0f);
    std::string type = args.load_string("type");
    if (type == "aces") {
        return std::make_unique<ACESToneMapper>(exposure);
    } else if (type == "agx") {
        std::string look_str = args.load_string("look");
        AgXLook look = AgXLook::None;
        if (look_str == "none") {
            look = AgXLook::None;
        } else if (look_str == "golden") {
            look = AgXLook::Golden;
        } else if (look_str == "punchy") {
            look = AgXLook::Punchy;
        } else {
            fprintf(stderr, "Invalid AgX tone mapper look [%s]", look_str.c_str());
            std::abort();
        }
        return std::make_unique<AgXToneMapper>(exposure, look);
    } else if (type == "khronos_pbr_neutral") {
        return std::make_unique<KhronosPBRNeutralToneMapper>();
    }
    fprintf(stderr, "Invalid tone mapper type [%s]", type.c_str());
    std::abort();
}

} // namespace ks