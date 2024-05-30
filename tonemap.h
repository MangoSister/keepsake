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

// TODO: maybe an actual spectral/color production workflow later...
// https://github.com/google/filament/blob/main/filament/src/ToneMapper.cpp
// https://github.com/Apress/physically-based-shader-dev-for-unity-2017/blob/master/PostProcessingv2/Assets/PostProcessing/PostProcessing/Shaders/ACES.hlsl
// https://github.com/ampas/aces-dev/blob/master/transforms/ctl/README-MATRIX.md
// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// https://github.com/godotengine/godot/blob/master/servers/rendering/renderer_rd/shaders/effects/tonemap.glsl
// https://dev.epicgames.com/documentation/en-us/unreal-engine/color-grading-and-the-filmic-ToneMapper-in-unreal-engine

// clang-format off
static const mat3 Rec2020_to_XYZ((mat3() <<
        0.6369530f, 0.1446169f, 0.1688558f,
        0.2626983f, 0.6780088f, 0.0592929f,
        0.0000000f, 0.0280731f, 1.0608272f)
.finished());
// clang-format on

// clang-format off
static const mat3 XYZ_to_Rec2020((mat3() <<
        1.7166634f, -0.3556733f, -0.2533681f,
        -0.6666738f, 1.6164557f, 0.0157683f,
        0.0176425f, -0.0427770f, 0.9422433f)
.finished());
// clang-format on

// clang-format off
static const mat3 AP1_to_XYZ((mat3() <<
        0.6624541811f, 0.1340042065f, 0.1561876870f,
        0.2722287168f, 0.6740817658f, 0.0536895174f,
        -0.0055746495f, 0.0040607335f, 1.0103391003f)
.finished());
// clang-format on

// clang-format off
static const mat3 XYZ_to_AP1((mat3() <<
        1.6410233797f, -0.3248032942f, -0.2364246952f,
        -0.6636628587f, 1.6153315917f, 0.0167563477f,
        0.0117218943f, -0.0082844420f, 0.9883948585f)
.finished());
// clang-format on

// clang-format off
static const mat3 AP1_to_AP0((mat3() <<
        0.6954522414f, 0.1406786965f, 0.1638690622f,
        0.0447945634f, 0.8596711185f, 0.0955343182f,
        -0.0055258826f, 0.0040252103f, 1.0015006723f)
.finished());
// clang-format on

// clang-format off
static const mat3 AP0_to_AP1((mat3() <<
        1.4514393161f, -0.2365107469f, -0.2149285693f,
        -0.0765537734f, 1.1762296998f, -0.0996759264f,
        0.0083161484f, -0.0060324498f, 0.9977163014f)
.finished());
// clang-format on

static const mat3 Rec2020_to_AP0 = AP1_to_AP0 * XYZ_to_AP1 * Rec2020_to_XYZ;

static const mat3 AP1_to_Rec2020 = XYZ_to_Rec2020 * AP1_to_XYZ;

static const color3 LUMINANCE_AP1(0.272229f, 0.674082f, 0.0536895f);

inline float rgb_2_saturation(color3 rgb)
{
    // Input:  ACES
    // Output: OCES
    constexpr float TINY = 1e-5f;
    float mi = rgb.minCoeff();
    float ma = rgb.maxCoeff();
    return (std::max(ma, TINY) - std::max(mi, TINY)) / std::max(ma, 1e-2f);
}

inline float rgb_2_yc(color3 rgb)
{
    constexpr float ycRadiusWeight = 1.75f;

    // Converts RGB to a luminance proxy, here called YC
    // YC is ~ Y + K * Chroma
    // Constant YC is a cone-shaped surface in RGB space, with the tip on the
    // neutral axis, towards white.
    // YC is normalized: RGB 1 1 1 maps to YC = 1
    //
    // ycRadiusWeight defaults to 1.75, although can be overridden in function
    // call to rgb_2_yc
    // ycRadiusWeight = 1 -> YC for pure cyan, magenta, yellow == YC for neutral
    // of same value
    // ycRadiusWeight = 2 -> YC for pure red, green, blue  == YC for  neutral of
    // same value.

    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];

    float chroma = std::sqrt(b * (b - g) + g * (g - r) + r * (r - b));

    return (b + g + r + ycRadiusWeight * chroma) / 3.0f;
}

inline float sigmoid_shaper(float x)
{
    // Sigmoid function in the range 0 to 1 spanning -2 to +2.
    float t = std::max(1.0f - std::abs(x / 2.0f), 0.0f);
    float y = 1.0f + sgn(x) * (1.0f - t * t);
    return y / 2.0f;
}

inline float glow_fwd(float ycIn, float glowGainIn, float glowMid)
{
    float glowGainOut;

    if (ycIn <= 2.0f / 3.0f * glowMid) {
        glowGainOut = glowGainIn;
    } else if (ycIn >= 2.0f * glowMid) {
        glowGainOut = 0.0f;
    } else {
        glowGainOut = glowGainIn * (glowMid / ycIn - 1.0f / 2.0f);
    }

    return glowGainOut;
}

inline float rgb_2_hue(color3 rgb)
{
    // Returns a geometric hue angle in degrees (0-360) based on RGB values.
    // For neutral colors, hue is undefined and the function will return a quiet NaN value.
    float hue = 0.0f;
    // RGB triplets where RGB are equal have an undefined hue
    if (!(rgb.x() == rgb.y() && rgb.y() == rgb.z())) {
        hue = to_degree(std::atan2(std::sqrt(3.0f) * (rgb.y() - rgb.z()), 2.0f * rgb.x() - rgb.y() - rgb.z()));
    }
    return (hue < 0.0f) ? hue + 360.0f : hue;
}

inline float center_hue(float hue, float centerH)
{
    float hueCentered = hue - centerH;
    if (hueCentered < -180.0f) {
        hueCentered = hueCentered + 360.0f;
    } else if (hueCentered > 180.0f) {
        hueCentered = hueCentered - 360.0f;
    }
    return hueCentered;
}

inline color3 xyY_to_XYZ(color3 v)
{
    const float a = v.z() / std::max(v.y(), 1e-5f);
    return color3{v.x() * a, v.z(), (1.0f - v.x() - v.y()) * a};
}

inline color3 XYZ_to_xyY(color3 v)
{
    float denom = std::max(v.x() + v.y() + v.z(), 1e-5f);
    return {v.x() / denom, v.y() / denom, v.y()};
}

inline color3 darkSurround_to_dimSurround(color3 linearCV)
{
    constexpr float DIM_SURROUND_GAMMA = 0.9811f;

    color3 XYZ = AP1_to_XYZ * linearCV.matrix();
    color3 xyY = XYZ_to_xyY(XYZ);

    xyY.z() = std::clamp(xyY.z(), 0.0f, (float)std::numeric_limits<float>::max());
    xyY.z() = std::pow(xyY.z(), DIM_SURROUND_GAMMA);

    XYZ = xyY_to_XYZ(xyY);
    return XYZ_to_AP1 * XYZ.matrix();
}

struct ACESToneMapper : public ToneMapper
{
    color3 operator()(color3 input) const final
    {
        // "Glow" module constants
        constexpr float RRT_GLOW_GAIN = 0.05f;
        constexpr float RRT_GLOW_MID = 0.08f;

        // Red modifier constants
        constexpr float RRT_RED_SCALE = 0.82f;
        constexpr float RRT_RED_PIVOT = 0.03f;
        constexpr float RRT_RED_HUE = 0.0f;
        constexpr float RRT_RED_WIDTH = 135.0f;

        // Desaturation constants
        constexpr float RRT_SAT_FACTOR = 0.96f;
        constexpr float ODT_SAT_FACTOR = 0.93f;

        color3 ap0 = Rec2020_to_AP0 * input.matrix();

        // Glow module
        float saturation = rgb_2_saturation(ap0);
        float ycIn = rgb_2_yc(ap0);
        float s = sigmoid_shaper((saturation - 0.4f) / 0.2f);
        float addedGlow = 1.0f + glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
        ap0 *= addedGlow;

        // Red modifier
        float hue = rgb_2_hue(ap0);
        float centeredHue = center_hue(hue, RRT_RED_HUE);
        float hueWeight = smoothstep(0.0f, 1.0f, 1.0f - std::abs(2.0f * centeredHue / RRT_RED_WIDTH));
        hueWeight *= hueWeight;

        ap0[0] += hueWeight * saturation * (RRT_RED_PIVOT - ap0[0]) * (1.0f - RRT_RED_SCALE);

        // ACES to RGB rendering space
        color3 ap1 = clamp((AP0_to_AP1 * ap0.matrix()).array(), color3::Zero(),
                           color3::Constant(std::numeric_limits<float>::max()));

        // Global desaturation
        ap1 = lerp(color3(ap1.matrix().dot(LUMINANCE_AP1.matrix())), ap1, RRT_SAT_FACTOR);

        // Fitting of RRT + ODT (RGB monitor 100 nits dim) from:
        // https://github.com/colour-science/colour-unity/blob/master/Assets/Colour/Notebooks/CIECAM02_Unity.ipynb
        constexpr float a = 2.785085f;
        constexpr float b = 0.107772f;
        constexpr float c = 2.936045f;
        constexpr float d = 0.887122f;
        constexpr float e = 0.806889f;
        color3 rgbPost = (ap1 * (a * ap1 + b)) / (ap1 * (c * ap1 + d) + e);

        // Apply gamma adjustment to compensate for dim surround
        color3 linearCV = darkSurround_to_dimSurround(rgbPost);

        // Apply desaturation to compensate for luminance difference
        linearCV = lerp(color3(linearCV.matrix().dot(LUMINANCE_AP1.matrix())), linearCV, ODT_SAT_FACTOR);

        return AP1_to_Rec2020 * linearCV.matrix();
    }
};

// TODO: The default look seems a bit flat? expected?
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
    color3 x6 = x4 * x2;
    return -17.86f * x6 * x + 78.01f * x6 - 126.7f * x4 * x + 92.06f * x4 - 28.72f * x2 * x + 4.361f * x2 -
           0.1718f * x + 0.002857f;
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

// LOG2_MIN      = -10.0
// LOG2_MAX      =  +6.5
// MIDDLE_GRAY   =  0.18
const float AgxMinEv = -12.47393f; // log2(pow(2, LOG2_MIN) * MIDDLE_GRAY)
const float AgxMaxEv = 4.026069f;  // log2(pow(2, LOG2_MAX) * MIDDLE_GRAY)

struct AgXToneMapper : public ToneMapper
{
    explicit AgXToneMapper(AgXLook look = AgXLook::None) : look(look) {}

    color3 operator()(color3 v) const final
    {
        // Ensure no negative values
        v = v.cwiseMax(0.0f);

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

        return v;
    }

    AgXLook look;
};

inline std::unique_ptr<ToneMapper> create_tone_mapper(const ConfigArgs &args)
{
    std::string type = args.load_string("type");
    if (type == "aces") {
        return std::make_unique<ACESToneMapper>();
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
        return std::make_unique<AgXToneMapper>(look);
    }
    fprintf(stderr, "Invalid pixel filter type [%s]", type.c_str());
    std::abort();
}

} // namespace ks