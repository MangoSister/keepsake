#pragma once

#include "ks/maths.h"

KS_NAMESPACE_BEGIN

// eta = eta_i / eta_t
inline float fresnel_dielectric(float cos_theta_i, float eta)
{
    float sin_theta_t_2 = eta * eta * (1.0f - cos_theta_i * cos_theta_i);

    // Total internal reflection
    if (sin_theta_t_2 > 1.0)
        return 1.0;

    float cos_theta_t = std::sqrt(std::max(1.0f - sin_theta_t_2, 0.0f));

    float rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    return 0.5f * (rs * rs + rp * rp);
}

// Parallel polarization (P-polarization) only
inline float fresnel_dielectric_parallel(float cos_theta_i, float eta)
{
    float sin_theta_t_2 = eta * eta * (1.0f - cos_theta_i * cos_theta_i);

    // Total internal reflection
    if (sin_theta_t_2 > 1.0)
        return 1.0;

    float cos_theta_t = std::sqrt(std::max(1.0f - sin_theta_t_2, 0.0f));

    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    return rp * rp;
}

inline float fresnel_schlick(float cos_theta_i)
{
    float schlick = saturate(1.0f - cos_theta_i);
    float schlick2 = sqr(schlick);
    schlick = schlick2 * schlick2 * schlick;
    return schlick;
}

inline float fresnel_mix(float metallic, float eta, float VDotH)
{
    float metallicFresnel = fresnel_schlick(VDotH);
    float dielectricFresnel = fresnel_dielectric(VDotH, eta);
    return std::lerp(dielectricFresnel, metallicFresnel, metallic);
}

KS_NAMESPACE_END