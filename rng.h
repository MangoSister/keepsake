#pragma once

#include "maths.h"
#include <array>
#include <pcg_random.hpp>
#include <random>

struct RNG
{
    RNG() = default;
    explicit RNG(uint64_t seed) : pcg(seed) {}

    uint32_t nextu32() { return pcg(); }

    float next()
    {
        uint32_t r = pcg();
        return std::min((float)r / 0xFFFFFFFFu, before_one);
    }

    template <int N>
    Eigen::Matrix<float, N, 1> next()
    {
        Eigen::Matrix<float, N, 1> u;
        for (int i = 0; i < N; ++i) {
            u[i] = next();
        }
        return u;
    }

    vec2 next2d() { return next<2>(); }
    vec3 next3d() { return next<3>(); }

    pcg32 pcg;
};

constexpr float radical_inverse(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

inline vec2 hammersley_2d(uint32_t i, uint32_t N) { return vec2(float(i) / float(N), radical_inverse(i)); }

// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// https://www.shadertoy.com/view/4dtBWH
inline vec2 roberts_qmc_2d(uint32_t n, const vec2 &p0 = vec2::Zero())
{
    vec2 p;
    p.x() = fract(p0.x() + (float)(n * 12664745) / (float)(1 << 24));
    p.y() = fract(p0.y() + (float)(n * 9560333) / (float)(1 << 24));
    return p;
}

inline vec3 sample_uniform_hemisphere(const vec2 &u)
{
    float z = u.y();
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = two_pi * u.x();
    float sin_phi = std::sin(phi);
    float cos_phi = std::cos(phi);

    return vec3(r * cos_phi, r * sin_phi, z);
}

inline vec2 sample_disk(const vec2 &u)
{
    float a = 2.0f * u[0] - 1.0f;
    float b = 2.0f * u[1] - 1.0f;
    float r, phi;
    if (a * a > b * b) {
        r = a;
        phi = 0.25f * pi * (b / a);
    } else {
        r = b;
        phi = half_pi - 0.25f * pi * (a / b);
    }
    return r * vec2(std::cos(phi), std::sin(phi));
}

inline vec3 sample_cosine_hemisphere(const vec2 &u)
{
    vec2 d = sample_disk(u);
    float z = std::sqrt(std::max(0.0f, 1.0f - d.squaredNorm()));
    return vec3(d.x(), d.y(), z);
    // pdf = z / pi;
}

inline vec3 sample_cosine_hemisphere(const vec2 &u, const vec3 &n)
{
    float a = 1.0f - 2.0f * u[0];
    float b = safe_sqrt(1.0f - a * a);
    // Avoid zero vector output in case the sample is in the opposite direction of n.
    a *= 0.999f;
    b *= 0.999f;

    float phi = two_pi * u[1];
    float x = n.x() + b * cos(phi);
    float y = n.y() + b * sin(phi);
    float z = n.z() + a;
    return vec3(x, y, z).normalized();
}

inline vec3 sample_uniform_sphere(const vec2 &u)
{
    float z = 1.0f - 2.0f * u.y();
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = two_pi * u.x();
    float sin_phi = std::sin(phi);
    float cos_phi = std::cos(phi);
    return vec3(r * cos_phi, r * sin_phi, z);
}

inline vec3 sample_uniform_sphere_vol(const vec3 &u)
{
    float phi = u.x() * two_pi;
    float cos_theta = u.y() * 2.0f - 1.0f;
    float sin_theta = std::sqrt(std::max(0.0f, 1.0f - square(cos_theta)));
    float r = std::cbrt(u.z()); // cube root
    return r * vec3(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta);
}

inline float sample_normal(RNG &rng, float mean, float stddev)
{
    std::normal_distribution<> normal_dist(mean, stddev);
    return normal_dist(rng.pcg);
}

template <int N>
inline std::array<float, N> sample_normal(RNG &rng)
{
    std::normal_distribution<> normal_dist(0.0f, 1.0f);
    std::array<float, N> samples;
    for (int i = 0; i < N; ++i) {
        samples[i] = normal_dist(rng.pcg);
    }
    return samples;
}

inline bool russian_roulette(color3 &beta, float u)
{
    if (beta.maxCoeff() < 0.05f) {
        float q = std::max(0.05f, 1.0f - beta.maxCoeff());
        if (u < q) {
            return false;
        }
        beta /= 1.0f - q;
    }
    return true;
}

inline vec3 sample_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec2 &u)
{
    float su0 = std::sqrt(u[0]);
    float alpha = 1.0f - su0;
    float beta = u[1] * su0;
    float gamma = 1.0f - alpha - beta;
    return alpha * v0 + beta * v1 + gamma * v2;
}