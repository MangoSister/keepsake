#pragma once

#include "ks/assertion.h"
#include "ks/maths.h"

struct MicrofacetDistribution
{
    virtual ~MicrofacetDistribution() = default;
    virtual float D(const vec3 &wm) const = 0;
    virtual float G1(const vec3 &w) const = 0;
    virtual float G2(const vec3 &wo, const vec3 &wi) const = 0;
    virtual float pdf(const vec3 &wo, const vec3 &wm) const = 0;
    virtual vec3 sample(const vec3 &wo, const vec2 &u) const = 0;
};

struct GGX : public MicrofacetDistribution
{
    GGX() = default;

    explicit GGX(float alpha)
    {
        alpha_x = std::clamp(alpha, min_alpha, 1.0f);
        alpha_y = std::clamp(alpha, min_alpha, 1.0f);
    }

    GGX(float ax, float ay)
    {
        alpha_x = std::clamp(ax, min_alpha, 1.0f);
        alpha_y = std::clamp(ay, min_alpha, 1.0f);
    }

    bool isotropic() const { return alpha_x == alpha_y; }

    float D(const vec3 &wm) const
    {
        if (wm.z() <= 0.0f) {
            return 0.0f;
        }

        if (isotropic()) {
            float alpha = alpha_x;
            float a2 = alpha * alpha;
            float t = 1.0f + (a2 - 1.0f) * wm.z() * wm.z();
            return a2 / (pi * t * t);
        } else {
            float hx = wm.x() / alpha_x;
            float hy = wm.y() / alpha_y;
            float t = sqr(hx) + sqr(hy) + sqr(wm.z());
            return 1.0f / (pi * alpha_x * alpha_y * sqr(t));
        }
    }

    float lambda(const vec3 &w) const
    {
        if (w.z() >= 1.0f || w.z() <= -1.0f) {
            return 0.0f;
        }
        float alpha;
        if (isotropic()) {
            alpha = alpha_x;
        } else {
            float inv_sin_theta2 = 1.0f / (1.0f - w.z() * w.z());
            float cos_phi2 = w.x() * w.x() * inv_sin_theta2;
            float sin_phi2 = w.y() * w.y() * inv_sin_theta2;
            alpha = std::sqrt(cos_phi2 * alpha_x * alpha_x + sin_phi2 * alpha_y * alpha_y);
        }
        float alpha2 = alpha * alpha;
        float NdotV2 = w.z() * w.z();
        float t = (1.0f - NdotV2) * alpha2 / NdotV2;
        return 0.5f * (-1.0f + std::sqrt(1.0f + t));
    }

    float G1(const vec3 &w) const { return 1.0f / (1.0f + lambda(w)); }

    float G2(const vec3 &wo, const vec3 &wi) const
    {
        // See Heitz et al. 16, Appendix A.
        // lambda(w) = lambda(-w)
        if (wo.z() * wi.z() >= 0.0f) {
            return 1.0f / (1.0f + lambda(wo) + lambda(wi));
        } else {
            // beta function is symmetric
            return (float)std::beta(1.0f + lambda(wo), 1.0f + lambda(wi));
        }
    }

    float pdf(const vec3 &wo, const vec3 &wm) const
    {
        float pdf = G1(wo) * std::abs(wo.dot(wm)) * D(wm) / std::abs(wo.z());
        return pdf;
        // Don't forget to correct jacobian after this, depending on reflection or refraction.
    }

    vec3 sample(const vec3 &wo, const vec2 &u) const
    {
        // Section 3.2: transforming the view direction to the hemisphere configuration
        vec3 Vh = (vec3(alpha_x * wo.x(), alpha_y * wo.y(), wo.z())).normalized();
        // Section 4.1: orthonormal basis (with special case if cross product is zero)
        float lensq = Vh.x() * Vh.x() + Vh.y() * Vh.y();
        vec3 T1 = lensq > 0 ? vec3(-Vh.y(), Vh.x(), 0.0f) * (1.0f / std::sqrt(lensq)) : vec3(1.0f, 0.0f, 0.0f);
        vec3 T2 = Vh.cross(T1);
        // Section 4.2: parameterization of the projected area
        float r = std::sqrt(u.x());
        float phi = two_pi * u.y();
        float t1 = r * std::cos(phi);
        float t2 = r * std::sin(phi);
        float s = 0.5f * (1.0f + Vh.z());
        t2 = (1.0f - s) * std::sqrt(1.0f - t1 * t1) + s * t2;
        // Section 4.3: reprojection onto hemisphere
        vec3 Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
        // Section 3.4: transforming the normal back to the ellipsoid configuration
        vec3 Ne = (vec3(alpha_x * Nh.x(), alpha_y * Nh.y(), std::max(0.0f, Nh.z()))).normalized();
        return Ne;
    }

    float alpha_x = 0.1f, alpha_y = 0.1f;

    static constexpr float min_alpha = 1e-3f;
};

static void BeckmannSample11(float cosThetaI, float U1, float U2, float *slope_x, float *slope_y)
{
    /* Special case (normal incidence) */
    if (cosThetaI > .9999) {
        float r = std::sqrt(-std::log(1.0f - U1));
        float sinPhi = std::sin(2 * pi * U2);
        float cosPhi = std::cos(2 * pi * U2);
        *slope_x = r * cosPhi;
        *slope_y = r * sinPhi;
        return;
    }

    /* The original inversion routine from the paper contained
       discontinuities, which causes issues for QMC integration
       and techniques like Kelemen-style MLT. The following code
       performs a numerical inversion with better behavior */
    float sinThetaI = std::sqrt(std::max((float)0, (float)1 - cosThetaI * cosThetaI));
    float tanThetaI = sinThetaI / cosThetaI;
    float cotThetaI = 1 / tanThetaI;

    /* Search interval -- everything is parameterized
       in the Erf() domain */
    float a = -1, c = std::erf(cotThetaI);
    float sample_x = std::max(U1, (float)1e-6f);

    /* Start with a good initial guess */
    // float b = (1-sample_x) * a + sample_x * c;

    /* We can do better (inverse of an approximation computed in
     * Mathematica) */
    float thetaI = std::acos(cosThetaI);
    float fit = 1 + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
    float b = c - (1 + c) * std::pow(1 - sample_x, fit);

    /* Normalization factor for the CDF */
    static const float SQRT_PI_INV = 1.f / std::sqrt(pi);
    float normalization = 1 / (1 + c + SQRT_PI_INV * tanThetaI * std::exp(-cotThetaI * cotThetaI));

    int it = 0;
    while (++it < 10) {
        /* Bisection criterion -- the oddly-looking
           Boolean expression are intentional to check
           for NaNs at little additional cost */
        if (!(b >= a && b <= c))
            b = 0.5f * (a + c);

        /* Evaluate the CDF and its derivative
           (i.e. the density function) */
        float invErf = erfinv(b);
        float value = normalization * (1 + b + SQRT_PI_INV * tanThetaI * std::exp(-invErf * invErf)) - sample_x;
        float derivative = normalization * (1 - invErf * tanThetaI);

        if (std::abs(value) < 1e-5f)
            break;

        /* Update bisection intervals */
        if (value > 0)
            c = b;
        else
            a = b;

        b -= value / derivative;
    }

    /* Now convert back into a slope value */
    *slope_x = erfinv(b);

    /* Simulate Y component */
    *slope_y = erfinv(2.0f * std::max(U2, (float)1e-6f) - 1.0f);

    ASSERT(!std::isinf(*slope_x));
    ASSERT(!std::isnan(*slope_x));
    ASSERT(!std::isinf(*slope_y));
    ASSERT(!std::isnan(*slope_y));
}

static vec3 BeckmannSample(const vec3 &wi, float alpha_x, float alpha_y, float U1, float U2)
{
    // 1. stretch wi
    vec3 wiStretched = (vec3(alpha_x * wi.x(), alpha_y * wi.y(), wi.z())).normalized();

    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    BeckmannSample11(wiStretched.z(), U1, U2, &slope_x, &slope_y);

    float cos_theta2 = sqr(wiStretched.z());
    float sin_theta2 = std::max(1.0f - cos_theta2, 0.0f);
    float sin_theta = std::sqrt(sin_theta2);
    float cos_phi = (sin_theta == 0) ? 1 : std::clamp(wiStretched.x() / sin_theta, -1.0f, 1.0f);
    float sin_phi = (sin_theta == 0) ? 0 : std::clamp(wiStretched.y() / sin_theta, -1.0f, 1.0f);

    // 3. rotate
    float tmp = cos_phi * slope_x - sin_phi * slope_y;
    slope_y = sin_phi * slope_x + cos_phi * slope_y;
    slope_x = tmp;

    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;

    // 5. compute normal
    return (vec3(-slope_x, -slope_y, 1.f)).normalized();
}

struct Beckmann : public MicrofacetDistribution
{
    Beckmann() = default;

    explicit Beckmann(float alpha)
    {
        alpha_x = std::clamp(alpha, min_alpha, 1.0f);
        alpha_y = std::clamp(alpha, min_alpha, 1.0f);
    }

    Beckmann(float ax, float ay)
    {
        alpha_x = std::clamp(ax, min_alpha, 1.0f);
        alpha_y = std::clamp(ay, min_alpha, 1.0f);
    }

    bool isotropic() const { return alpha_x == alpha_y; }

    float D(const vec3 &wm) const
    {
        float cos_theta2 = sqr(wm.z());
        float sin_theta2 = std::max(1.0f - cos_theta2, 0.0f);
        float tan_theta2 = sin_theta2 / cos_theta2;
        if (std::isinf(tan_theta2))
            return 0.0f;
        float cos_theta4 = sqr(cos_theta2);

        float sin_theta = std::sqrt(sin_theta2);
        float cos_phi = (sin_theta == 0) ? 1 : std::clamp(wm.x() / sin_theta, -1.0f, 1.0f);
        float cos_phi2 = sqr(cos_phi);
        float sin_phi = (sin_theta == 0) ? 0 : std::clamp(wm.y() / sin_theta, -1.0f, 1.0f);
        float sin_phi2 = sqr(sin_phi);

        return std::exp(-tan_theta2 * (cos_phi2 / sqr(alpha_x) + sin_phi2 / sqr(alpha_y))) /
               (pi * alpha_x * alpha_y * cos_theta4);
    }

    float lambda(const vec3 &w) const
    {
        float cos_theta2 = sqr(w.z());
        float sin_theta2 = std::max(1.0f - cos_theta2, 0.0f);
        float sin_theta = std::sqrt(sin_theta2);
        float tan_theta = sin_theta / w.z();
        float absTanTheta = std::abs(tan_theta);
        if (std::isinf(absTanTheta))
            return 0.;

        float cos_phi = (sin_theta == 0) ? 1 : std::clamp(w.x() / sin_theta, -1.0f, 1.0f);
        float cos_phi2 = sqr(cos_phi);
        float sin_phi = (sin_theta == 0) ? 0 : std::clamp(w.y() / sin_theta, -1.0f, 1.0f);
        float sin_phi2 = sqr(sin_phi);

        float alpha = std::sqrt(cos_phi2 * sqr(alpha_x) + sin_phi2 * sqr(alpha_y));
        float a = 1 / (alpha * absTanTheta);
        if (a >= 1.6f)
            return 0;
        return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
    }

    float G1(const vec3 &w) const { return 1.0f / (1.0f + lambda(w)); }

    float G2(const vec3 &wo, const vec3 &wi) const
    {
        // See Heitz et al. 16, Appendix A.
        // lambda(w) = lambda(-w)
        if (wo.z() * wi.z() >= 0.0f) {
            return 1.0f / (1.0f + lambda(wo) + lambda(wi));
        } else {
            // beta function is symmetric
            return (float)std::beta(1.0f + lambda(wo), 1.0f + lambda(wi));
        }
    }

    float pdf(const vec3 &wo, const vec3 &wm) const
    {
        float pdf = G1(wo) * std::abs(wo.dot(wm)) * D(wm) / std::abs(wo.z());
        return pdf;
        // Don't forget to correct jacobian after this, depending on reflection or refraction.
    }

    vec3 sample(const vec3 &wo, const vec2 &u) const
    {
        vec3 wh;
        bool flip = wo.z() < 0;
        wh = BeckmannSample(flip ? -wo : wo, alpha_x, alpha_y, u[0], u[1]);
        if (flip)
            wh = -wh;
        return wh;
    }

    float alpha_x = 0.1f, alpha_y = 0.1f;

    static constexpr float min_alpha = 1e-3f;
};

struct MicrofacetAdapter
{
    virtual float D(float ax, float ay, const vec3 &wm) const = 0;
    virtual float G1(float ax, float ay, const vec3 &w) const = 0;
    virtual float G2(float ax, float ay, const vec3 &wo, const vec3 &wi) const = 0;
    virtual float pdf(float ax, float ay, const vec3 &wo, const vec3 &wm) const = 0;
    virtual vec3 sample(float ax, float ay, const vec3 &wo, const vec2 &u) const = 0;
};

template <typename M>
struct MicrofacetAdapterDerived : public MicrofacetAdapter
{
    float D(float ax, float ay, const vec3 &wm) const { return M(ax, ay).D(wm); }
    float G1(float ax, float ay, const vec3 &w) const { return M(ax, ay).G1(w); };
    float G2(float ax, float ay, const vec3 &wo, const vec3 &wi) const { return M(ax, ay).G2(wo, wi); };
    float pdf(float ax, float ay, const vec3 &wo, const vec3 &wm) const { return M(ax, ay).pdf(wo, wm); };
    vec3 sample(float ax, float ay, const vec3 &wo, const vec2 &u) const { return M(ax, ay).sample(wo, u); };
};