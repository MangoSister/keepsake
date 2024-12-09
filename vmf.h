#pragma once

#include "maths.h"
#include "rng.h"
#include <span>

namespace ks
{

struct VMF
{
    VMF() = default;

    VMF(const vec3 &mu, float kappa) : mu(mu), kappa(std::min(kappa, max_kappa)) {}

    explicit VMF(const vec3 &r)
    {
        float rn = std::clamp(r.norm(), 0.0f, 1.0f);
        if (rn == 0.0f) {
            kappa = 0.0f;
            mu = vec3::UnitZ();
        } else if (rn == 1.0f) {
            kappa = max_kappa;
            mu = r;
        } else {
            kappa = mean_cosine_to_kappa(rn);
            kappa = std::min(kappa, max_kappa);
            mu = r / rn;
        }
    }

    float eval(const vec3 &w) const
    {
        if (kappa == 0.0f) {
            return inv_pi * 0.25f;
        }
        float scale = kappa / (two_pi * (1.0f - std::exp(-2.0f * kappa)));
        return scale * std::exp(kappa * std::min(0.0f, mu.dot(w) - 1.0f));
    }

    vec3 sample(const vec2 &u) const
    {
        if (kappa == 0.0f) {
            return sample_uniform_sphere(u);
        }
        float cos_theta = 1.0f + (std::log(u.x() + std::exp(-2.0f * kappa) * (1.0f - u.x()))) / kappa;
        float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = two_pi * u.y();
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);

        vec3 w = vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
        vec3 X, Y;
        orthonormal_basis(mu, X, Y);
        return w.x() * X + w.y() * Y + w.z() * mu;
    }

    float mean_cosine() const
    {
        if (kappa == 0.0f) {
            return 0.0f;
        }
        float coth = kappa > 6 ? 1 : ((std::exp(2.0f * kappa) + 1.0f) / (std::exp(2 * kappa) - 1.0f));
        return coth - 1.0f / kappa;
    }

    float norm_term() const
    {
        if (kappa == 0.0f) {
            return 1.0 / (4.0f * pi);
        }
        return kappa / (two_pi * (1.0f - std::exp(-2.0 * kappa)));
    }

    VMF exp_map(const vec3 &v) const
    {
        float k_out = kappa * std::exp(v.x());
        vec2 u = v.tail(2);
        float un = u.norm();
        float sc = sinx_over_x(un);
        vec3 u3(u.x() * sc, u.y() * sc, std::cos(un));
        vec3 X, Y;
        orthonormal_basis(mu, X, Y);
        vec3 mu_out = u3.x() * X + u3.y() * Y + u3.z() * mu;
        return VMF(mu_out, k_out);
    }

    vec3 log_map(const VMF &p) const
    {
        float a = std::log(p.kappa / kappa);
        vec3 X, Y;
        orthonormal_basis(mu, X, Y);
        vec3 u3(X.dot(p.mu), Y.dot(p.mu), mu.dot(p.mu));
        float un = std::acos(std::clamp(u3.z(), -1.0f, 1.0f));
        float sc = sinx_over_x(un);
        vec2 u(u3.x() / sc, u3.y() / sc);
        return vec3(a, u.x(), u.y());
    }

    static constexpr float mean_cosine_to_kappa(float mean_cosine)
    {
        float mean_cos2 = sqr(mean_cosine);
        float mean_cos3 = mean_cos2 * mean_cosine;
        float kappa = (3.0f * mean_cosine - mean_cos3) / (1.0f - mean_cos2);
        kappa = std::min(kappa, max_kappa);
        return kappa;
    }

    // Avoid delta distribution.
    static constexpr float max_kappa = 1e4f;

    vec3 mu = vec3::UnitZ();
    float kappa = 0.0f;
};

// TODO: The VMF mixture fitter needs a rework.
struct WVMF
{
    VMF lobe;
    float w = 0.0f;
};

inline vec3 sample_vmm(std::span<const WVMF> mixture, vec2 u)
{
    int idx = sample_small_distrib<WVMF, &WVMF::w>(mixture, u[0], &u[0]);
    return mixture[idx].lobe.sample(u);
}

float vmm_merge_cost(const WVMF &lobe0, const WVMF &lobe1);

struct VMMFitSample
{
    vec3 omega; // sample direction
    float w;    // sample weight
    // Samples are typically obtained by importance sampling, and the weight of each sample
    // is its importance sampling weight divided by the average importance sampling weight from all samples.
};

struct VMMFitterArgs
{
    int max_iter = 100;
    float tol = 1e-3f;
    float min_likelihood = 1e-8f;
    float weight_prior = 0.01f;
    float mean_cosine_prior = 0.0f;
    float mean_cosine_prior_strength = 0.2f;
    float merge_threshold = 0.025f;
};

struct VMMFitterLogger
{
    int n_iter = 0;
    std::vector<float> L;
    std::vector<int> K;
};

struct SufficientStats
{
    SufficientStats() = default;
    SufficientStats(int K) : sample_w_gamma(K, 0.0f), sample_w_gamma_omega(K, vec3::Zero()) {}

    void set_to_zero();
    void add(float new_sample_w_gamma, const vec3 &new_sample_w_gamma_omega);
    void normalize();
    void combine(const SufficientStats &other, int N, int other_N);

    std::vector<float> sample_w_gamma;
    std::vector<vec3> sample_w_gamma_omega;
};

struct VMMFitter
{
    void initialize_uniform(int K);
    void initialize(std::span<const WVMF> lobes);

    void update(std::span<const VMMFitSample> samples, const VMMFitterArgs &args, VMMFitterLogger *logger = nullptr);
    void incremental_em(std::span<const VMMFitSample> samples, const VMMFitterArgs &args,
                        VMMFitterLogger *logger = nullptr);
    float log_likelihood(std::span<const VMMFitSample> samples) const;

    void merge(const VMMFitterArgs &args);

    std::vector<WVMF> mixture;
    struct
    {
        int N = 0;
        SufficientStats stats;
    } history;
};

} // namespace ks