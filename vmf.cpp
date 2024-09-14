#include "vmf.h"

namespace ks
{

void VMMFitter::initialize_uniform(int K) { mixture.resize(K, WVMF{VMF(), 1.0f / (float)K}); }

void VMMFitter::initialize(std::span<const WVMF> lobes)
{
    mixture.resize(lobes.size());
    std::copy(lobes.begin(), lobes.end(), mixture.begin());
}

void VMMFitter::update(std::span<const VMMFitSample> samples, const VMMFitterArgs &args, VMMFitterLogger *logger)
{
    incremental_em(samples, args, logger);
    merge(args);
}

void VMMFitter::incremental_em(std::span<const VMMFitSample> samples, const VMMFitterArgs &args,
                               VMMFitterLogger *logger)
{
    int N = samples.size();
    int K = mixture.size();
    std::vector<float> gamma(N * K);
    SufficientStats stats(K);
    struct
    {
        int n;
        float sample_w;
        vec3 sample_w_omega;
    } unassigned;

    float L_prev = 0.0f;
    for (int iter = 0; iter < args.max_iter; ++iter) {
        // Clear all per-iteration storage.
        std::fill(gamma.begin(), gamma.end(), 0.0f);
        stats.set_to_zero();
        unassigned.n = 0;
        unassigned.sample_w = 0.0f;
        unassigned.sample_w_omega = vec3::Zero();

        // E step
        float L = 0.0f;
        for (int n = 0; n < N; ++n) {
            float V = 0.0f;
            for (int k = 0; k < K; ++k) {
                gamma[n * K + k] = mixture[k].w * mixture[k].lobe.eval(samples[n].omega);
                V += gamma[n * K + k];
            }
            if (V < args.min_likelihood) {
                // This sample is not covered by any existing lobe.
                std::fill_n(&gamma[n * K], K, 0.0f);
                ++unassigned.n;
                unassigned.sample_w += samples[n].w;
                unassigned.sample_w_omega += samples[n].w * samples[n].omega;
            } else {
                L += samples[n].w * std::log(V);
                float V_inv = 1.0f / V;
                for (int k = 0; k < K; ++k) {
                    gamma[n * K + k] *= V_inv;

                    stats.sample_w_gamma[k] += samples[n].w * gamma[n * K + k];
                    stats.sample_w_gamma_omega[k] += samples[n].w * gamma[n * K + k] * samples[n].omega;
                }
            }
        }

        // Spawn new lobe for unassigned samples
        if (unassigned.sample_w > 0.0f) {
            stats.add(unassigned.sample_w, unassigned.sample_w_omega);
            if (history.N > 0)
                history.stats.add(0.0f, vec3::Zero());
            ++K;
            gamma.resize(N * K);
            mixture.resize(K);
        }

        // NOTE: move "normalization" after potential spawning.
        stats.normalize();

        if (logger) {
            logger->n_iter = iter + 1;
            logger->L.push_back(L);
            logger->K.push_back(K);
        }

        // Convergence criteria: relative change of log likelihood
        if (iter > 0 && std::abs(L - L_prev) / std::abs(L_prev) < args.tol) {
            break;
        }
        L_prev = L;

        // M step (MAP estimator with history)
        for (int k = 0; k < K; ++k) {
            int N_total = N + history.N;
            float w = stats.sample_w_gamma[k] * N;
            if (history.N > 0)
                w += history.stats.sample_w_gamma[k] * history.N;
            w = (args.weight_prior + w) / ((args.weight_prior * K) + N_total);

            vec3 mu;
            if (stats.sample_w_gamma[k] > 0.0f && (history.N > 0 && history.stats.sample_w_gamma[k] > 0.0f)) {
                mu = stats.sample_w_gamma_omega[k] / stats.sample_w_gamma[k];
                vec3 mu_history = history.stats.sample_w_gamma_omega[k] / history.stats.sample_w_gamma[k];
                mu = lerp(mu_history, mu, (float)N / (float)(N_total));
            } else if (stats.sample_w_gamma[k] > 0.0f) {
                mu = stats.sample_w_gamma_omega[k] / stats.sample_w_gamma[k];
            } else if (history.N > 0 && history.stats.sample_w_gamma[k] > 0.0f) {
                mu = history.stats.sample_w_gamma_omega[k] / history.stats.sample_w_gamma[k];
            } else {
                // No history and no current data. In this case the algorithm has no idea what to do.
                ASSERT(false);
            }
            float mean_cosine = mu.norm();
            mu /= mu.norm();

            mean_cosine = (mean_cosine * w * N_total + args.mean_cosine_prior * args.mean_cosine_prior_strength) /
                          (w * N_total + args.mean_cosine_prior_strength);
            float kappa = VMF::mean_cosine_to_kappa(mean_cosine);

            ASSERT(w >= 0.0f && w <= 1.0f + 1e-5f); // ocassionally w can be slightly larger than 1.0 due to rounding...
            ASSERT(mu.allFinite());
            ASSERT(std::isfinite(kappa) && kappa > 0.0f);
            mixture[k] = {VMF(mu, kappa), w};
        }
    }

    history.stats.combine(stats, history.N, N);
    //
    history.N += samples.size();
}

void SufficientStats::set_to_zero()
{
    std::fill(sample_w_gamma.begin(), sample_w_gamma.end(), 0.0f);
    std::fill(sample_w_gamma_omega.begin(), sample_w_gamma_omega.end(), vec3::Zero());
}

void SufficientStats::add(float new_sample_w_gamma, const vec3 &new_sample_w_gamma_omega)
{
    sample_w_gamma.push_back(new_sample_w_gamma);
    sample_w_gamma_omega.push_back(new_sample_w_gamma_omega);
}

void SufficientStats::normalize()
{
    float sample_w_gamma_sum_inv = 0.0f;
    for (int k = 0; k < (int)sample_w_gamma.size(); ++k)
        sample_w_gamma_sum_inv += sample_w_gamma[k];
    // Avoid divide by 0.
    if (sample_w_gamma_sum_inv > 0.0f) {
        sample_w_gamma_sum_inv = 1.0f / sample_w_gamma_sum_inv;
        for (int k = 0; k < (int)sample_w_gamma.size(); ++k) {
            sample_w_gamma[k] *= sample_w_gamma_sum_inv;
            sample_w_gamma_omega[k] *= sample_w_gamma_sum_inv;
        }
    }
}

void SufficientStats::combine(const SufficientStats &other, int N, int other_N)
{
    if (sample_w_gamma.empty()) {
        sample_w_gamma = other.sample_w_gamma;
        sample_w_gamma_omega = other.sample_w_gamma_omega;
    } else {
        for (int k = 0; k < (int)sample_w_gamma.size(); ++k) {
            sample_w_gamma[k] = (sample_w_gamma[k] * N + other.sample_w_gamma[k] * other_N) / (N + other_N);
            sample_w_gamma_omega[k] =
                (sample_w_gamma_omega[k] * N + other.sample_w_gamma_omega[k] * other_N) / (N + other_N);
        }
    }
}

float VMMFitter::log_likelihood(std::span<const VMMFitSample> samples) const
{
    int N = samples.size();
    int K = mixture.size();
    float L = 0.0f;
    for (int n = 0; n < N; ++n) {
        float V = 0.0f;
        for (int k = 0; k < K; ++k) {
            V += mixture[k].w * mixture[k].lobe.eval(samples[n].omega);
        }
        L += samples[n].w * std::log(V);
    }
    return L;
}

static float vmf_mul(const vec3 &mu0, float kappa0, float norm0, const vec3 &mu1, float kappa1, float norm1, vec3 &mu,
                     float &kappa, float &norm)
{
    mu = kappa0 * mu0 + kappa1 * mu1;
    kappa = mu.norm();

    norm = 1.0f / (4.0f * pi);
    if (kappa > 1e-3f) {
        norm = kappa / (two_pi * (1.0f - std::exp(-2.0f * kappa)));
        mu /= kappa;
    } else {
        kappa = 0.0f;
        mu = mu0;
    }

    float scale = (norm0 * norm1) / norm;
    float cos_theta0 = mu0.dot(mu);
    float cos_theta1 = mu1.dot(mu);
    scale *= std::exp(kappa0 * (cos_theta0 - 1.0f) + kappa1 * (cos_theta1 - 1.0f));

    return scale;
}

static float vmf_div_integral(const vec3 &mu0, float kappa0, float norm0, const vec3 &mu1, float kappa1, float norm1)
{
    vec3 mu = kappa0 * mu0 + kappa1 * mu1;
    float kappa = mu.norm();

    float norm = 1.0f / (4.0f * pi);
    if (kappa > 1e-3f) {
        norm = kappa / (two_pi * (1.0f - std::exp(-2.0f * kappa)));
        mu /= kappa;
    } else {
        kappa = 0.0f;
        mu = mu0;
    }

    float scale = (norm0 * norm1) / norm;
    float cos_theta0 = mu0.dot(mu);
    float cos_theta1 = mu1.dot(mu);

    // TODO: double check this. The term (1.0f - std::exp(-2.0f * kappa1)) should be squared according to paper.
    // openpgl bug here?
    // scale *= (4.0f * pi * pi * (1.0f - std::exp(-2.0f * kappa1))) / (kappa1 * kappa1);
    // scale *= std::exp((kappa0 * (cos_theta0 - 1.0f) + kappa1 * (cos_theta1 - 1.0f)) + (2.0f * kappa1));
    scale *= std::exp(kappa0 * (cos_theta0 - 1.0f) + kappa1 * (cos_theta1 - 1.0f));
    if (kappa1 > 0.0f) {
        // scale *= (4.0f * pi * pi * (1.0f - std::exp(-2.0f * kappa1))) / (kappa1 * kappa1) * std::exp(2.0f * kappa1);
        scale *= (4.0f * sqr(pi) * sqr(1.0f - std::exp(-2.0f * kappa1))) / sqr(kappa1) * std::exp(2.0f * kappa1);
    } else {
        scale *= sqr(4.0f * pi);
    }

    return scale;
}

float vmm_merge_cost(const WVMF &lobe0, const WVMF &lobe1)
{
    float w0 = lobe0.w;
    const vec3 &mu0 = lobe0.lobe.mu;
    float kappa0 = lobe0.lobe.kappa;
    float mean_cos0 = lobe0.lobe.mean_cosine();
    float norm0 = lobe0.lobe.norm_term();

    float w1 = lobe1.w;
    const vec3 &mu1 = lobe1.lobe.mu;
    float kappa1 = lobe1.lobe.kappa;
    float mean_cos1 = lobe1.lobe.mean_cosine();
    float norm1 = lobe1.lobe.norm_term();

    float w = w0 + w1;
    vec3 mu = (w0 * mean_cos0 * mu0 + w1 * mean_cos1 * mu1) / w;
    float mean_cos = mu.dot(mu);
    float kappa = 0.0f;
    float norm = 1.0f / (4.0f * pi);
    if (mean_cos > 0.f) {
        mean_cos = std::sqrt(mean_cos);
        kappa = VMF::mean_cosine_to_kappa(mean_cos);
        mu /= mean_cos;
        norm = kappa / (two_pi * (1.0f - std::exp(-2.0 * kappa)));
    } else {
        mu = mu0;
    }

    float w00 = w0 * w0;
    float kappa00;
    float norm00;
    vec3 mu00;
    float scale00 = vmf_mul(mu0, kappa0, norm0, mu0, kappa0, norm0, mu00, kappa00, norm00);

    float w11 = w1 * w1;
    float kappa11;
    float norm11;
    vec3 mu11;
    float scale11 = vmf_mul(mu1, kappa1, norm1, mu1, kappa1, norm1, mu11, kappa11, norm11);

    float w01 = w0 * w1;
    float kappa01;
    float norm01;
    vec3 mu01;
    float scale01 = vmf_mul(mu0, kappa0, norm0, mu1, kappa1, norm1, mu01, kappa01, norm01);

    float Dx2 = 0.0f;
    Dx2 += (w00 / w) * (scale00 * vmf_div_integral(mu00, kappa00, norm00, -mu, kappa, norm));
    Dx2 += (w11 / w) * (scale11 * vmf_div_integral(mu11, kappa11, norm11, -mu, kappa, norm));
    Dx2 += 2.0f * (w01 / w) * (scale01 * vmf_div_integral(mu01, kappa01, norm01, -mu, kappa, norm));
    Dx2 -= w;

    // ASSERT(std::isfinite(Dx2) && Dx2 >= 0.0f);
    // NOTE: it is possible that Dx2 ended up very big or inf because
    // when two lobes are very sharp and not aligned, by def the chi-squared divergence will be degenerate.
    // We may improve improve the metric in these cases.
    return std::isfinite(Dx2) ? Dx2 : inf;
}

void VMMFitter::merge(const VMMFitterArgs &args)
{
    while (true) {
        int K = (int)mixture.size();
        // float min_cost = inf;
        bool found_candidates = false;
        int i0, i1;
        for (i0 = 0; i0 < K; ++i0)
            for (i1 = i0 + 1; i1 < K; ++i1) {
                float cost = vmm_merge_cost(mixture[i0], mixture[i1]);
                // min_cost = std::min(min_cost, cost);
                if (cost < args.merge_threshold) {
                    found_candidates = true;
                    goto do_merge;
                }
            }
    do_merge:
        // printf("min_cost: %f\n", min_cost);
        if (!found_candidates) {
            return;
        }

        float w0 = mixture[i0].w;
        const vec3 &mu0 = mixture[i0].lobe.mu;
        float mean_cos0 = mixture[i0].lobe.mean_cosine();

        float w1 = mixture[i1].w;
        const vec3 &mu1 = mixture[i1].lobe.mu;
        float mean_cos1 = mixture[i1].lobe.mean_cosine();

        float w = w0 + w1;
        vec3 mu = (w0 * mean_cos0 * mu0 + w1 * mean_cos1 * mu1) / w;
        float mean_cos = mu.dot(mu);
        float kappa = 0.0f;
        float norm = 1.0f / (4.0f * pi);
        if (mean_cos > 0.f) {
            mean_cos = std::sqrt(mean_cos);
            kappa = VMF::mean_cosine_to_kappa(mean_cos);
            mu /= mean_cos;
            norm = kappa / (two_pi * (1.0f - std::exp(-2.0 * kappa)));
        } else {
            mu = mu0;
        }

        // write to index 0 and discard index 1 by swap-and-shrink
        mixture[i0].w = w;
        mixture[i0].lobe = VMF(mu, kappa);
        std::swap(mixture[i1], mixture.back());
        mixture.back().w = 0.0f;
        mixture.back().lobe = VMF();
        mixture.pop_back();

        history.stats.sample_w_gamma[i0] += history.stats.sample_w_gamma[i1];
        std::swap(history.stats.sample_w_gamma[i1], history.stats.sample_w_gamma.back());
        history.stats.sample_w_gamma.back() = 0.0f;
        history.stats.sample_w_gamma.pop_back();

        history.stats.sample_w_gamma_omega[i0] += history.stats.sample_w_gamma_omega[i1];
        std::swap(history.stats.sample_w_gamma_omega[i1], history.stats.sample_w_gamma_omega.back());
        history.stats.sample_w_gamma_omega.back() = vec3::Zero();
        history.stats.sample_w_gamma_omega.pop_back();
    }
}

} // namespace ks