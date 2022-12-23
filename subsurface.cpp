#include "subsurface.h"
#include "maths.h"
#include "ray.h"
#include "rng.h"
#include "scene.h"

KS_NAMESPACE_BEGIN

inline color3 safe_divide_color(const color3 &a, const color3 &b) { return (b == 0.0f).select(color3::Zero(), a / b); }

inline color3 volume_color_transmittance(const color3 &sigma, float t) { return exp(-sigma * t); }

inline float volume_channel_get(const color3 &value, int channel) { return value[channel]; }

inline int volume_sample_channel(const color3 &albedo, const color3 &throughput, float rand, color3 *pdf)
{
    /* Sample color channel proportional to throughput and single scattering
     * albedo, to significantly reduce noise with many bounce, following:
     *
     * "Practical and Controllable Subsurface Scattering for Production Path
     *  Tracing". Matt Jen-Yuan Chiang, Peter Kutz, Brent Burley. SIGGRAPH 2016. */
    color3 weights = (throughput * albedo).abs();
    float sum_weights = weights.sum();

    if (sum_weights > 0.0f) {
        *pdf = weights / sum_weights;
    } else {
        *pdf = color3::Constant(1.0f / (float)color3::SizeAtCompileTime);
    }

    float pdf_sum = 0.0f;
    for (int i = 0; i < color3::SizeAtCompileTime; ++i) {
        pdf_sum += (*pdf)[i];
        if (rand < pdf_sum) {
            return i;
        }
    }
    return color3::SizeAtCompileTime - 1;
}

inline float safe_powf(float a, float b)
{
    if (a == 0.0 && b == 0.0f)
        return 0.0f;
    return std::pow(a, b);
}

/* Given cosine between rays, return probability density that a photon bounces
 * to that direction. The g parameter controls how different it is from the
 * uniform sphere. g=0 uniform diffuse-like, g=1 close to sharp single ray. */
inline float single_peaked_henyey_greenstein(float cos_theta, float g)
{
    return ((1.0f - g * g) / safe_powf(1.0f + g * g - 2.0f * g * cos_theta, 1.5f)) * (inv_pi * 0.25f);
};

vec3 henyey_greenstrein_sample(const vec3 &D, float g, float randu, float randv, float *pdf)
{
    /* match pdf for small g */
    float cos_theta;
    bool isotropic = fabsf(g) < 1e-3f;

    if (isotropic) {
        cos_theta = (1.0f - 2.0f * randu);
        if (pdf) {
            *pdf = inv_pi * 0.25f;
        }
    } else {
        float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * randu);
        cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
        if (pdf) {
            *pdf = single_peaked_henyey_greenstein(cos_theta, g);
        }
    }

    float sin_theta = safe_sqrt(1.0f - cos_theta * cos_theta);
    float phi = two_pi * randv;
    vec3 dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    vec3 T, B;
    orthonormal_basis(D, T, B);
    dir = dir.x() * T + dir.y() * B + dir.z() * D;

    return dir;
}

void subsurface_random_walk_remap(const float albedo, const float d, float g, float *sigma_t, float *alpha)
{
    /* Compute attenuation and scattering coefficients from albedo. */
    const float g2 = g * g;
    const float g3 = g2 * g;
    const float g4 = g3 * g;
    const float g5 = g4 * g;
    const float g6 = g5 * g;
    const float g7 = g6 * g;

    const float A = 1.8260523782f + -1.28451056436f * g + -1.79904629312f * g2 + 9.19393289202f * g3 +
                    -22.8215585862f * g4 + 32.0234874259f * g5 + -23.6264803333f * g6 + 7.21067002658f * g7;
    const float B =
        4.98511194385f +
        0.127355959438f * expf(31.1491581433f * g + -201.847017512f * g2 + 841.576016723f * g3 + -2018.09288505f * g4 +
                               2731.71560286f * g5 + -1935.41424244f * g6 + 559.009054474f * g7);
    const float C = 1.09686102424f + -0.394704063468f * g + 1.05258115941f * g2 + -8.83963712726f * g3 +
                    28.8643230661f * g4 + -46.8802913581f * g5 + 38.5402837518f * g6 + -12.7181042538f * g7;
    const float D = 0.496310210422f + 0.360146581622f * g + -2.15139309747f * g2 + 17.8896899217f * g3 +
                    -55.2984010333f * g4 + 82.065982243f * g5 + -58.5106008578f * g6 + 15.8478295021f * g7;
    const float E =
        4.23190299701f +
        0.00310603949088f * expf(76.7316253952f * g + -594.356773233f * g2 + 2448.8834203f * g3 + -5576.68528998f * g4 +
                                 7116.60171912f * g5 + -4763.54467887f * g6 + 1303.5318055f * g7);
    const float F = 2.40602999408f + -2.51814844609f * g + 9.18494908356f * g2 + -79.2191708682f * g3 +
                    259.082868209f * g4 + -403.613804597f * g5 + 302.85712436f * g6 + -87.4370473567f * g7;

    const float blend = powf(albedo, 0.25f);

    *alpha = (1.0f - blend) * A * powf(atanf(B * albedo), C) + blend * D * powf(atanf(E * albedo), F);
    *alpha = clamp(*alpha, 0.0f, 0.999999f); // because of numerical precision

    float sigma_t_prime = 1.0f / fmaxf(d, 1e-16f);
    *sigma_t = sigma_t_prime / (1.0f - g);
}

void subsurface_random_walk_coefficients(const color3 &albedo, const color3 &radius, const float anisotropy,
                                         color3 *sigma_t, color3 *alpha, color3 *throughput)
{
    for (int i = 0; i < color3::SizeAtCompileTime; ++i) {
        subsurface_random_walk_remap(albedo[i], radius[i], anisotropy, &(*sigma_t)[i], &(*alpha)[i]);
    }

    // TODO: Should I do this ???
    /* Throughput already contains closure weight at this point, which includes the
     * albedo, as well as closure mixing and Fresnel weights. Divide out the albedo
     * which will be added through scattering. */
    // *throughput = safe_divide_color(*throughput, albedo);

    /* With low albedo values (like 0.025) we get diffusion_length 1.0 and
     * infinite phase functions. To avoid a sharp discontinuity as we go from
     * such values to 0.0, increase alpha and reduce the throughput to compensate. */
    const float min_alpha = 0.2f;
    for (int i = 0; i < color3::SizeAtCompileTime; ++i) {
        if ((*alpha)[i] < min_alpha) {
            (*throughput)[i] *= (*alpha)[i] / min_alpha;
            (*alpha)[i] = min_alpha;
        }
    }
}

/* References for Dwivedi sampling:
 *
 * [1] "A Zero-variance-based Sampling Scheme for Monte Carlo Subsurface Scattering"
 * by Jaroslav Křivánek and Eugene d'Eon (SIGGRAPH 2014)
 * https://cgg.mff.cuni.cz/~jaroslav/papers/2014-zerovar/
 *
 * [2] "Improving the Dwivedi Sampling Scheme"
 * by Johannes Meng, Johannes Hanika, and Carsten Dachsbacher (EGSR 2016)
 * https://cg.ivd.kit.edu/1951.php
 *
 * [3] "Zero-Variance Theory for Efficient Subsurface Scattering"
 * by Eugene d'Eon and Jaroslav Křivánek (SIGGRAPH 2020)
 * https://iliyan.com/publications/RenderingCourse2020
 */

inline float eval_phase_dwivedi(float v, float phase_log, float cos_theta)
{
    /* Eq. 9 from [2] using precomputed log((v + 1) / (v - 1)) */
    return 1.0f / ((v - cos_theta) * phase_log);
}

inline float sample_phase_dwivedi(float v, float phase_log, float rand)
{
    /* Based on Eq. 10 from [2]: `v - (v + 1) * pow((v - 1) / (v + 1), rand)`
     * Since we're already pre-computing `phase_log = log((v + 1) / (v - 1))` for the evaluation,
     * we can implement the power function like this. */
    return v - (v + 1.0f) * expf(-rand * phase_log);
}

inline float diffusion_length_dwivedi(float alpha)
{
    /* Eq. 67 from [3] */
    return 1.0f / sqrtf(1.0f - powf(alpha, 2.44294f - 0.0215813f * alpha + 0.578637f / alpha));
}

inline vec3 direction_from_cosine(const vec3 &D, float cos_theta, float randv)
{
    float sin_theta = safe_sqrt(1.0f - cos_theta * cos_theta);
    float phi = two_pi * randv;
    vec3 dir(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    vec3 T, B;
    orthonormal_basis(D, T, B);
    return dir.x() * T + dir.y() * B + dir.z() * D;
}

inline color3 subsurface_random_walk_pdf(const color3 &sigma_t, float t, bool hit, color3 *transmittance)
{
    color3 T = volume_color_transmittance(sigma_t, t);
    if (transmittance) {
        *transmittance = T;
    }
    return hit ? T : sigma_t * T;
}

inline float bssrdf_dipole_compute_Rd(float alpha_prime, float fourthirdA)
{
    float s = sqrtf(3.0f * (1.0f - alpha_prime));
    return 0.5f * alpha_prime * (1.0f + expf(-fourthirdA * s)) * expf(-s);
}

static float bssrdf_dipole_compute_alpha_prime(float rd, float fourthirdA)
{
    /* Little Newton solver. */
    if (rd < 1e-4f) {
        return 0.0f;
    }
    if (rd >= 0.995f) {
        return 0.999999f;
    }

    float x0 = 0.0f;
    float x1 = 1.0f;
    float xmid, fmid;

    constexpr const int max_num_iterations = 12;
    for (int i = 0; i < max_num_iterations; ++i) {
        xmid = 0.5f * (x0 + x1);
        fmid = bssrdf_dipole_compute_Rd(xmid, fourthirdA);
        if (fmid < rd) {
            x0 = xmid;
        } else {
            x1 = xmid;
        }
    }

    return xmid;
}

struct SubsurfaceProfile
{
    color3 albedo;
    color3 radius;
    float anisotropy;
    float ior; // Should keep consistent with the surface BSDF.
    float rfr_entry_prob;
};

inline void bssrdf_setup_radius(SubsurfaceProfile &profile)
{
    /* Adjust radius based on IOR and albedo. */
    const float eta = profile.ior;
    const float inv_eta = 1.0f / eta;
    const float F_dr = inv_eta * (-1.440f * inv_eta + 0.710f) + 0.668f + 0.0636f * eta;
    const float fourthirdA = (4.0f / 3.0f) * (1.0f + F_dr) / (1.0f - F_dr); /* From Jensen's `Fdr` ratio formula. */

    color3 alpha_prime;
    for (int i = 0; i < color3::SizeAtCompileTime; ++i) {
        alpha_prime[i] = bssrdf_dipole_compute_alpha_prime(profile.albedo[i], fourthirdA);
    }

    profile.radius *= sqrt(3.0f * (color3::Ones() - alpha_prime));
}

/* Define the below variable to get the similarity code active,
 * and the value represents the cutoff level */
#define SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL 9
constexpr int BSSRDF_MAX_BOUNCES = 256;
constexpr float VOLUME_THROUGHPUT_EPSILON = 1e-6f;

// Following the convention, D is the entry direction (used to be diffuse entry).
// wi is the random walk output direction.
// Pass in the current throughput to be accumulated during the random walk.
// Return whether the random walk is successful.
bool subsurface_random_walk(SubsurfaceProfile profile, const LocalGeometry &local_geometry, const Intersection &entry,
                            vec3 D, RNG &rng, color3 &throughput, SceneHit &exit, vec3 &wi)
{
    bssrdf_setup_radius(profile);

    const vec3 &P = entry.p;          // entry position
    const vec3 &N = entry.sh_frame.n; // entry shading normal
    const vec3 &Ng = entry.frame.n;   // entry geometric normal
    if (rng.next() < profile.rfr_entry_prob) {
        // Do nothing because D is already from rfr entry.
        // TODO:
        // Per Christophe, using D as-is may not get the best result in terms of matching references,
        // even if it seems to be the more "principal" way.
        // Empirically, re-sample refractive D given wo with a fixed 1.0 alpha GGX gives better result.
    } else {
        // Override D with classic cosine entry.
        D = sample_cosine_hemisphere(rng.next2d(), -N);
    }
    //
    if (-Ng.dot(D) <= 0.0f) {
        return false;
    }

    /* Setup ray. */
    Ray ray = spawn_ray<OffsetType::NextBounce>(P, D, Ng, 0.0f, inf);
    // TODO: avoid self intersection on first bounce

    /* Convert subsurface to volume coefficients.
     * The single-scattering albedo is named alpha to avoid confusion with the surface albedo. */
    const color3 &albedo = profile.albedo;
    const color3 &radius = profile.radius;
    const float anisotropy = profile.anisotropy;

    color3 sigma_t, alpha;
    subsurface_random_walk_coefficients(albedo, radius, anisotropy, &sigma_t, &alpha, &throughput);
    color3 sigma_s = sigma_t * alpha;

    /* Theoretically it should be better to use the exact alpha for the channel we're sampling at
     * each bounce, but in practice there doesn't seem to be a noticeable difference in exchange
     * for making the code significantly more complex and slower (if direction sampling depends on
     * the sampled channel, we need to compute its PDF per-channel and consider it for MIS later on).
     *
     * Since the strength of the guided sampling increases as alpha gets lower, using a value that
     * is too low results in fireflies while one that's too high just gives a bit more noise.
     * Therefore, the code here uses the highest of the three albedos to be safe. */
    float diffusion_length = diffusion_length_dwivedi(alpha.maxCoeff());

    if (diffusion_length == 1.0f) {
        /* With specific values of alpha the length might become 1, which in asymptotic makes phase to
         * be infinite. After first bounce it will cause throughput to be 0. Do early output, avoiding
         * numerical issues and extra unneeded work. */
        return false;
    }

    /* Precompute term for phase sampling. */
    const float phase_log = logf((diffusion_length + 1.0f) / (diffusion_length - 1.0f));

    /* Random walk until we hit the surface again. */
    bool hit = false;
    bool have_opposite_interface = false;
    float opposite_distance = 0.0f;

    /* TODO: Disable for `alpha > 0.999` or so? */
    /* Our heuristic, a compromise between guiding and classic. */
    const float guided_fraction = 1.0f - fmaxf(0.5f, powf(fabsf(anisotropy), 0.125f));

#ifdef SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL
    color3 sigma_s_star = sigma_s * (1.0f - anisotropy);
    color3 sigma_t_star = sigma_t - sigma_s + sigma_s_star;
    color3 sigma_t_org = sigma_t;
    color3 sigma_s_org = sigma_s;
    const float anisotropy_org = anisotropy;
    const float guided_fraction_org = guided_fraction;
#endif

    for (int bounce = 0; bounce < BSSRDF_MAX_BOUNCES; bounce++) {
#ifdef SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL
        // shadow with local variables according to depth
        float anisotropy, guided_fraction;
        color3 sigma_s, sigma_t;
        if (bounce <= SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL) {
            anisotropy = anisotropy_org;
            guided_fraction = guided_fraction_org;
            sigma_t = sigma_t_org;
            sigma_s = sigma_s_org;
        } else {
            anisotropy = 0.0f;
            guided_fraction = 0.75f; // back to isotropic heuristic from Blender
            sigma_t = sigma_t_star;
            sigma_s = sigma_s_star;
        }
#endif

        /* Sample color channel, use MIS with balance heuristic. */
        float rphase = rng.next();
        color3 channel_pdf;
        int channel = volume_sample_channel(alpha, throughput, rphase, &channel_pdf);
        float sample_sigma_t = volume_channel_get(sigma_t, channel);
        float randt = rng.next();

        /* We need the result of the ray-cast to compute the full guided PDF, so just remember the
         * relevant terms to avoid recomputing them later. */
        float backward_fraction = 0.0f;
        float forward_pdf_factor = 0.0f;
        float forward_stretching = 1.0f;
        float backward_pdf_factor = 0.0f;
        float backward_stretching = 1.0f;

        /* For the initial ray, we already know the direction, so just do classic distance sampling. */
        if (bounce > 0) {
            /* Decide whether we should use guided or classic sampling. */
            bool guided = (rng.next() < guided_fraction);

            /* Determine if we want to sample away from the incoming interface.
             * This only happens if we found a nearby opposite interface, and the probability for it
             * depends on how close we are to it already.
             * This probability term comes from the recorded presentation of [3]. */
            bool guide_backward = false;
            if (have_opposite_interface) {
                /* Compute distance of the random walk between the tangent plane at the starting point
                 * and the assumed opposite interface (the parallel plane that contains the point we
                 * found in our ray query for the opposite side). */
                float x = std::clamp((ray.origin - P).dot(-N), 0.0f, opposite_distance);
                backward_fraction = 1.0f / (1.0f + expf((opposite_distance - 2.0f * x) / diffusion_length));
                guide_backward = rng.next() < backward_fraction;
            }

            /* Sample scattering direction. */
            const vec2 rand_scatter = rng.next2d();
            float cos_theta;
            float hg_pdf;
            if (guided) {
                cos_theta = sample_phase_dwivedi(diffusion_length, phase_log, rand_scatter.x());
                /* The backwards guiding distribution is just mirrored along `sd->N`, so swapping the
                 * sign here is enough to sample from that instead. */
                if (guide_backward) {
                    cos_theta = -cos_theta;
                }
                vec3 newD = direction_from_cosine(N, cos_theta, rand_scatter.y());
                hg_pdf = single_peaked_henyey_greenstein(ray.dir.dot(newD), anisotropy);
                ray.dir = newD;
            } else {
                vec3 newD = henyey_greenstrein_sample(ray.dir, anisotropy, rand_scatter.x(), rand_scatter.y(), &hg_pdf);
                cos_theta = newD.dot(N);
                ray.dir = newD;
            }

            /* Compute PDF factor caused by phase sampling (as the ratio of guided / classic).
             * Since phase sampling is channel-independent, we can get away with applying a factor
             * to the guided PDF, which implicitly means pulling out the classic PDF term and letting
             * it cancel with an equivalent term in the numerator of the full estimator.
             * For the backward PDF, we again reuse the same probability distribution with a sign swap.
             */
            forward_pdf_factor = (1.0f / two_pi) * eval_phase_dwivedi(diffusion_length, phase_log, cos_theta) / hg_pdf;
            backward_pdf_factor =
                (1.0f / two_pi) * eval_phase_dwivedi(diffusion_length, phase_log, -cos_theta) / hg_pdf;

            /* Prepare distance sampling.
             * For the backwards case, this also needs the sign swapped since now directions against
             * `sd->N` (and therefore with negative cos_theta) are preferred. */
            forward_stretching = (1.0f - cos_theta / diffusion_length);
            backward_stretching = (1.0f + cos_theta / diffusion_length);
            if (guided) {
                sample_sigma_t *= guide_backward ? backward_stretching : forward_stretching;
            }
        }

        /* Sample distance along ray. */
        float t = -logf(1.0f - randt) / sample_sigma_t;

        /* On the first bounce, we use the ray-cast to check if the opposite side is nearby.
         * If yes, we will later use backwards guided sampling in order to have a decent
         * chance of connecting to it.
         * TODO: Maybe use less than 10 times the mean free path? */
        if (bounce == 0) {
            ray.tmax = std::max(t, 10.0f / ((sigma_t).minCoeff()));
        } else {
            ray.tmax = t;
            // TODO:
            /* After the first bounce the object can intersect the same surface again */
            // ray.self.object = OBJECT_NONE;
            // ray.self.prim = PRIM_NONE;
        }
        // scene_intersect_local(kg, &ray, &ss_isect, object, NULL, 1);
        // hit = (ss_isect.num_hits > 0);
        hit = local_geometry.intersect1(ray, exit);

        if (hit) {
            ray.tmax = exit.it.thit;
        }

        if (bounce == 0) {
            /* Check if we hit the opposite side. */
            if (hit) {
                have_opposite_interface = true;
                opposite_distance = (ray.origin + ray.tmax * ray.dir - P).dot(-N);
            }
            /* Apart from the opposite side check, we were supposed to only trace up to distance t,
             * so check if there would have been a hit in that case. */
            hit = ray.tmax < t;
        }

        /* Use the distance to the exit point for the throughput update if we found one. */
        if (hit) {
            t = ray.tmax;
        }

        /* Advance to new scatter location. */
        ray.origin += t * ray.dir;

        color3 transmittance;
        color3 pdf = subsurface_random_walk_pdf(sigma_t, t, hit, &transmittance);
        if (bounce > 0) {
            /* Compute PDF just like we do for classic sampling, but with the stretched sigma_t. */
            color3 guided_pdf = subsurface_random_walk_pdf(forward_stretching * sigma_t, t, hit, nullptr);

            if (have_opposite_interface) {
                /* First step of MIS: Depending on geometry we might have two methods for guided
                 * sampling, so perform MIS between them. */
                color3 back_pdf = subsurface_random_walk_pdf(backward_stretching * sigma_t, t, hit, nullptr);
                guided_pdf = lerp(guided_pdf * forward_pdf_factor, back_pdf * backward_pdf_factor, backward_fraction);
            } else {
                /* Just include phase sampling factor otherwise. */
                guided_pdf *= forward_pdf_factor;
            }

            /* Now we apply the MIS balance heuristic between the classic and guided sampling. */
            pdf = lerp(pdf, guided_pdf, guided_fraction);
        }

        /* Finally, we're applying MIS again to combine the three color channels.
         * Altogether, the MIS computation combines up to nine different estimators:
         * {classic, guided, backward_guided} x {r, g, b} */
        throughput *= (hit ? transmittance : sigma_s * transmittance) / ((channel_pdf * pdf).sum());

        if (hit) {
            /* If we hit the surface, we are done. */
            break;
        } else if ((throughput).maxCoeff() < VOLUME_THROUGHPUT_EPSILON) {
            /* Avoid unnecessary work and precision issue when throughput gets really small. */
            break;
        }
    }

    wi = ray.dir;
    ASSERT(throughput.allFinite() && (throughput >= 0.0f).all());
    return hit;
}

color3 LambertianSubsurfaceExitAdapter::eval(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wi.z() <= 0.0f)
        return color3::Zero();

    color3 f = color3::Constant(inv_pi * wi.z());
    f *= sqr(eta);
    return f;
}

color3 LambertianSubsurfaceExitAdapter::sample(const vec3 &wo, vec3 &wi, const Intersection &it, const vec2 &u,
                                               float &pdf) const
{
    wi = sample_cosine_hemisphere(u);
    pdf = wi.z() * inv_pi;
    color3 f_beta = color3::Constant(sqr(eta));

    return f_beta;
}

float LambertianSubsurfaceExitAdapter::pdf(const vec3 &wo, const vec3 &wi, const Intersection &it) const
{
    if (wi.z() <= 0.0f)
        return 0.0f;
    return wi.z() * inv_pi;
}

bool BSSRDF::sample(const LocalGeometry &local_geometry, const Intersection &entry, vec3 D, RNG &rng,
                    color3 &throughput, SceneHit &exit, vec3 &wi) const
{
    SubsurfaceProfile profile;
    profile.albedo = (*albedo)(entry);
    profile.albedo = clamp(profile.albedo, color3::Zero(), color3::Ones());
    profile.radius = (*radius)(entry);
    profile.anisotropy = anisotropy;
    profile.ior = ior;
    profile.rfr_entry_prob = rfr_entry_prob;
    if (!subsurface_random_walk(profile, local_geometry, entry, D, rng, throughput, exit, wi))
        return false;

    return true;
}

std::unique_ptr<BSSRDF> create_bssrdf(const ConfigArgs &args)
{
    std::unique_ptr<BSSRDF> bssrdf = std::make_unique<BSSRDF>();
    bssrdf->albedo = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["albedo"]);
    bssrdf->radius = args.asset_table().create_in_place<ShaderField3>("shader_field_3", args["radius"]);
    bssrdf->anisotropy = args.load_float("anisotropy");
    bssrdf->ior = args.load_float("ior");
    bssrdf->rfr_entry_prob = args.load_float("rfr_entry_prob");
    return bssrdf;
}

KS_NAMESPACE_END