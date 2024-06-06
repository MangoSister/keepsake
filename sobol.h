#pragma once

#include "hash.h"
#include "rng.h"
#include <cstdint>

namespace ks
{

// Porting three sampling sequences from pbrt 4
// All considered state-of-the-art and have similar convergence rate

static constexpr int n_sobol_dimensions = 1024;
static constexpr int sobol_matrix_size = 52;
extern uint32_t sobol_matrices_32[n_sobol_dimensions * sobol_matrix_size];

struct FastOwenScrambler
{
    FastOwenScrambler(uint32_t seed) : seed(seed) {}
    // FastOwenScrambler Public Methods
    uint32_t operator()(uint32_t v) const
    {
        v = reverse_bits_32(v);
        v ^= v * 0x3d20adea;
        v += seed;
        v *= (seed >> 16) | 1;
        v ^= v * 0x05526c56;
        v ^= v * 0x53a22864;
        return reverse_bits_32(v);
    }

    uint32_t seed;
};

template <typename R>
inline float sobol_sample(int64_t a, int dimension, R randomizer)
{
    ASSERT(dimension < n_sobol_dimensions);
    ASSERT(a >= 0 && a < (1ull << sobol_matrix_size));
    // Compute initial Sobol\+$'$ sample _v_ using generator matrices
    uint32_t v = 0;
    for (int i = dimension * sobol_matrix_size; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= sobol_matrices_32[i];

    // Randomize Sobol\+$'$ sample and return floating-point value
    v = randomizer(v);
    return std::min(v * 0x1p-32f, fp32_before_one);
}

inline int permutation_element(uint32_t i, uint32_t l, uint32_t p)
{
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

// - Owen-scrambled sobol (0, 2)
// We need to allow samples to be arranged on other domains than 2D image space (pixels), e.g. 3D voxels, 4D BRDF etc.
template <size_t M>
struct OwenScrambledSobol02
{
    // non power-of-2 samples_per_entry is suboptimal
    OwenScrambledSobol02(int samples_per_entry, int seed) : samples_per_entry(samples_per_entry), seed(seed) {}

    void start_entry(arri<M> e, int index, int dim)
    {
        entry = e;
        sample_index = index;
        dimension = dim;
    }

    float next()
    {
        // Get permuted index for current pixel sample
        uint64_t hash = hash(entry, dimension, seed);
        int index = permutation_element(sample_index, samples_per_entry, hash);

        int dim = dimension++;
        // Return randomized 1D van der Corput sample for dimension _dim_
        return sobol_sample(index, 0, FastOwenScrambler((uint32_t)(hash >> 32)));
    }

    vec2 next2d()
    {
        // Get permuted index for current pixel sample
        uint64_t hash = hash(entry, dimension, seed);
        int index = permutation_element(sample_index, samples_per_entry, hash);

        int dim = dimension;
        dimension += 2;
        // Return randomized 2D Sobol\+$'$ sample
        return vec2(sobol_sample(index, 0, FastOwenScrambler(uint32_t(hash))),
                    sobol_sample(index, 1, FastOwenScrambler(uint32_t(hash >> 32))));
    }

    arri<M> entry;
    int sample_index = 0;
    int dimension = 0;
    int samples_per_entry;
    int seed;
};

// OwenScrambledSobol02 + permuted Morton indices: distribute error as high-frequency blue noise. looks better
// (and easier to denoise?), but assumes first two dimensions is in screen space (just for actual rendering instead of
// generic sampling)
struct OwenScrambledSobol02Morton
{
    OwenScrambledSobol02Morton(int samples_per_pixel, arr2i full_resolution, int seed) : seed(seed)
    {
        // non power-of-2 samples_per_pixel is suboptimal
        log2_samples_per_pixel = log2_int(samples_per_pixel);
        int res = round_up_pow2(std::max(full_resolution.x(), full_resolution.y()));
        int log4_samples_per_pixel = (log2_samples_per_pixel + 1) / 2;
        n_base4_digits = log2_int(res) + log4_samples_per_pixel;
    }

    void start_pixel(arr2i pixel, int index, int dim)
    {
        dimension = dim;
        morton_index = (encode_morton_2(pixel.x(), pixel.y()) << log2_samples_per_pixel) | index;
    }

    float next()
    {
        uint64_t sample_index = get_sample_index();
        ++dimension;
        // Generate 1D Sobol\+$'$ sample at _sampleIndex_
        uint32_t sample_hash = hash(dimension, seed);
        return sobol_sample(sample_index, 0, FastOwenScrambler(sample_hash));
    }

    vec2 next2d()
    {
        uint64_t sample_index = get_sample_index();
        dimension += 2;
        // Generate 2D Sobol\+$'$ sample at _sampleIndex_
        uint64_t bits = hash(dimension, seed);
        uint32_t sample_hash[2] = {uint32_t(bits), uint32_t(bits >> 32)};
        return vec2(sobol_sample(sample_index, 0, FastOwenScrambler(sample_hash[0])),
                    sobol_sample(sample_index, 1, FastOwenScrambler(sample_hash[1])));
    }

    uint64_t get_sample_index() const
    {
        // Define the full set of 4-way permutations in _permutations_
        constexpr uint8_t permutations[24][4] = {{0, 1, 2, 3},
                                                 {0, 1, 3, 2},
                                                 {0, 2, 1, 3},
                                                 {0, 2, 3, 1},
                                                 // Define remaining 20 4-way permutations
                                                 {0, 3, 2, 1},
                                                 {0, 3, 1, 2},
                                                 {1, 0, 2, 3},
                                                 {1, 0, 3, 2},
                                                 {1, 2, 0, 3},
                                                 {1, 2, 3, 0},
                                                 {1, 3, 2, 0},
                                                 {1, 3, 0, 2},
                                                 {2, 1, 0, 3},
                                                 {2, 1, 3, 0},
                                                 {2, 0, 1, 3},
                                                 {2, 0, 3, 1},
                                                 {2, 3, 0, 1},
                                                 {2, 3, 1, 0},
                                                 {3, 1, 2, 0},
                                                 {3, 1, 0, 2},
                                                 {3, 2, 1, 0},
                                                 {3, 2, 0, 1},
                                                 {3, 0, 2, 1},
                                                 {3, 0, 1, 2}

        };

        uint64_t sample_index = 0;
        // Apply random permutations to full base-4 digits
        bool pow_2_samples = log2_samples_per_pixel & 1;
        int last_digit = pow_2_samples ? 1 : 0;
        for (int i = n_base4_digits - 1; i >= last_digit; --i) {
            // Randomly permute $i$th base-4 digit in _mortonIndex_
            int digitShift = 2 * i - (pow_2_samples ? 1 : 0);
            int digit = (morton_index >> digitShift) & 3;
            // Choose permutation _p_ to use for _digit_
            uint64_t higher_digits = morton_index >> (digitShift + 2);
            int p = (mix_bits(higher_digits ^ (0x55555555u * dimension)) >> 24) % 24;

            digit = permutations[p][digit];
            sample_index |= uint64_t(digit) << digitShift;
        }

        // Handle power-of-2 (but not 4) sample count
        if (pow_2_samples) {
            int digit = morton_index & 1;
            sample_index |= digit ^ (mix_bits((morton_index >> 1) ^ (0x55555555u * dimension)) & 1);
        }

        return sample_index;
    }

    // This is only for screen space rendering.
    int dimension = 0;
    int seed, log2_samples_per_pixel, n_base4_digits;
    uint64_t morton_index = 0;
};

// Sampler for unidirectional PT style renderer.
// Deterministic rng state for easier debugging.
// Renderer should try to use sobol and align dimensions (pixel, bsdf, light, etc)
// Use RNG only when we need an undertermined number of random numbers (volume sampling, subsurface, stochastic eval,
// anything else fancy)
struct PTRenderSampler
{
    PTRenderSampler(arr2u full_resolution, arr2u pixel, uint32_t samples_per_pixel, uint32_t sample_index,
                    uint32_t seed)
        : sobol((int)samples_per_pixel, full_resolution.cast<int>(), (int)seed), rng(hash(pixel, seed))
    {
        sobol.start_pixel(pixel.cast<int>(), (int)sample_index, 0);
        // Assume each sample uses <65536 dims...
        rng.advance(sample_index * 65536ull);
    }

    OwenScrambledSobol02Morton sobol;
    RNG rng;
};

// TODO:
// - pmj02bn (allow adaptive sampling)
// struct PMJ02BN;

// Helmer, Andrew, Per H.Christensen, and Andrew Kensler."Stochastic Generation of (t, s) Sample Sequences."
// EGSR(DL).2021.
// https://github.com/Andrew-Helmer/stochastic-generation
// Helmer's stateless iterative stochastic Owen-scrambled Sobol:
// Support up to 64 dimensions

// Query for an arbitrary coordinate from a scrambled Sobol (0,2)-sequence. Seed
// can be the same across all dimensions, or it can be different for each
// dimension.
//
// nd is the number of dimensions shared by the same seed value, to get separate
// hashed values for each sample index. If a different seed is provided for each
// dimension, this can be used with nd=1.
float sobol_owen(int idx, int dim, uint32_t seed, int nd);

} // namespace ks