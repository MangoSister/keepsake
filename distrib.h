#pragma once

#include "maths.h"
#include <span>

namespace ks
{

struct DistribTable
{
    DistribTable() = default;

    DistribTable(const float *f, uint32_t n);
    uint32_t sample(float u, float &prob) const;
    float sample_linear(float u, float &pdf, uint32_t &index) const;
    float pdf(uint32_t index) const;
    float pdf(float x) const;

    std::vector<float> cdf;
    float acc;
};

struct DistribTable2D
{
    DistribTable2D() = default;
    DistribTable2D(const float *f, uint32_t nx, uint32_t ny);

    vec2 sample_linear(const vec2 &u, float &pdf, vec2i *index = nullptr) const;
    float pdf(uint32_t x, uint32_t y) const;
    float pdf(const vec2 &p) const;

    std::vector<DistribTable> cond;
    DistribTable margin;
};

struct AliasTable
{
    AliasTable() = default;
    AliasTable(std::span<const float> weights);

    int sample(float u, float *pmf = nullptr, float *u_remap = nullptr) const;
    float pmf(int index) const { return bins[index].p; }

    struct Bin
    {
        float q, p;
        int alias;
    };

    std::vector<Bin> bins;
};

} // namespace ks