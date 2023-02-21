#pragma once

#include "maths.h"

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

    vec2 sample_linear(const vec2 &u, float &pdf) const;
    float pdf(uint32_t x, uint32_t y) const;
    float pdf(const vec2 &p) const;

    std::vector<DistribTable> cond;
    DistribTable margin;
};

} // namespace ks