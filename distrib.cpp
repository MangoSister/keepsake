#include "distrib.h"
#include "assertion.h"
#include <algorithm>
#include <numeric>

namespace ks
{

DistribTable::DistribTable(const float *f, uint32_t n) : cdf(n + 1)
{
    cdf[0] = 0;
    float invn = 1.0f / (float)n;
    for (uint32_t i = 1; i < n + 1; ++i) {
        cdf[i] = cdf[i - 1] + f[i - 1] * invn;
    }
    acc = cdf[n];
    // ASSERT(acc > 0.0f);
    if (acc == 0.0f) {
        for (uint32_t i = 0; i < n + 1; ++i) {
            cdf[i] = (float)i / (float)n;
        }
    } else {
        float invacc = 1.0f / acc;
        for (uint32_t i = 0; i < n + 1; ++i) {
            cdf[i] *= invacc;
        }
    }

    cdf[n] = 1.0f; // Avoid numerical inaccuracy.
}

uint32_t DistribTable::sample(float u, float &prob) const
{
    u = std::clamp(u, 0.0f, std::nextafter(1.0f, 0.0f));
    ASSERT(u >= 0.0f && u < 1.0f);
    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    uint32_t index = (uint32_t)std::distance(cdf.begin(), std::prev(it));
    prob = cdf[index + 1] - cdf[index];
    return index;
}

float DistribTable::sample_linear(float u, float &pdf, uint32_t &index) const
{
    u = std::clamp(u, 0.0f, std::nextafter(1.0f, 0.0f));
    ASSERT(u >= 0.0f && u < 1.0f);
    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    index = (uint32_t)std::distance(cdf.begin(), std::prev(it));
    float n = (float)(cdf.size() - 1);
    pdf = (cdf[index + 1] - cdf[index]) * n;
    float du = (u - cdf[index]) / (cdf[index + 1] - cdf[index]);

    return (index + du) / n;
}

float DistribTable::pdf(uint32_t index) const
{
    uint32_t n = (uint32_t)(cdf.size() - 1);
    ASSERT(index < n);
    return (cdf[index + 1] - cdf[index]) * (float)n;
}

float DistribTable::pdf(float x) const
{
    ASSERT(x >= 0.0 && x <= 1.0f);
    int n = (int)(cdf.size() - 1);
    uint32_t idx = (uint32_t)std::clamp((int)std::floor(x * n), 0, n - 1);
    return pdf(idx);
}

DistribTable2D::DistribTable2D(const float *f, uint32_t nx, uint32_t ny)
{
    cond = std::vector<DistribTable>(ny);
    for (uint32_t y = 0; y < ny; ++y) {
        cond[y] = DistribTable(f + (y * nx), nx);
    }
    std::vector<float> marginPmf(ny);
    for (uint32_t y = 0; y < ny; ++y) {
        marginPmf[y] = cond[y].acc;
    }
    margin = DistribTable(marginPmf.data(), ny);
}

vec2 DistribTable2D::sample_linear(const vec2 &u, float &pdf, vec2i *index) const
{
    float pdfMargin, pdfCond;
    uint32_t uIndex, vIndex;
    float vSample = margin.sample_linear(u[1], pdfMargin, vIndex);
    float uSample = cond[vIndex].sample_linear(u[0], pdfCond, uIndex);
    pdf = pdfMargin * pdfCond;
    if (index) {
        (*index) = vec2i(uIndex, vIndex);
    }
    return vec2(uSample, vSample);
}

float DistribTable2D::pdf(uint32_t x, uint32_t y) const
{
    return margin.pdf(y) * cond[y].pdf(x);
    // return cond[y].pdf(x) * cond[y].acc / margin.acc;
}

float DistribTable2D::pdf(const vec2 &p) const
{
    int x = std::clamp(int(p[0] * (cond[0].cdf.size() - 1)), 0, (int)cond[0].cdf.size() - 2);
    int y = std::clamp(int(p[1] * cond.size()), 0, (int)cond.size() - 1);
    return pdf(x, y);
}

AliasTable::AliasTable(std::span<const float> weights) : bins(weights.size())
{
    // Normalize _weights_ to compute alias table PDF
    float sum = std::accumulate(weights.begin(), weights.end(), 0.);
    ASSERT(sum > 0.0f);
    for (size_t i = 0; i < weights.size(); ++i)
        bins[i].p = weights[i] / sum;

    // Create alias table work lists
    struct Outcome
    {
        float pHat;
        size_t index;
    };
    std::vector<Outcome> under, over;
    for (size_t i = 0; i < bins.size(); ++i) {
        // Add outcome _i_ to an alias table work list
        float pHat = bins[i].p * bins.size();
        if (pHat < 1)
            under.push_back(Outcome{pHat, i});
        else
            over.push_back(Outcome{pHat, i});
    }

    // Process under and over work item together
    while (!under.empty() && !over.empty()) {
        // Remove items _un_ and _ov_ from the alias table work lists
        Outcome un = under.back(), ov = over.back();
        under.pop_back();
        over.pop_back();

        // Initialize probability and alias for _un_
        bins[un.index].q = un.pHat;
        bins[un.index].alias = ov.index;

        // Push excess probability on to work list
        float pExcess = un.pHat + ov.pHat - 1;
        if (pExcess < 1)
            under.push_back(Outcome{pExcess, ov.index});
        else
            over.push_back(Outcome{pExcess, ov.index});
    }

    // Handle remaining alias table work items
    while (!over.empty()) {
        Outcome ov = over.back();
        over.pop_back();
        bins[ov.index].q = 1;
        bins[ov.index].alias = -1;
    }
    while (!under.empty()) {
        Outcome un = under.back();
        under.pop_back();
        bins[un.index].q = 1;
        bins[un.index].alias = -1;
    }
}

int AliasTable::sample(float u, float *pmf, float *u_remap) const
{
    // Compute alias table _offset_ and remapped random sample _up_
    size_t offset = std::min((size_t)floor(u * bins.size()), bins.size() - 1);
    float up = std::min(u * bins.size() - offset, fp32_before_one);

    if (up < bins[offset].q) {
        // Return sample for alias table at _offset_
        ASSERT(bins[offset].p > 0);
        if (pmf)
            *pmf = bins[offset].p;
        if (u_remap)
            *u_remap = std::min(up / bins[offset].q, fp32_before_one);
        return offset;

    } else {
        // Return sample for alias table at _alias[offset]_
        int alias = bins[offset].alias;
        ASSERT(alias >= 0);
        ASSERT(bins[alias].p > 0);
        if (pmf)
            *pmf = bins[alias].p;
        if (u_remap)
            *u_remap = std::min((up - bins[offset].q) / (1 - bins[offset].q), fp32_before_one);
        return alias;
    }
}

} // namespace ks