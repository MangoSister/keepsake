#pragma once
#include "aabb.h"
#include "maths.h"
#include <vector>

namespace ks
{

struct SummedAreaTable
{
    SummedAreaTable(std::span<const float> values, vec2i res) : nx(res.x()), ny(res.y())
    {
        table.resize(values.size());
        table[0] = values[0];
        for (int x = 1; x < nx; ++x)
            table[x] = values[x] + table[x - 1];
        for (int y = 1; y < ny; ++y)
            table[y * nx] = values[y * nx] + table[(y - 1) * nx];
        for (int y = 1; y < ny; ++y)
            for (int x = 1; x < nx; ++x)
                table[y * nx + x] = (values[y * nx + x] + table[y * nx + (x - 1)] + table[(y - 1) * nx + x] -
                                     table[(y - 1) * nx + (x - 1)]);
    }

    float sum(AABB2 bound) const
    {
        // Use double; SAT is prone to catastrophic cancellation.
        double s = (lookup(vec2(bound.max.x(), bound.max.y())) - lookup(vec2(bound.min.x(), bound.max.y()))) +
                   (lookup(vec2(bound.min.x(), bound.min.y())) - lookup(vec2(bound.max.x(), bound.min.y())));
        return std::max((float)s, 0.0f);
    }

    double lookup(vec2 p) const
    {
        // NOTE: offset by 1
        vec2i res_1(nx + 1, ny + 1);
        vec2i lo, hi;
        vec2 w;
        lerp_helper<2>(p.data(), res_1.data(), WrapMode::Clamp, TickMode::Boundary, lo.data(), hi.data(), w.data());
        int x0 = lo[0], y0 = lo[1];
        int x1 = hi[0], y1 = hi[1];
        float wx = w[0], wy = w[1];
        return fetch(x0, y0) * (1.0 - wx) * (1.0 - wy) + fetch(x1, y0) * (wx) * (1.0 - wy) +
               fetch(x0, y1) * (1.0 - wx) * (wy) + fetch(x1, y1) * (wx) * (wy);
    }

    double fetch(int x, int y) const
    {
        // NOTE: offset by 1
        if (x == 0 || y == 0)
            return 0;

        x = std::min(x - 1, nx - 1);
        y = std::min(y - 1, ny - 1);
        return table[y * nx + x];
    }

    std::vector<double> table;
    int nx, ny;
};
} // namespace ks