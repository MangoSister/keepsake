#pragma once
#include <algorithm>
#if _WIN32 || _WIN64
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
// FUCK: this includes windows.h on windows
// https://github.com/oneapi-src/oneTBB/issues/573
#include <tbb/combinable.h>
#if _WIN32 || _WIN64
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef ERROR
#undef ERROR
#endif
#endif
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/spin_mutex.h>
#include <thread>

namespace ks
{

inline int num_system_cores() { return std::max(1u, std::thread::hardware_concurrency()); }

void init_parallel(int nthreads = 0);

template <typename Index, typename Func>
void parallel_for(const Index N, const Func &func, Index grainsize = 1)
{
    auto body = [&](const tbb::blocked_range<Index> &r) {
        for (Index i = r.begin(); i != r.end(); ++i)
            func(i);
    };
    tbb::parallel_for(tbb::blocked_range<Index>(0, N, grainsize), body);
}

template <typename RandomAccessIterator>
void parallel_sort(RandomAccessIterator begin, RandomAccessIterator end)
{
    tbb::parallel_sort(begin, end);
}

template <typename RandomAccessIterator, typename Compare>
void parallel_sort(RandomAccessIterator begin, RandomAccessIterator end, const Compare &comp)
{
    tbb::parallel_sort(begin, end, comp);
}

template <typename T>
class Combinable
{
  public:
    Combinable() = default;

    template <typename Func>
    explicit Combinable(Func func_init) : cb(func_init)
    {}

    Combinable(const Combinable &other) = default;
    Combinable &operator=(const Combinable &other) = default;
    Combinable(Combinable &&other) = default;
    Combinable &operator=(Combinable &&other) = default;

    template <typename Func>
    T combine(Func func_combine)
    {
        return cb.combine(func_combine);
    }

    template <typename Func>
    void combine_each(Func func_combine)
    {
        return cb.combine_each(func_combine);
    }

    T &local() { return cb.local(); }

    T &local(bool &exists) { return cb.local(exists); }

    void clear() { cb.clear(); }

  private:
    tbb::combinable<T> cb;
};

template <typename Func>
void parallel_tile_2d(int width, int height, const Func &pixel_func)
{
    constexpr int tile_width = 4;
    constexpr int tile_height = 4;
    int num_tiles_x = (width + tile_width - 1) / tile_width;
    int num_tiles_y = (height + tile_height - 1) / tile_height;
    int num_tiles = num_tiles_x * num_tiles_y;
    parallel_for(num_tiles, [&](int tile_idx) {
        int tile_idx_y = tile_idx / num_tiles_x;
        int tile_idx_x = tile_idx - tile_idx_y * num_tiles_x;

        int y_start = tile_idx_y * tile_height;
        int y_end = std::min(y_start + tile_height, height);
        int x_start = tile_idx_x * tile_width;
        int x_end = std::min(x_start + tile_width, width);
        for (int y = y_start; y < y_end; ++y)
            for (int x = x_start; x < x_end; ++x)
                pixel_func(x, y);
    });
}

using spin_lock = tbb::spin_mutex;

} // namespace ks