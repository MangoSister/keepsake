#pragma once

// Modified from pbrt.

#include "memory_util.h"

template <typename T, int log_block_size = 2>
class BlockedArray
{
  public:
    // BlockedArray Public Methods
    BlockedArray(int ures, int vres, const T *d = nullptr)
        : ures(ures), vres(vres), ublocks(round_up(ures) >> log_block_size)
    {
        int n_alloc = round_up(ures) * round_up(vres);
        constexpr size_t kCacheLine = 64;
        data = alloc_aligned<T>(n_alloc, kCacheLine);
        for (int i = 0; i < n_alloc; ++i)
            new (&data[i]) T();
        if (d)
            for (int v = 0; v < vres; ++v)
                for (int u = 0; u < ures; ++u)
                    (*this)(u, v) = d[v * ures + u];
    }
    ~BlockedArray()
    {
        for (int i = 0; i < ures * vres; ++i)
            data[i].~T();
        free_aligned(data);
    }
    constexpr int block_size() const { return 1 << log_block_size; }
    int round_up(int x) const { return (x + block_size() - 1) & ~(block_size() - 1); }
    int usize() const { return ures; }
    int vsize() const { return vres; }
    int block(int a) const { return a >> log_block_size; }
    int offset(int a) const { return (a & (block_size() - 1)); }
    T &operator()(int u, int v)
    {
        int bu = block(u), bv = block(v);
        int ou = offset(u), ov = offset(v);
        int offset = block_size() * block_size() * (ublocks * bv + bu);
        offset += block_size() * ov + ou;
        return data[offset];
    }
    const T &operator()(int u, int v) const
    {
        int bu = block(u), bv = block(v);
        int ou = offset(u), ov = offset(v);
        int offset = block_size() * block_size() * (ublocks * bv + bu);
        offset += block_size() * ov + ou;
        return data[offset];
    }
    void get_linear_array(T *a) const
    {
        for (int v = 0; v < vres; ++v)
            for (int u = 0; u < ures; ++u)
                *a++ = (*this)(u, v);
    }

  private:
    // BlockedArray Private Data
    T *data;
    const int ures, vres, ublocks;
};