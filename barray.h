#pragma once

// Modified from pbrt.

#include "memory_util.h"
#include <algorithm>
#include <memory>
#include <utility>

KS_NAMESPACE_BEGIN

template <typename T>
struct BlockedArray
{
    ~BlockedArray()
    {
        int n_alloc = round_up(ures) * round_up(vres) * stride;
        std::destroy_n(data, n_alloc);
        free_aligned(data);
    }

    BlockedArray() = default;

    BlockedArray(int ures, int vres, int stride, const T *d = nullptr, int log_block_size = 2)
        : ures(ures), vres(vres), stride(stride), log_block_size(log_block_size)
    {
        ublocks = round_up(ures) >> log_block_size;
        int n_alloc = round_up(ures) * round_up(vres) * stride;
        constexpr size_t cache_line = 64;
        data = alloc_aligned<T>(n_alloc, cache_line);
        for (int v = 0; v < vres; ++v)
            for (int u = 0; u < ures; ++u)
                for (int c = 0; c < stride; ++c)
                    std::construct_at(&(*this)(u, v, c), d ? d[(v * ures + u) * stride + c] : T());
    }

    BlockedArray(const BlockedArray &other)
        : ures(other.ures), vres(other.vres), stride(other.stride), ublocks(other.ublocks),
          log_block_size(log_block_size)
    {
        int n_alloc = round_up(ures) * round_up(vres);
        constexpr size_t cache_line = 64;
        data = alloc_aligned<T>(n_alloc, cache_line);
        for (int i = 0; i < n_alloc; ++i)
            new (&data[i]) T();
        std::copy(other.data, other.data + n_alloc, data);
    }

    friend void swap(BlockedArray &first, BlockedArray &second)
    {
        using std::swap;
        swap(first.data, second.data);
        swap(first.ures, second.ures);
        swap(first.vres, second.vres);
        swap(first.ublocks, second.ublocks);
        swap(first.log_block_size, second.log_block_size);
        swap(first.stride, second.stride);
    }

    BlockedArray(BlockedArray &&other) noexcept : BlockedArray() { swap(*this, other); }

    BlockedArray &operator=(BlockedArray other)
    {
        swap(*this, other);
        return *this;
    }

    int block_size() const { return 1 << log_block_size; }
    int round_up(int x) const { return (x + block_size() - 1) & ~(block_size() - 1); }
    int block(int a) const { return a >> log_block_size; }
    int offset(int a) const { return (a & (block_size() - 1)); }

    const T &operator()(int u, int v, int c = 0) const
    {
        int bu = block(u), bv = block(v);
        int ou = offset(u), ov = offset(v);
        int offset = block_size() * block_size() * (ublocks * bv + bu);
        offset += block_size() * ov + ou;
        return data[offset * stride + c];
    }

    T &operator()(int u, int v, int c = 0) { return const_cast<T &>(std::as_const(*this)(u, v, c)); }

    const T *fetch_multi(int u, int v) const { return &(*this)(u, v, 0); }

    T *fetch_multi(int u, int v) { return const_cast<T &>(std::as_const(*this).fetch_multi(u, v)); }

    void copy_to_linear_array(T *a) const
    {
        for (int v = 0; v < vres; ++v)
            for (int u = 0; u < ures; ++u)
                for (int c = 0; c < stride; ++c)
                    *a++ = (*this)(u, v, c);
    }

    T *data = nullptr;
    int ures = 0;
    int vres = 0;
    int ublocks = 0;
    int log_block_size = 2;
    int stride = 0;
};

KS_NAMESPACE_END