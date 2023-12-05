#pragma once
#include "span.cuh"

namespace ksc
{

template <typename T>
struct indirect_span
{
    indirect_span() = default;
    CUDA_HOST_DEVICE
    indirect_span(span<T> data, span<const uint32_t> index) : data(data), index(index) {}

    CUDA_HOST_DEVICE T &operator[](size_t i) { return data[index[i]]; }
    CUDA_HOST_DEVICE const T &operator[](size_t i) const { return data[index[i]]; }

    CUDA_HOST_DEVICE size_t size() const { return index.size(); };

    span<T> data;
    span<const uint32_t> index;
};

} // namespace ksc