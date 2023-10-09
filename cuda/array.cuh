#pragma once

#include "basic.cuh"
#include <cstddef>

namespace ksc
{

template <typename T, int N>
class array;

// Specialization for zero element arrays (to make MSVC happy)
template <typename T>
class array<T, 0>
{
  public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using size_t = std::size_t;

    array() = default;

    CUDA_HOST_DEVICE void fill(const T &v) { assert(!"should never be called"); }

    CUDA_HOST_DEVICE bool operator==(const array<T, 0> &a) const { return true; }
    CUDA_HOST_DEVICE bool operator!=(const array<T, 0> &a) const { return false; }

    CUDA_HOST_DEVICE iterator begin() { return nullptr; }
    CUDA_HOST_DEVICE iterator end() { return nullptr; }
    CUDA_HOST_DEVICE const_iterator begin() const { return nullptr; }
    CUDA_HOST_DEVICE const_iterator end() const { return nullptr; }

    CUDA_HOST_DEVICE size_t size() const { return 0; }

    CUDA_HOST_DEVICE T &operator[](size_t i)
    {
        assert(!"should never be called");
        static T t;
        return t;
    }
    CUDA_HOST_DEVICE const T &operator[](size_t i) const
    {
        assert(!"should never be called");
        static T t;
        return t;
    }

    CUDA_HOST_DEVICE T *data() { return nullptr; }
    CUDA_HOST_DEVICE const T *data() const { return nullptr; }
};

template <typename T, int N>
class array
{
  public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using size_t = std::size_t;

    array() = default;
    CUDA_HOST_DEVICE array(std::initializer_list<T> v)
    {
        size_t i = 0;
        for (const T &val : v)
            values[i++] = val;
    }

    CUDA_HOST_DEVICE void fill(const T &v)
    {
        for (int i = 0; i < N; ++i)
            values[i] = v;
    }

    CUDA_HOST_DEVICE bool operator==(const array<T, N> &a) const
    {
        for (int i = 0; i < N; ++i)
            if (values[i] != a.values[i])
                return false;
        return true;
    }
    CUDA_HOST_DEVICE bool operator!=(const array<T, N> &a) const { return !(*this == a); }

    CUDA_HOST_DEVICE iterator begin() { return values; }
    CUDA_HOST_DEVICE iterator end() { return values + N; }
    CUDA_HOST_DEVICE const_iterator begin() const { return values; }
    CUDA_HOST_DEVICE const_iterator end() const { return values + N; }

    CUDA_HOST_DEVICE size_t size() const { return N; }

    CUDA_HOST_DEVICE T &operator[](size_t i) { return values[i]; }
    CUDA_HOST_DEVICE const T &operator[](size_t i) const { return values[i]; }

    CUDA_HOST_DEVICE T *data() { return values; }
    CUDA_HOST_DEVICE const T *data() const { return values; }

  private:
    T values[N] = {};
};

} // namespace ksc