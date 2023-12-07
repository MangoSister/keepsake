#pragma once

#include "basic.cuh"
#include <new>
#include <type_traits>

namespace ksc
{

template <typename T>
class optional
{
  public:
    using value_type = T;

    optional() = default;
    CUDA_HOST_DEVICE
    optional(const T &v) : set(true) { new (ptr()) T(v); }
    CUDA_HOST_DEVICE
    optional(T &&v) : set(true) { new (ptr()) T(std::move(v)); }
    CUDA_HOST_DEVICE
    optional(const optional &v) : set(v.has_value())
    {
        if (v.has_value())
            new (ptr()) T(v.value());
    }
    CUDA_HOST_DEVICE
    optional(optional &&v) : set(v.has_value())
    {
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            v.reset();
        }
    }

    CUDA_HOST_DEVICE
    optional &operator=(const T &v)
    {
        reset();
        new (ptr()) T(v);
        set = true;
        return *this;
    }
    CUDA_HOST_DEVICE
    optional &operator=(T &&v)
    {
        reset();
        new (ptr()) T(std::move(v));
        set = true;
        return *this;
    }
    CUDA_HOST_DEVICE
    optional &operator=(const optional &v)
    {
        reset();
        if (v.has_value()) {
            new (ptr()) T(v.value());
            set = true;
        }
        return *this;
    }
    CUDA_HOST_DEVICE
    optional &operator=(optional &&v)
    {
        reset();
        if (v.has_value()) {
            new (ptr()) T(std::move(v.value()));
            set = true;
            v.reset();
        }
        return *this;
    }

    CUDA_HOST_DEVICE
    ~optional() { reset(); }

    CUDA_HOST_DEVICE
    explicit operator bool() const { return set; }

    CUDA_HOST_DEVICE
    T value_or(const T &alt) const { return set ? value() : alt; }

    CUDA_HOST_DEVICE
    T *operator->() { return &value(); }
    CUDA_HOST_DEVICE
    const T *operator->() const { return &value(); }
    CUDA_HOST_DEVICE
    T &operator*() { return value(); }
    CUDA_HOST_DEVICE
    const T &operator*() const { return value(); }
    CUDA_HOST_DEVICE
    T &value()
    {
        KSC_ASSERT(set);
        return *ptr();
    }
    CUDA_HOST_DEVICE
    const T &value() const
    {
        KSC_ASSERT(set);
        return *ptr();
    }

    CUDA_HOST_DEVICE
    void reset()
    {
        if (set) {
            value().~T();
            set = false;
        }
    }

    CUDA_HOST_DEVICE
    bool has_value() const { return set; }

  private:
#ifdef __NVCC__
    // Work-around NVCC bug
    CUDA_HOST_DEVICE
    T *ptr() { return reinterpret_cast<T *>(&optionalValue); }
    CUDA_HOST_DEVICE
    const T *ptr() const { return reinterpret_cast<const T *>(&optionalValue); }
#else
    CUDA_HOST_DEVICE
    T *ptr() { return std::launder(reinterpret_cast<T *>(&optionalValue)); }
    CUDA_HOST_DEVICE
    const T *ptr() const { return std::launder(reinterpret_cast<const T *>(&optionalValue)); }
#endif

    std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
    bool set = false;
};

} // namespace ksc