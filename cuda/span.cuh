#pragma once
#include "basic.cuh"
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits.h>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace ksc
{

namespace span_internal
{

// Wrappers for access to container data pointers.
template <typename C>
CUDA_HOST_DEVICE inline constexpr auto GetDataImpl(C &c, char) noexcept -> decltype(c.data())
{
    return c.data();
}

template <typename C>
CUDA_HOST_DEVICE inline constexpr auto GetData(C &c) noexcept -> decltype(GetDataImpl(c, 0))
{
    return GetDataImpl(c, 0);
}

// Detection idioms for size() and data().
template <typename C>
using HasSize = std::is_integral<typename std::decay_t<decltype(std::declval<C &>().size())>>;

// We want to enable conversion from vector<T*> to span<const T* const> but
// disable conversion from vector<Derived> to span<Base>. Here we use
// the fact that U** is convertible to Q* const* if and only if Q is the same
// type or a more cv-qualified version of U.  We also decay the result type of
// data() to avoid problems with classes which have a member function data()
// which returns a reference.
template <typename T, typename C>
using HasData = std::is_convertible<typename std::decay_t<decltype(GetData(std::declval<C &>()))> *, T *const *>;

} // namespace span_internal

inline constexpr std::size_t dynamic_extent = (std::size_t)-1;

// span implementation partially based on absl::Span from Google's Abseil library.
template <typename T>
class span
{
  public:
    // Used to determine whether a Span can be constructed from a container of
    // type C.
    template <typename C>
    using EnableIfConvertibleFrom =
        typename std::enable_if_t<span_internal::HasData<T, C>::value && span_internal::HasSize<C>::value>;

    // Used to SFINAE-enable a function when the slice elements are const.
    template <typename U>
    using EnableIfConstView = typename std::enable_if_t<std::is_const_v<T>, U>;

    // Used to SFINAE-enable a function when the slice elements are mutable.
    template <typename U>
    using EnableIfMutableView = typename std::enable_if_t<!std::is_const_v<T>, U>;

    using value_type = typename std::remove_cv_t<T>;
    using iterator = T *;
    using const_iterator = const T *;

    CUDA_HOST_DEVICE span() : ptr(nullptr), n(0) {}
    CUDA_HOST_DEVICE span(T *ptr, size_t n) : ptr(ptr), n(n) {}
    template <size_t N>
    CUDA_HOST_DEVICE span(T (&a)[N]) : span(a, N)
    {}
    CUDA_HOST_DEVICE span(std::initializer_list<value_type> v) : span(v.begin(), v.size()) {}

    // Explicit reference constructor for a mutable `span<T>` type. Can be
    // replaced with MakeSpan() to infer the type parameter.
    template <typename V, typename X = EnableIfConvertibleFrom<V>, typename Y = EnableIfMutableView<V>>
    CUDA_HOST_DEVICE explicit span(V &v) noexcept : span(v.data(), v.size())
    {}

    // Hack: explicit constructors for std::vector to work around warnings
    // about calling a host function (e.g. vector::size()) form a
    // host+device function (the regular span constructor.)
    template <typename V>
    span(std::vector<V> &v) noexcept : span(v.data(), v.size())
    {}
    template <typename V>
    span(const std::vector<V> &v) noexcept : span(v.data(), v.size())
    {}

    // Implicit reference constructor for a read-only `span<const T>` type
    template <typename V, typename X = EnableIfConvertibleFrom<V>, typename Y = EnableIfConstView<V>>
    CUDA_HOST_DEVICE constexpr span(const V &v) noexcept : span(v.data(), v.size())
    {}

    CUDA_HOST_DEVICE iterator begin() { return ptr; }
    CUDA_HOST_DEVICE iterator end() { return ptr + n; }
    CUDA_HOST_DEVICE const_iterator begin() const { return ptr; }
    CUDA_HOST_DEVICE const_iterator end() const { return ptr + n; }

    CUDA_HOST_DEVICE T &operator[](size_t i)
    {
        CUDA_ASSERT(i < size());
        return ptr[i];
    }
    CUDA_HOST_DEVICE const T &operator[](size_t i) const
    {
        CUDA_ASSERT(i < size());
        return ptr[i];
    }

    CUDA_HOST_DEVICE size_t size() const { return n; };
    CUDA_HOST_DEVICE bool empty() const { return size() == 0; }
    CUDA_HOST_DEVICE T *data() { return ptr; }
    CUDA_HOST_DEVICE const T *data() const { return ptr; }

    CUDA_HOST_DEVICE T front() const { return ptr[0]; }
    CUDA_HOST_DEVICE T back() const { return ptr[n - 1]; }

    CUDA_HOST_DEVICE void remove_prefix(size_t count)
    {
        // assert(size() >= count);
        ptr += count;
        n -= count;
    }
    CUDA_HOST_DEVICE void remove_suffix(size_t count)
    {
        // assert(size() > = count);
        n -= count;
    }

    CUDA_HOST_DEVICE span subspan(size_t pos, size_t count = dynamic_extent)
    {
        size_t np = count < (size() - pos) ? count : (size() - pos);
        return span(ptr + pos, np);
    }

  private:
    T *ptr;
    size_t n;
};

template <int &...ExplicitArgumentBarrier, typename T>
CUDA_HOST_DEVICE inline constexpr span<T> MakeSpan(T *ptr, size_t size) noexcept
{
    return span<T>(ptr, size);
}

template <int &...ExplicitArgumentBarrier, typename T>
CUDA_HOST_DEVICE inline span<T> MakeSpan(T *begin, T *end) noexcept
{
    return span<T>(begin, end - begin);
}

template <int &...ExplicitArgumentBarrier, typename T>
inline span<T> MakeSpan(std::vector<T> &v) noexcept
{
    return span<T>(v.data(), v.size());
}

template <int &...ExplicitArgumentBarrier, typename C>
CUDA_HOST_DEVICE inline constexpr auto MakeSpan(C &c) noexcept
    -> decltype(MakeSpan(span_internal::GetData(c), c.size()))
{
    return MakeSpan(span_internal::GetData(c), c.size());
}

template <int &...ExplicitArgumentBarrier, typename T, size_t N>
CUDA_HOST_DEVICE inline constexpr span<T> MakeSpan(T (&array)[N]) noexcept
{
    return span<T>(array, N);
}

template <int &...ExplicitArgumentBarrier, typename T>
CUDA_HOST_DEVICE inline constexpr span<const T> MakeConstSpan(T *ptr, size_t size) noexcept
{
    return span<const T>(ptr, size);
}

template <int &...ExplicitArgumentBarrier, typename T>
CUDA_HOST_DEVICE inline span<const T> MakeConstSpan(T *begin, T *end) noexcept
{
    return span<const T>(begin, end - begin);
}

template <int &...ExplicitArgumentBarrier, typename T>
inline span<const T> MakeConstSpan(const std::vector<T> &v) noexcept
{
    return span<const T>(v.data(), v.size());
}

template <int &...ExplicitArgumentBarrier, typename C>
CUDA_HOST_DEVICE inline constexpr auto MakeConstSpan(const C &c) noexcept -> decltype(MakeSpan(c))
{
    return MakeSpan(c);
}

template <int &...ExplicitArgumentBarrier, typename T, size_t N>
CUDA_HOST_DEVICE inline constexpr span<const T> MakeConstSpan(const T (&array)[N]) noexcept
{
    return span<const T>(array, N);
}

} // namespace ksc