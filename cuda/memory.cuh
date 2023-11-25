#pragma once
#include "basic.cuh"
#include "span.cuh"
#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace ksc
{

void *cuda_alloc_managed(size_t size);

void cuda_free_managed(void *ptr);

//////////////////////////////////////////////////////////////////////////////////////////
// unique_ptr implementation stolen from MSVC but with CUDA_HOST_DEVICE tags

struct _Zero_then_variadic_args_t
{
    explicit _Zero_then_variadic_args_t() = default;
}; // tag type for value-initializing first, constructing second from remaining args

struct _One_then_variadic_args_t
{
    explicit _One_then_variadic_args_t() = default;
}; // tag type for constructing first from one arg, constructing second from remaining args

template <class _Ty1, class _Ty2, bool = std::is_empty_v<_Ty1> && !std::is_final_v<_Ty1>>
class _Compressed_pair final : private _Ty1
{ // store a pair of values, deriving from empty first
  public:
    _Ty2 _Myval2;

    using _Mybase = _Ty1; // for visualization

    template <class... _Other2>
    constexpr explicit _Compressed_pair(_Zero_then_variadic_args_t, _Other2 &&..._Val2) noexcept(
        std::conjunction_v<std::is_nothrow_default_constructible<_Ty1>,
                           std::is_nothrow_constructible<_Ty2, _Other2...>>)
        : _Ty1(), _Myval2(std::forward<_Other2>(_Val2)...)
    {}

    template <class _Other1, class... _Other2>
    constexpr _Compressed_pair(_One_then_variadic_args_t, _Other1 &&_Val1, _Other2 &&..._Val2) noexcept(
        std::conjunction_v<std::is_nothrow_constructible<_Ty1, _Other1>,
                           std::is_nothrow_constructible<_Ty2, _Other2...>>)
        : _Ty1(std::forward<_Other1>(_Val1)), _Myval2(std::forward<_Other2>(_Val2)...)
    {}

    constexpr _Ty1 &_Get_first() noexcept { return *this; }

    constexpr const _Ty1 &_Get_first() const noexcept { return *this; }
};

template <class _Ty>
struct default_delete;

template <class _Ty, class _Dx = default_delete<_Ty>>
class unique_ptr;

template <class _Ty>
struct default_delete
{ // default deleter for unique_ptr
    constexpr default_delete() noexcept = default;

    template <class _Ty2, std::enable_if_t<std::is_convertible_v<_Ty2 *, _Ty *>, int> = 0>
    inline default_delete(const default_delete<_Ty2> &) noexcept
    {}

    inline void operator()(_Ty *_Ptr) const noexcept /* strengthened */
    {                                                // delete a pointer
        static_assert(0 < sizeof(_Ty), "can't delete an incomplete type");
        delete _Ptr;
    }
};

template <class _Ty>
struct default_delete<_Ty[]>
{ // default deleter for unique_ptr to array of unknown size
    constexpr default_delete() noexcept = default;

    template <class _Uty, std::enable_if_t<std::is_convertible_v<_Uty (*)[], _Ty (*)[]>, int> = 0>
    inline default_delete(const default_delete<_Uty[]> &) noexcept
    {}

    template <class _Uty, std::enable_if_t<std::is_convertible_v<_Uty (*)[], _Ty (*)[]>, int> = 0>
    inline void operator()(_Uty *_Ptr) const noexcept /* strengthened */
    {                                                 // delete a pointer
        static_assert(0 < sizeof(_Uty), "can't delete an incomplete type");
        delete[] _Ptr;
    }
};

template <class _Ty, class _Dx_noref, class = void>
struct _Get_deleter_pointer_type
{ // provide fallback
    using type = _Ty *;
};

template <class _Ty, class _Dx_noref>
struct _Get_deleter_pointer_type<_Ty, _Dx_noref, std::void_t<typename _Dx_noref::pointer>>
{ // get _Dx_noref::pointer
    using type = typename _Dx_noref::pointer;
};

template <class _Dx2>
using _Unique_ptr_enable_default_t =
    std::enable_if_t<std::conjunction_v<std::negation<std::is_pointer<_Dx2>>, std::is_default_constructible<_Dx2>>,
                     int>;

template <class _Ty, class _Dx /* = default_delete<_Ty> */>
class unique_ptr
{ // non-copyable pointer to an object
  public:
    using pointer = typename _Get_deleter_pointer_type<_Ty, std::remove_reference_t<_Dx>>::type;
    using element_type = _Ty;
    using deleter_type = _Dx;

    template <class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0>
    constexpr unique_ptr() noexcept : _Mypair(_Zero_then_variadic_args_t{})
    {}

    template <class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0>
    constexpr unique_ptr(nullptr_t) noexcept : _Mypair(_Zero_then_variadic_args_t{})
    {}

    inline unique_ptr &operator=(nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    // The Standard depicts these constructors that accept pointer as taking type_identity_t<pointer> to inhibit CTAD.
    // Since pointer is an opaque type alias in our implementation, it inhibits CTAD without extra decoration.
    template <class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0>
    inline explicit unique_ptr(pointer _Ptr) noexcept : _Mypair(_Zero_then_variadic_args_t{}, _Ptr)
    {}

    template <class _Dx2 = _Dx, std::enable_if_t<std::is_constructible_v<_Dx2, const _Dx2 &>, int> = 0>
    inline unique_ptr(pointer _Ptr, const _Dx &_Dt) noexcept : _Mypair(_One_then_variadic_args_t{}, _Dt, _Ptr)
    {}

    template <
        class _Dx2 = _Dx,
        std::enable_if_t<std::conjunction_v<std::negation<std::is_reference<_Dx2>>, std::is_constructible<_Dx2, _Dx2>>,
                         int> = 0>
    inline unique_ptr(pointer _Ptr, _Dx &&_Dt) noexcept : _Mypair(_One_then_variadic_args_t{}, std::move(_Dt), _Ptr)
    {}

    template <class _Dx2 = _Dx,
              std::enable_if_t<std::conjunction_v<std::is_reference<_Dx2>,
                                                  std::is_constructible<_Dx2, std::remove_reference_t<_Dx2>>>,
                               int> = 0>
    unique_ptr(pointer, std::remove_reference_t<_Dx> &&) = delete;

    template <class _Dx2 = _Dx, std::enable_if_t<std::is_move_constructible_v<_Dx2>, int> = 0>
    inline unique_ptr(unique_ptr &&_Right) noexcept
        : _Mypair(_One_then_variadic_args_t{}, std::forward<_Dx>(_Right.get_deleter()), _Right.release())
    {}

    template <
        class _Ty2, class _Dx2,
        std::enable_if_t<std::conjunction_v<std::negation<std::is_array<_Ty2>>,
                                            std::is_convertible<typename unique_ptr<_Ty2, _Dx2>::pointer, pointer>,
                                            std::conditional_t<std::is_reference_v<_Dx>, std::is_same<_Dx2, _Dx>,
                                                               std::is_convertible<_Dx2, _Dx>>>,
                         int> = 0>
    inline unique_ptr(unique_ptr<_Ty2, _Dx2> &&_Right) noexcept
        : _Mypair(_One_then_variadic_args_t{}, std::forward<_Dx2>(_Right.get_deleter()), _Right.release())
    {}

    template <
        class _Ty2, class _Dx2,
        std::enable_if_t<std::conjunction_v<std::negation<std::is_array<_Ty2>>, std::is_assignable<_Dx &, _Dx2>,
                                            std::is_convertible<typename unique_ptr<_Ty2, _Dx2>::pointer, pointer>>,
                         int> = 0>
    inline unique_ptr &operator=(unique_ptr<_Ty2, _Dx2> &&_Right) noexcept
    {
        reset(_Right.release());
        _Mypair._Get_first() = std::forward<_Dx2>(_Right._Mypair._Get_first());
        return *this;
    }

    template <class _Dx2 = _Dx, std::enable_if_t<std::is_move_assignable_v<_Dx2>, int> = 0>
    inline unique_ptr &operator=(unique_ptr &&_Right) noexcept
    {
        if (this != std::addressof(_Right)) {
            reset(_Right.release());
            _Mypair._Get_first() = std::forward<_Dx>(_Right._Mypair._Get_first());
        }
        return *this;
    }

    inline void swap(unique_ptr &_Right) noexcept
    {
        _Swap_adl(_Mypair._Myval2, _Right._Mypair._Myval2);
        _Swap_adl(_Mypair._Get_first(), _Right._Mypair._Get_first());
    }

    inline ~unique_ptr() noexcept
    {
        if (_Mypair._Myval2) {
            _Mypair._Get_first()(_Mypair._Myval2);
        }
    }

    inline _Dx &get_deleter() noexcept { return _Mypair._Get_first(); }

    inline const _Dx &get_deleter() const noexcept { return _Mypair._Get_first(); }

    inline std::add_lvalue_reference_t<_Ty> operator*() const noexcept(noexcept(*std::declval<pointer>()))
    {
        return *_Mypair._Myval2;
    }

    CUDA_HOST_DEVICE inline pointer operator->() const noexcept { return _Mypair._Myval2; }

    CUDA_HOST_DEVICE inline pointer get() const noexcept { return _Mypair._Myval2; }

    CUDA_HOST_DEVICE inline explicit operator bool() const noexcept { return static_cast<bool>(_Mypair._Myval2); }

    inline pointer release() noexcept { return std::exchange(_Mypair._Myval2, nullptr); }

    inline void reset(pointer _Ptr = nullptr) noexcept
    {
        pointer _Old = std::exchange(_Mypair._Myval2, _Ptr);
        if (_Old) {
            _Mypair._Get_first()(_Old);
        }
    }

    unique_ptr(const unique_ptr &) = delete;
    unique_ptr &operator=(const unique_ptr &) = delete;

  private:
    template <class, class>
    friend class unique_ptr;

    _Compressed_pair<_Dx, pointer> _Mypair;
};

template <class _Ty, class _Dx>
class unique_ptr<_Ty[], _Dx>
{ // non-copyable pointer to an array object
  public:
    using pointer = typename _Get_deleter_pointer_type<_Ty, std::remove_reference_t<_Dx>>::type;
    using element_type = _Ty;
    using deleter_type = _Dx;

    template <class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0>
    constexpr unique_ptr() noexcept : _Mypair(_Zero_then_variadic_args_t{})
    {}

    template <class _Uty, class _Is_nullptr = std::is_same<_Uty, nullptr_t>>
    using _Enable_ctor_reset =
        std::enable_if_t<std::is_same_v<_Uty, pointer>                   //
                             || _Is_nullptr::value                       //
                             || (std::is_same_v<pointer, element_type *> //
                                 && std::is_pointer_v<_Uty>              //
                                 && std::is_convertible_v<std::remove_pointer_t<_Uty> (*)[], element_type (*)[]>),
                         int>;

    template <class _Uty, class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0, _Enable_ctor_reset<_Uty> = 0>
    inline explicit unique_ptr(_Uty _Ptr) noexcept : _Mypair(_Zero_then_variadic_args_t{}, _Ptr)
    {}

    template <class _Uty, class _Dx2 = _Dx, std::enable_if_t<std::is_constructible_v<_Dx2, const _Dx2 &>, int> = 0,
              _Enable_ctor_reset<_Uty> = 0>
    inline unique_ptr(_Uty _Ptr, const _Dx &_Dt) noexcept : _Mypair(_One_then_variadic_args_t{}, _Dt, _Ptr)
    {}

    template <
        class _Uty, class _Dx2 = _Dx,
        std::enable_if_t<std::conjunction_v<std::negation<std::is_reference<_Dx2>>, std::is_constructible<_Dx2, _Dx2>>,
                         int> = 0,
        _Enable_ctor_reset<_Uty> = 0>
    inline unique_ptr(_Uty _Ptr, _Dx &&_Dt) noexcept : _Mypair(_One_then_variadic_args_t{}, std::move(_Dt), _Ptr)
    {}

    template <class _Uty, class _Dx2 = _Dx,
              std::enable_if_t<std::conjunction_v<std::is_reference<_Dx2>,
                                                  std::is_constructible<_Dx2, std::remove_reference_t<_Dx2>>>,
                               int> = 0>
    unique_ptr(_Uty, std::remove_reference_t<_Dx> &&) = delete;

    template <class _Dx2 = _Dx, std::enable_if_t<std::is_move_constructible_v<_Dx2>, int> = 0>
    inline unique_ptr(unique_ptr &&_Right) noexcept
        : _Mypair(_One_then_variadic_args_t{}, std::forward<_Dx>(_Right.get_deleter()), _Right.release())
    {}

    template <class _Dx2 = _Dx, std::enable_if_t<std::is_move_assignable_v<_Dx2>, int> = 0>
    inline unique_ptr &operator=(unique_ptr &&_Right) noexcept
    {
        if (this != std::addressof(_Right)) {
            reset(_Right.release());
            _Mypair._Get_first() = std::move(_Right._Mypair._Get_first());
        }

        return *this;
    }

    template <class _Uty, class _Ex, class _More, class _UP_pointer = typename unique_ptr<_Uty, _Ex>::pointer,
              class _UP_element_type = typename unique_ptr<_Uty, _Ex>::element_type>
    using _Enable_conversion =
        std::enable_if_t<std::conjunction_v<std::is_array<_Uty>, std::is_same<pointer, element_type *>,
                                            std::is_same<_UP_pointer, _UP_element_type *>,
                                            std::is_convertible<_UP_element_type (*)[], element_type (*)[]>, _More>,
                         int>;

    template <class _Uty, class _Ex,
              _Enable_conversion<_Uty, _Ex,
                                 std::conditional_t<std::is_reference_v<_Dx>, std::is_same<_Ex, _Dx>,
                                                    std::is_convertible<_Ex, _Dx>>> = 0>
    inline unique_ptr(unique_ptr<_Uty, _Ex> &&_Right) noexcept
        : _Mypair(_One_then_variadic_args_t{}, std::forward<_Ex>(_Right.get_deleter()), _Right.release())
    {}

    template <class _Uty, class _Ex, _Enable_conversion<_Uty, _Ex, std::is_assignable<_Dx &, _Ex>> = 0>
    inline unique_ptr &operator=(unique_ptr<_Uty, _Ex> &&_Right) noexcept
    {
        reset(_Right.release());
        _Mypair._Get_first() = std::forward<_Ex>(_Right._Mypair._Get_first());
        return *this;
    }

    template <class _Dx2 = _Dx, _Unique_ptr_enable_default_t<_Dx2> = 0>
    constexpr unique_ptr(nullptr_t) noexcept : _Mypair(_Zero_then_variadic_args_t{})
    {}

    inline unique_ptr &operator=(nullptr_t) noexcept
    {
        reset();
        return *this;
    }

    inline void reset(nullptr_t = nullptr) noexcept { reset(pointer()); }

    inline void swap(unique_ptr &_Right) noexcept
    {
        _Swap_adl(_Mypair._Myval2, _Right._Mypair._Myval2);
        _Swap_adl(_Mypair._Get_first(), _Right._Mypair._Get_first());
    }

    inline ~unique_ptr() noexcept
    {
        if (_Mypair._Myval2) {
            _Mypair._Get_first()(_Mypair._Myval2);
        }
    }

    inline _Dx &get_deleter() noexcept { return _Mypair._Get_first(); }

    inline const _Dx &get_deleter() const noexcept { return _Mypair._Get_first(); }

    CUDA_HOST_DEVICE
    inline _Ty &operator[](size_t _Idx) const noexcept /* strengthened */ { return _Mypair._Myval2[_Idx]; }

    CUDA_HOST_DEVICE
    inline pointer get() const noexcept { return _Mypair._Myval2; }

    CUDA_HOST_DEVICE
    inline explicit operator bool() const noexcept { return static_cast<bool>(_Mypair._Myval2); }

    inline pointer release() noexcept { return std::exchange(_Mypair._Myval2, nullptr); }

    template <class _Uty, _Enable_ctor_reset<_Uty, std::false_type> = 0>
    inline void reset(_Uty _Ptr) noexcept
    {
        pointer _Old = std::exchange(_Mypair._Myval2, _Ptr);
        if (_Old) {
            _Mypair._Get_first()(_Old);
        }
    }

    unique_ptr(const unique_ptr &) = delete;
    unique_ptr &operator=(const unique_ptr &) = delete;

  private:
    template <class, class>
    friend class unique_ptr;

    _Compressed_pair<_Dx, pointer> _Mypair;
};

template <class _Ty, class... _Types, std::enable_if_t<!std::is_array_v<_Ty>, int> = 0>
inline unique_ptr<_Ty> make_unique(_Types &&..._Args)
{ // make a unique_ptr
    return unique_ptr<_Ty>(new _Ty(std::forward<_Types>(_Args)...));
}

template <class _Ty, std::enable_if_t<std::is_array_v<_Ty> && std::extent_v<_Ty> == 0, int> = 0>
inline unique_ptr<_Ty> make_unique(const size_t _Size)
{ // make a unique_ptr
    using _Elem = std::remove_extent_t<_Ty>;
    return unique_ptr<_Ty>(new _Elem[_Size]());
}

template <class _Ty, class... _Types, std::enable_if_t<std::extent_v<_Ty> != 0, int> = 0>
void make_unique(_Types &&...) = delete;

template <class _Ty, std::enable_if_t<!std::is_array_v<_Ty>, int> = 0>
inline unique_ptr<_Ty> make_unique_for_overwrite()
{
    // make a unique_ptr with default initialization
    return unique_ptr<_Ty>(new _Ty);
}

template <class _Ty, std::enable_if_t<std::is_unbounded_array_v<_Ty>, int> = 0>
inline unique_ptr<_Ty> make_unique_for_overwrite(const size_t _Size)
{
    // make a unique_ptr with default initialization
    using _Elem = std::remove_extent_t<_Ty>;
    return unique_ptr<_Ty>(new _Elem[_Size]);
}

template <class _Ty, class... _Types, std::enable_if_t<std::is_bounded_array_v<_Ty>, int> = 0>
void make_unique_for_overwrite(_Types &&...) = delete;

template <class _Ty, class _Dx, std::enable_if_t<std::is_swappable<_Dx>::value, int> = 0>
inline void swap(unique_ptr<_Ty, _Dx> &_Left, unique_ptr<_Ty, _Dx> &_Right) noexcept
{
    _Left.swap(_Right);
}

template <class _Ty1, class _Dx1, class _Ty2, class _Dx2>
CUDA_HOST_DEVICE inline bool operator==(const unique_ptr<_Ty1, _Dx1> &_Left, const unique_ptr<_Ty2, _Dx2> &_Right)
{
    return _Left.get() == _Right.get();
}

template <class _Ty1, class _Dx1, class _Ty2, class _Dx2>
    requires std::three_way_comparable_with<typename unique_ptr<_Ty1, _Dx1>::pointer,
                                            typename unique_ptr<_Ty2, _Dx2>::pointer>
CUDA_HOST_DEVICE
    std::compare_three_way_result_t<typename unique_ptr<_Ty1, _Dx1>::pointer, typename unique_ptr<_Ty2, _Dx2>::pointer>
    operator<=>(const unique_ptr<_Ty1, _Dx1> &_Left, const unique_ptr<_Ty2, _Dx2> &_Right)
{

    return _Left.get() <=> _Right.get();
}

template <class _Ty, class _Dx>
CUDA_HOST_DEVICE inline bool operator==(const unique_ptr<_Ty, _Dx> &_Left, nullptr_t) noexcept
{
    return !_Left;
}

template <class _Ty, class _Dx>
    requires std::three_way_comparable<typename unique_ptr<_Ty, _Dx>::pointer>
CUDA_HOST_DEVICE inline std::compare_three_way_result_t<typename unique_ptr<_Ty, _Dx>::pointer>
operator<=>(const unique_ptr<_Ty, _Dx> &_Left, nullptr_t)
{
    return _Left.get() <=> static_cast<typename unique_ptr<_Ty, _Dx>::pointer>(nullptr);
}

//////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
    requires(!std::is_array_v<T>)
struct CudaObjectDeleter
{
    void operator()(T *obj) const
    {
        if (!obj) {
            return;
        }
        std::destroy_at(obj);
        cuda_free_managed(obj);
    }
};

template <typename T>
    requires std::is_unbounded_array_v<T>
struct CudaObjectArrayDeleter
{
    void operator()(T obj) const
    {
        if (!obj) {
            return;
        }
        std::destroy_at(obj);
        cuda_free_managed(obj);
    }
};

template <typename T, typename... Args>
    requires(!std::is_array_v<T>)
unique_ptr<T, CudaObjectDeleter<T>> make_unique_cuda_managed(Args &&...args)
{
    void *ptr = cuda_alloc_managed(sizeof(T));
    T *obj = std::construct_at<T>(reinterpret_cast<T *>(ptr), std::forward<Args>(args)...);
    unique_ptr<T, CudaObjectDeleter<T>> p;
    p.reset(obj);
    return p;
}

template <typename T, typename... Args>
    requires std::is_unbounded_array_v<T>
void make_unique_cuda_managed(size_t n) = delete;

template <typename T, typename... Args>
    requires std::is_bounded_array_v<T>
void make_unique_cuda_managed(Args &&...args) = delete;

template <typename T>
    requires(!std::is_array_v<T>)
unique_ptr<T, CudaObjectDeleter<T>> make_unique_for_overwrite_cuda_managed()
{
    void *ptr = cuda_alloc_managed(sizeof(T));
    unique_ptr<T, CudaObjectDeleter<T>> p;
    p.reset(ptr);
    return p;
}

template <typename T>
    requires(!std::is_array_v<T>)
unique_ptr<T[], CudaObjectArrayDeleter<T[]>> make_unique_for_overwrite_cuda_managed(size_t n)
{
    void *ptr = cuda_alloc_managed(sizeof(T) * n);
    T *arr = reinterpret_cast<T *>(ptr);
    unique_ptr<T[], CudaObjectArrayDeleter<T[]>> p;
    p.reset(arr);
    return p;
}

template <typename T, typename... Args>
    requires std::is_bounded_array_v<T>
void make_unique_for_overwrite_cuda(Args &&...) = delete;

template <typename T>
    requires(!std::is_array_v<T>)
using cuda_managed_unique_ptr = unique_ptr<T, CudaObjectDeleter<T>>;
template <typename T>
    requires(!std::is_array_v<T>)
using cuda_managed_unique_array = unique_ptr<T[], CudaObjectArrayDeleter<T[]>>;

template <typename T>
    requires(!std::is_array_v<T>)
struct CudaManagedArray
{
    CudaManagedArray() = default;
    explicit CudaManagedArray(size_t size) : ptr(make_unique_for_overwrite_cuda_managed<T>(size)), size(size) {}

    CudaManagedArray(const CudaManagedArray &other)
        : ptr(make_unique_for_overwrite_cuda_managed<T>(other.size)), size(other.size)
    {
        std::copy_n(other.ptr.get(), size, ptr.get());
    }

    explicit CudaManagedArray(span<const T> s)
        : ptr(make_unique_for_overwrite_cuda_managed<T>(s.size())), size(s.size())
    {
        std::copy_n(s.data(), size, ptr.get());
    }

    CudaManagedArray &operator=(const CudaManagedArray &other)
    {
        if (this == &other) {
            return *this;
        }

        ptr = make_unique_for_overwrite_cuda_managed<T>(other.size);
        size = other.size;
        std::copy_n(other.ptr.get(), size, ptr.get());
    }

    CudaManagedArray(CudaManagedArray &&) = default;
    CudaManagedArray &operator=(CudaManagedArray &&) = default;

    CUDA_HOST_DEVICE
    T &operator[](size_t idx)
    {
        return ptr[idx];
    }
    CUDA_HOST_DEVICE
    const T &operator[](size_t idx) const
    {
        return ptr[idx];
    }
    CUDA_HOST_DEVICE
    explicit operator span<T>() { return MakeSpan(ptr.get(), size); }
    CUDA_HOST_DEVICE
    explicit operator span<const T>() const { return MakeConstSpan(ptr.get(), size); }

    unique_ptr<T[], CudaObjectArrayDeleter<T[]>> ptr;
    size_t size = 0;
};

template <typename T>
    requires std::is_trivially_destructible_v<T>
struct Uninitialized

{
    template <typename... Args>
    void construct(Args &&...args)
    {
        std::construct_at(std::launder(reinterpret_cast<T *>(&storage)), std::forward<Args>(args)...);
    }
    void destroy() { std::destroy_at(std::launder(reinterpret_cast<T *>(&storage))); }

#ifdef __NVCC__
    CUDA_HOST_DEVICE
    T &get() { return *reinterpret_cast<T *>(&storage); }

    CUDA_HOST_DEVICE
    const T &get() const { return *reinterpret_cast<const T *>(&storage); }
#else
    CUDA_HOST_DEVICE
    T &get() { return *std::launder(reinterpret_cast<T *>(&storage)); }

    CUDA_HOST_DEVICE
    const T &get() const { return *std::launder(reinterpret_cast<const T *>(&storage)); }
#endif

    std::aligned_storage_t<sizeof(T), alignof(T)> storage;
};

} // namespace ksc