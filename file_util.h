#pragma once

#include "assertion.h"
#include "maths.h"
#include <filesystem>
#include <fstream>
#include <vector>
namespace fs = std::filesystem;

struct BinaryReader
{
    explicit BinaryReader(const fs::path &path) : stream(path, std::ios::binary) { ASSERT(stream); }

    void read(void *ptr, size_t bytes)
    {
        stream.read(reinterpret_cast<char *>(ptr), bytes);
        ASSERT(stream);
    }

    template <typename T>
    T read()
    {
        // static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "Only accept pod types.");
        T val;
        read(&val, sizeof(T));
        return val;
    }

    //#define READ_FIXED_SIZE_EIGEN(T)                                                                                       \
//    template <>                                                                                                        \
//    T read()                                                                                                           \
//    {                                                                                                                  \
//        static_assert(T::SizeAtCompileTime > 0);                                                                       \
//        T dest;                                                                                                        \
//        read_array<T::Scalar>((T::Scalar *)dest.data(), T::SizeAtCompileTime);                                         \
//        return dest;                                                                                                   \
//    }
    //
    //    READ_FIXED_SIZE_EIGEN(vec2);
    //    READ_FIXED_SIZE_EIGEN(vec3);
    //    READ_FIXED_SIZE_EIGEN(vec4);
    //    READ_FIXED_SIZE_EIGEN(vec2i);
    //    READ_FIXED_SIZE_EIGEN(vec3i);
    //    READ_FIXED_SIZE_EIGEN(color3);
    //    READ_FIXED_SIZE_EIGEN(color4);

    template <typename T>
    void read_array(T *dest, size_t size)
    {
        // static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "Only accept pod types.");
        read(dest, sizeof(T) * size);
    }

    //#define READ_ARRAY_FIXED_SIZE_EIGEN(T)                                                                                 \
//    template <>                                                                                                        \
//    void read_array(T *dest, size_t size)                                                                              \
//    {                                                                                                                  \
//        static_assert(T::SizeAtCompileTime > 0);                                                                       \
//        read_array<T::Scalar>((T::Scalar *)dest, size * T::SizeAtCompileTime);                                         \
//    }
    //
    //    READ_ARRAY_FIXED_SIZE_EIGEN(vec2);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(vec3);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(vec4);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(vec2i);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(vec3i);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(color3);
    //    READ_ARRAY_FIXED_SIZE_EIGEN(color4);

    template <typename T>
    std::vector<T> read_vector()
    {
        size_t size = read<size_t>();
        std::vector<T> vec;
        vec.resize(size);
        read_array(vec.data(), size);
        return vec;
    }

    // Error handling?
    void set_pos(size_t bytes) { stream.seekg(bytes); }

    bool eof() { return stream.peek() == EOF; }

    std::ifstream stream;
};

struct BinaryWriter
{
    explicit BinaryWriter(const fs::path &path) : stream(path, std::ios::binary) { ASSERT(stream); }

    void write(const void *ptr, size_t bytes)
    {
        stream.write(reinterpret_cast<const char *>(ptr), bytes);
        ASSERT(stream);
    }

    template <typename T>
    void write(const T &src)
    {
        // static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "Only accept pod types.");
        write(&src, sizeof(T));
    }

    //#define WRITE_FIXED_SIZE_EIGEN(T)                                                                                      \
//    template <>                                                                                                        \
//    void write(const T &src)                                                                                           \
//    {                                                                                                                  \
//        static_assert(T::SizeAtCompileTime > 0);                                                                       \
//        write_array<T::Scalar>((const T::Scalar *)src.data(), T::SizeAtCompileTime);                                   \
//    }
    //
    //    WRITE_FIXED_SIZE_EIGEN(vec2);
    //    WRITE_FIXED_SIZE_EIGEN(vec3);
    //    WRITE_FIXED_SIZE_EIGEN(vec4);
    //    WRITE_FIXED_SIZE_EIGEN(vec2i);
    //    WRITE_FIXED_SIZE_EIGEN(vec3i);
    //    WRITE_FIXED_SIZE_EIGEN(color3);
    //    WRITE_FIXED_SIZE_EIGEN(color4);

    template <typename T>
    void write_array(const T *src, size_t size)
    {
        // static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>, "Only accept pod types.");
        write(src, sizeof(T) * size);
    }

    //#define WRITE_ARRAY_FIXED_SIZE_EIGEN(T)                                                                                \
//    template <>                                                                                                        \
//    void write_array(const T *src, size_t size)                                                                        \
//    {                                                                                                                  \
//        static_assert(T::SizeAtCompileTime > 0);                                                                       \
//        write_array<T::Scalar>((const T::Scalar *)src, size * T::SizeAtCompileTime);                                   \
//    }
    //
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(vec2);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(vec3);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(vec4);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(vec2i);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(vec3i);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(color3);
    //    WRITE_ARRAY_FIXED_SIZE_EIGEN(color4);

    template <typename T>
    void write_vector(const std::vector<T> &vec)
    {
        write(vec.size());
        write_array(vec.data(), vec.size());
    }

    std::ofstream stream;
};

// Poor man's std::format...
template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}