#pragma once

#include "assertion.h"
#include "maths.h"
#include "ray.h"

namespace ks
{

template<typename T>
struct AABB2_t
{
    AABB2_t() = default;
    AABB2_t(const Eigen::Vector2<T> &min, const Eigen::Vector2<T> &max) : min(min), max(max) {}

    void expand(const Eigen::Vector2<T> &point)
    {
        min.x() = std::min(min.x(), point.x());
        min.y() = std::min(min.y(), point.y());
        max.x() = std::max(max.x(), point.x());
        max.y() = std::max(max.y(), point.y());
    }

    void expand(const AABB2_t &aabb)
    {
        min.x() = std::min(min.x(), aabb.min.x());
        min.y() = std::min(min.y(), aabb.min.y());

        max.x() = std::max(max.x(), aabb.max.x());
        max.y() = std::max(max.y(), aabb.max.y());
    }

    bool contain(const Eigen::Vector2<T> &point) const
    {
        return point.x() >= min.x() && point.x() <= max.x() && point.y() >= min.y() && point.y() <= max.y();
    }

    bool contain(const AABB2_t &bound) const
    {
        return bound.min.x() >= min.x() && bound.max.x() <= max.x() && bound.min.y() >= min.y() &&
               bound.max.y() <= max.y();
    }

    bool isEmpty() const { return min.x() > max.x() || min.y() > max.y(); }

    Eigen::Vector2<T> center() const { return T(0.5) * (min + max); }

    Eigen::Vector2<T> extents() const
    {
        if (isEmpty()) {
            return Eigen::Vector2<T>::Zero();
        }
        return max - min;
    }

    float offset(const Eigen::Vector2<T> &point, uint32_t dim) const
    {
        float ext = extents()[dim];
        if (ext == 0.0) {
            return 0.0;
        }
        return (point[dim] - min[dim]) / ext;
    }

    Eigen::Vector2<T> offset(const Eigen::Vector2<T> &point) const
    {
        Eigen::Vector2<T> ext = extents();
        Eigen::Vector2<T> o = (point - min).cwiseQuotient(ext);
        for (int i = 0; i < 2; ++i) {
            if (ext[i] == 0.0f)
                o[i] = 0.0f;
        }
        return o;
    }

    Eigen::Vector2<T> lerp(const Eigen::Vector2<T> &t) const
    {
        return Eigen::Vector2<T>(std::lerp(min.x(), max.x(), t.x()), std::lerp(min.y(), max.y(), t.y()));
    }

    uint32_t largestAxis() const
    {
        Eigen::Vector2<T> exts = extents();
        if (exts.x() >= exts.y())
            return 0;
        else
            return 1;
    }

    float area() const
    {
        Eigen::Vector2<T> exts = extents();
        return exts.x() * exts.y();
    }

    const Eigen::Vector2<T> &operator[](uint32_t index) const
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<const Eigen::Vector2<T> *>(this))[index];
    }

    Eigen::Vector2<T> &operator[](uint32_t index)
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<Eigen::Vector2<T> *>(this))[index];
    }

    Eigen::Vector2<T> corner(int i) const
    {
        const Eigen::Vector2<T> *c = (const Eigen::Vector2<T> *)this;
        return Eigen::Vector2<T>(c[i & 1].x(), c[(i >> 1) & 1].y());
    }

    Eigen::Vector2<T> min = Eigen::Vector2<T>::Constant(inf);
    Eigen::Vector2<T> max = Eigen::Vector2<T>::Constant(-inf);
};

template<typename T>
inline bool intersectBool(const AABB2_t<T> &b0, const AABB2_t<T> &b1)
{
    return !(b0.min.x() > b1.max.x() || b1.min.x() > b0.max.x() || b0.min.y() > b1.max.y() || b1.min.y() > b0.max.y());
}

template<typename T>
inline AABB2_t<T> intersect(const AABB2_t<T> &b0, const AABB2_t<T> &b1)
{
    AABB2_t<T> ret;
    ret.min = b0.min.cwiseMax(b1.min);
    ret.max = b0.max.cwiseMin(b1.max);
    if (ret.min.x() > ret.max.x() || ret.min.y() > ret.max.y()) {
        return AABB2_t<T>();
    }
    return ret;
}

template<typename T>
inline AABB2_t<T> join(const AABB2_t<T> &b0, const AABB2_t<T> &b1)
{
    AABB2_t<T> ret;
    ret.min = b0.min.cwiseMin(b1.min);
    ret.max = b0.max.cwiseMax(b1.max);
    return ret;
}

using AABB2 = AABB2_t<float>;
using AABB2d = AABB2_t<double>;

template <typename T>
struct AABB3_t
{
    AABB3_t() = default;
    AABB3_t(const Eigen::Vector3<T> &min, const Eigen::Vector3<T> &max) : min(min), max(max) {}

    void expand(const Eigen::Vector3<T> &point)
    {
        min.x() = std::min(min.x(), point.x());
        min.y() = std::min(min.y(), point.y());
        min.z() = std::min(min.z(), point.z());

        max.x() = std::max(max.x(), point.x());
        max.y() = std::max(max.y(), point.y());
        max.z() = std::max(max.z(), point.z());
    }

    void expand(const AABB3_t &aabb)
    {
        min.x() = std::min(min.x(), aabb.min.x());
        min.y() = std::min(min.y(), aabb.min.y());
        min.z() = std::min(min.z(), aabb.min.z());

        max.x() = std::max(max.x(), aabb.max.x());
        max.y() = std::max(max.y(), aabb.max.y());
        max.z() = std::max(max.z(), aabb.max.z());
    }

    bool isEmpty() const { return min.x() > max.x() || min.y() > max.y() || min.z() > max.z(); }

    Eigen::Vector3<T> center() const { return 0.5f * (min + max); }

    Eigen::Vector3<T> extents() const { return max - min; }

    float offset(const Eigen::Vector3<T> &point, uint32_t dim) const
    {
        float ext = extents()[dim];
        if (ext == 0.0f) {
            return 0.0f;
        }
        return (point[dim] - min[dim]) / ext;
    }

    Eigen::Vector3<T> offset(const Eigen::Vector3<T> &point) const
    {
        Eigen::Vector3<T> ext = extents();
        Eigen::Vector3<T> o = (point - min).cwiseQuotient(ext);
        for (int i = 0; i < 3; ++i) {
            if (ext[i] == 0.0f)
                o[i] = 0.0f;
        }
        return o;
    }

    Eigen::Vector3<T> corner(int i) const
    {
        const Eigen::Vector3<T> *c = (const Eigen::Vector3<T> *)this;
        return Eigen::Vector3<T>(c[i & 1].x(), c[(i & 2) >> 1].y(), c[(i & 4) >> 2].z());
    }

    Eigen::Vector3<T> lerp(const Eigen::Vector3<T> &t) const
    {
        return Eigen::Vector3<T>(std::lerp(min.x(), max.x(), t.x()), std::lerp(min.y(), max.y(), t.y()),
                    std::lerp(min.z(), max.z(), t.z()));
    }

    uint32_t largestAxis() const
    {
        Eigen::Vector3<T> exts = extents();
        if (exts.x() >= exts.y() && exts.x() >= exts.z())
            return 0;
        else if (exts.y() >= exts.x() && exts.y() >= exts.z())
            return 1;
        else
            return 2;
    }

    float surfaceArea() const
    {
        Eigen::Vector3<T> exts = extents();
        return 2.0f * (exts.x() * exts.y() + exts.y() * exts.z() + exts.x() * exts.z());
    }

    float volume() const
    {
        Eigen::Vector3<T> exts = extents();
        return exts.x() * exts.y() * exts.z();
    }

    bool contain(const Eigen::Vector3<T> &p) const
    {
        return min.x() <= p.x() && min.y() <= p.y() && min.z() <= p.z() && max.x() >= p.x() && max.y() >= p.y() &&
               max.z() >= p.z();
    }

    bool contain(const AABB3_t &other) const
    {
        return min.x() <= other.min.x() && min.y() <= other.min.y() && min.z() <= other.min.z() &&
               max.x() >= other.max.x() && max.y() >= other.max.y() && max.z() >= other.max.z();
    }

    const Eigen::Vector3<T> &operator[](uint32_t index) const
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<const Eigen::Vector3<T> *>(this))[index];
    }

    Eigen::Vector3<T> &operator[](uint32_t index)
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<Eigen::Vector3<T> *>(this))[index];
    }

    Eigen::Vector3<T> min = Eigen::Vector3<T>::Constant(inf);
    Eigen::Vector3<T> max = Eigen::Vector3<T>::Constant(-inf);
};

template<typename T>
inline bool intersectBool(const AABB3_t<T> &b0, const AABB3_t<T> &b1)
{
    return !(b0.min.x() > b1.max.x() || b1.min.x() > b0.max.x() || b0.min.y() > b1.max.y() || b1.min.y() > b0.max.y() ||
             b0.min.x() > b1.max.z() || b1.min.z() > b0.max.z());
}

template<typename T>
inline AABB3_t<T> intersect(const AABB3_t<T> &b0, const AABB3_t<T> &b1)
{
    AABB3_t<T> ret;
    ret.min = b0.min.cwiseMax(b1.min);
    ret.max = b0.max.cwiseMin(b1.max);
    if (ret.min.x() > ret.max.x() || ret.min.y() > ret.max.y() || ret.min.z() > ret.max.z()) {
        return AABB3_t<T>();
    }
    return ret;
}

template<typename T>
inline AABB3_t<T> join(const AABB3_t<T> &b0, const AABB3_t<T> &b1) { return AABB3_t<T>(b0.min.cwiseMin(b1.min), b0.max.cwiseMax(b1.max)); }

using AABB3 = AABB3_t<float>;
using AABB3d = AABB3_t<double>;

inline bool isect_ray_aabb(const Ray &ray, const AABB3 &bounds, const vec3 &inv_dir, const int dir_is_neg[3],
                           float thit[2] = nullptr, vec3 phit[2] = nullptr)
{
    // Check for ray intersection against $x$ and $y$ slabs
    // Need to avoid 0 multiplied by inf here...
    float tmin = (bounds[dir_is_neg[0]].x() - ray.origin.x());
    if (tmin != 0.0f)
        tmin *= inv_dir.x();
    float tmax = (bounds[1 - dir_is_neg[0]].x() - ray.origin.x());
    if (tmax != 0.0f)
        tmax *= inv_dir.x();
    float tymin = (bounds[dir_is_neg[1]].y() - ray.origin.y());
    if (tymin != 0.0f)
        tymin *= inv_dir.y();
    float tymax = (bounds[1 - dir_is_neg[1]].y() - ray.origin.y());
    if (tymax != 0.0f)
        tymax *= inv_dir.y();

    int min_axis = dir_is_neg[0] * 3;
    int max_axis = (1 - dir_is_neg[0]) * 3;

    if (tmin > tymax || tymin > tmax)
        return false;
    if (tymin > tmin) {
        tmin = tymin;
        min_axis = dir_is_neg[1] * 3 + 1;
    }
    if (tymax < tmax) {
        tmax = tymax;
        max_axis = (1 - dir_is_neg[1]) * 3 + 1;
    }

    // Check for ray intersection against $z$ slab
    float tzmin = (bounds[dir_is_neg[2]].z() - ray.origin.z());
    if (tzmin != 0.0f)
        tzmin *= inv_dir.z();
    float tzmax = (bounds[1 - dir_is_neg[2]].z() - ray.origin.z());
    if (tzmax != 0.0f)
        tzmax *= inv_dir.z();

    // Update _tzMax_ to ensure robust bounds intersection
    if (tmin > tzmax || tzmin > tmax)
        return false;
    if (tzmin > tmin) {
        tmin = tzmin;
        min_axis = dir_is_neg[2] * 3 + 2;
    }
    if (tzmax < tmax) {
        tmax = tzmax;
        max_axis = (1 - dir_is_neg[2]) * 3 + 2;
    }

    if ((tmin < ray.tmax) && (tmax > ray.tmin)) {
        if (ray.tmin > tmin) {
            tmin = ray.tmin;
            min_axis = -1;
        }
        if (ray.tmax < tmax) {
            tmax = ray.tmax;
            max_axis = -1;
        }
        if (thit) {
            thit[0] = tmin;
            thit[1] = tmax;
        }
        if (phit) {
            phit[0] = ray(tmin);
            if (min_axis >= 0)
                phit[0][min_axis % 3] = ((float *)&bounds)[min_axis];
            phit[1] = ray(tmax);
            if (max_axis >= 0)
                phit[1][max_axis % 3] = ((float *)&bounds)[max_axis];
        }
        return true;
    }
    return false;
}

inline bool isect_ray_aabb(const Ray &ray, const AABB3 &bounds, float thit[2] = nullptr, vec3 phit[2] = nullptr)
{
    vec3 invDir = ray.dir.cwiseInverse();
    int dirIsNeg[3];
    for (int i = 0; i < 3; ++i) {
        // Actually need to handle negative zeros?
        dirIsNeg[i] = ray.dir[i] < 0.0f || (ray.dir[i] == 0.0f && std::signbit(ray.dir[i]));
    }
    return isect_ray_aabb(ray, bounds, invDir, dirIsNeg, thit, phit);
}

inline AABB3 transform_aabb(const Transform &t, const AABB3 &b)
{
    const vec3 *c = (const vec3 *)&b;
    AABB3 out;
    for (int i = 0; i < 8; ++i) {
        vec3 corner(c[i & 1].x(), c[(i & 2) >> 1].y(), c[(i & 4) >> 2].z());
        out.expand(t.point(corner));
    }
    return out;
}

} // namespace ks
