#pragma once

#include "assertion.h"
#include "maths.h"
#include "ray.h"

KS_NAMESPACE_BEGIN

struct AABB2
{
    AABB2() = default;
    AABB2(const vec2 &min, const vec2 &max) : min(min), max(max) {}

    void expand(const vec2 &point)
    {
        min.x() = std::min(min.x(), point.x());
        min.y() = std::min(min.y(), point.y());
        max.x() = std::max(max.x(), point.x());
        max.y() = std::max(max.y(), point.y());
    }

    void expand(const AABB2 &aabb)
    {
        min.x() = std::min(min.x(), aabb.min.x());
        min.y() = std::min(min.y(), aabb.min.y());

        max.x() = std::max(max.x(), aabb.max.x());
        max.y() = std::max(max.y(), aabb.max.y());
    }

    bool contain(const vec2 &point) const
    {
        return point.x() >= min.x() && point.x() <= max.x() && point.y() >= min.y() && point.y() <= max.y();
    }

    bool contain(const AABB2 &bound) const
    {
        return bound.min.x() >= min.x() && bound.max.x() <= max.x() && bound.min.y() >= min.y() &&
               bound.max.y() <= max.y();
    }

    bool isEmpty() const { return min.x() > max.x() || min.y() > max.y(); }

    vec2 center() const { return 0.5f * (min + max); }

    vec2 extents() const
    {
        if (isEmpty()) {
            return vec2::Zero();
        }
        return max - min;
    }

    float offset(const vec2 &point, uint32_t dim) const
    {
        float ext = extents()[dim];
        if (ext == 0.0) {
            return 0.0;
        }
        return (point[dim] - min[dim]) / ext;
    }

    vec2 offset(const vec2 &point) const
    {
        vec2 ext = extents();
        vec2 o = (point - min).cwiseQuotient(ext);
        for (int i = 0; i < 2; ++i) {
            if (ext[i] == 0.0f)
                o[i] = 0.0f;
        }
        return o;
    }

    vec2 lerp(const vec2 &t) const
    {
        return vec2(std::lerp(min.x(), max.x(), t.x()), std::lerp(min.y(), max.y(), t.y()));
    }

    uint32_t largestAxis() const
    {
        vec2 exts = extents();
        if (exts.x() >= exts.y())
            return 0;
        else
            return 1;
    }

    float area() const
    {
        vec2 exts = extents();
        return exts.x() * exts.y();
    }

    const vec2 &operator[](uint32_t index) const
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<const vec2 *>(this))[index];
    }

    vec2 &operator[](uint32_t index)
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<vec2 *>(this))[index];
    }

    vec2 corner(int i) const
    {
        const vec2 *c = (const vec2 *)this;
        return vec2(c[i & 1].x(), c[(i >> 1) & 1].y());
    }

    vec2 min = vec2::Constant(inf);
    vec2 max = vec2::Constant(-inf);
};

inline bool intersectBool(const AABB2 &b0, const AABB2 &b1)
{
    return !(b0.min.x() > b1.max.x() || b1.min.x() > b0.max.x() || b0.min.y() > b1.max.y() || b1.min.y() > b0.max.y());
}

inline AABB2 intersect(const AABB2 &b0, const AABB2 &b1)
{
    AABB2 ret;
    ret.min = b0.min.cwiseMax(b1.min);
    ret.max = b0.max.cwiseMin(b1.max);
    if (ret.min.x() > ret.max.x() || ret.min.y() > ret.max.y()) {
        return AABB2();
    }
    return ret;
}

inline AABB2 join(const AABB2 &b0, const AABB2 &b1)
{
    AABB2 ret;
    ret.min = b0.min.cwiseMin(b1.min);
    ret.max = b0.max.cwiseMax(b1.max);
    return ret;
}

struct AABB3
{
    AABB3() = default;
    AABB3(const vec3 &min, const vec3 &max) : min(min), max(max) {}

    void expand(const vec3 &point)
    {
        min.x() = std::min(min.x(), point.x());
        min.y() = std::min(min.y(), point.y());
        min.z() = std::min(min.z(), point.z());

        max.x() = std::max(max.x(), point.x());
        max.y() = std::max(max.y(), point.y());
        max.z() = std::max(max.z(), point.z());
    }

    void expand(const AABB3 &aabb)
    {
        min.x() = std::min(min.x(), aabb.min.x());
        min.y() = std::min(min.y(), aabb.min.y());
        min.z() = std::min(min.z(), aabb.min.z());

        max.x() = std::max(max.x(), aabb.max.x());
        max.y() = std::max(max.y(), aabb.max.y());
        max.z() = std::max(max.z(), aabb.max.z());
    }

    bool isEmpty() const { return min.x() > max.x() || min.y() > max.y() || min.z() > max.z(); }

    vec3 center() const { return 0.5f * (min + max); }

    vec3 extents() const { return max - min; }

    float offset(const vec3 &point, uint32_t dim) const
    {
        float ext = extents()[dim];
        if (ext == 0.0f) {
            return 0.0f;
        }
        return (point[dim] - min[dim]) / ext;
    }

    vec3 offset(const vec3 &point) const
    {
        vec3 ext = extents();
        vec3 o = (point - min).cwiseQuotient(ext);
        for (int i = 0; i < 3; ++i) {
            if (ext[i] == 0.0f)
                o[i] = 0.0f;
        }
        return o;
    }

    vec3 corner(int i) const
    {
        const vec3 *c = (const vec3 *)this;
        return vec3(c[i & 1].x(), c[(i & 2) >> 1].y(), c[(i & 4) >> 2].z());
    }

    vec3 lerp(const vec3 &t) const
    {
        return vec3(std::lerp(min.x(), max.x(), t.x()), std::lerp(min.y(), max.y(), t.y()),
                    std::lerp(min.z(), max.z(), t.z()));
    }

    uint32_t largestAxis() const
    {
        vec3 exts = extents();
        if (exts.x() >= exts.y() && exts.x() >= exts.z())
            return 0;
        else if (exts.y() >= exts.x() && exts.y() >= exts.z())
            return 1;
        else
            return 2;
    }

    float surfaceArea() const
    {
        vec3 exts = extents();
        return 2.0f * (exts.x() * exts.y() + exts.y() * exts.z() + exts.x() * exts.z());
    }

    float volume() const
    {
        vec3 exts = extents();
        return exts.x() * exts.y() * exts.z();
    }

    bool contain(const vec3 &p) const
    {
        return min.x() <= p.x() && min.y() <= p.y() && min.z() <= p.z() && max.x() >= p.x() && max.y() >= p.y() &&
               max.z() >= p.z();
    }

    bool contain(const AABB3 &other) const
    {
        return min.x() <= other.min.x() && min.y() <= other.min.y() && min.z() <= other.min.z() &&
               max.x() >= other.max.x() && max.y() >= other.max.y() && max.z() >= other.max.z();
    }

    const vec3 &operator[](uint32_t index) const
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<const vec3 *>(this))[index];
    }

    vec3 &operator[](uint32_t index)
    {
        ASSERT(index <= 1);
        return (reinterpret_cast<vec3 *>(this))[index];
    }

    vec3 min = vec3::Constant(inf);
    vec3 max = vec3::Constant(-inf);
};

inline bool intersectBool(const AABB3 &b0, const AABB3 &b1)
{
    return !(b0.min.x() > b1.max.x() || b1.min.x() > b0.max.x() || b0.min.y() > b1.max.y() || b1.min.y() > b0.max.y() ||
             b0.min.x() > b1.max.z() || b1.min.z() > b0.max.z());
}

inline AABB3 intersect(const AABB3 &b0, const AABB3 &b1)
{
    AABB3 ret;
    ret.min = b0.min.cwiseMax(b1.min);
    ret.max = b0.max.cwiseMin(b1.max);
    if (ret.min.x() > ret.max.x() || ret.min.y() > ret.max.y() || ret.min.z() > ret.max.z()) {
        return AABB3();
    }
    return ret;
}

inline AABB3 join(const AABB3 &b0, const AABB3 &b1) { return AABB3(b0.min.cwiseMin(b1.min), b0.max.cwiseMax(b1.max)); }

inline bool isect_ray_aabb(const Ray &ray, const AABB3 &bounds, const vec3 &inv_dir, const int dir_is_neg[3],
                           float thit[2] = nullptr, vec3 phit[2] = nullptr)
{
    // Check for ray intersection against $x$ and $y$ slabs
    float tmin = (bounds[dir_is_neg[0]].x() - ray.origin.x()) * inv_dir.x();
    float tmax = (bounds[1 - dir_is_neg[0]].x() - ray.origin.x()) * inv_dir.x();
    float tymin = (bounds[dir_is_neg[1]].y() - ray.origin.y()) * inv_dir.y();
    float tymax = (bounds[1 - dir_is_neg[1]].y() - ray.origin.y()) * inv_dir.y();

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
    float tzmin = (bounds[dir_is_neg[2]].z() - ray.origin.z()) * inv_dir.z();
    float tzmax = (bounds[1 - dir_is_neg[2]].z() - ray.origin.z()) * inv_dir.z();

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

KS_NAMESPACE_END