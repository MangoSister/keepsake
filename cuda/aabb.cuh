#pragma once

#include "ray.cuh"
#include "transform.cuh"
#include "vecmath.cuh"

namespace ksc
{

struct AABB3
{
    AABB3() = default;
    CUDA_HOST_DEVICE
    AABB3(const vec3 &min, const vec3 &max) : min(min), max(max) {}

    CUDA_HOST_DEVICE
    void expand(const vec3 &point)
    {
        min = ksc::min(min, point);
        max = ksc::max(max, point);
    }

    CUDA_HOST_DEVICE
    void expand(const AABB3 &aabb)
    {
        min = ksc::min(min, aabb.min);
        max = ksc::max(max, aabb.max);
    }

    CUDA_HOST_DEVICE
    bool is_empty() const { return min.x > max.x || min.y > max.y || min.z > max.z; }

    CUDA_HOST_DEVICE
    vec3 center() const { return 0.5f * (min + max); }

    CUDA_HOST_DEVICE
    vec3 extents() const { return max - min; }

    CUDA_HOST_DEVICE
    float offset(const vec3 &point, uint32_t dim) const
    {
        float ext = extents()[dim];
        if (ext == 0.0f) {
            return 0.0f;
        }
        return (point[dim] - min[dim]) / ext;
    }

    CUDA_HOST_DEVICE
    vec3 offset(const vec3 &point) const
    {
        vec3 ext = extents();
        vec3 o = (point - min) / ext;
        for (int i = 0; i < 3; ++i) {
            if (ext[i] == 0.0f)
                o[i] = 0.0f;
        }
        return o;
    }

    CUDA_HOST_DEVICE
    vec3 corner(int i) const
    {
        const vec3 *c = (const vec3 *)this;
        return vec3(c[i & 1].x, c[(i & 2) >> 1].y, c[(i & 4) >> 2].z);
    }

    CUDA_HOST_DEVICE
    vec3 lerp(const vec3 &t) const { return ksc::lerp(t, min, max); }

    CUDA_HOST_DEVICE
    uint32_t largest_axis() const
    {
        vec3 exts = extents();
        if (exts.x >= exts.y && exts.x >= exts.z)
            return 0;
        else if (exts.y >= exts.x && exts.y >= exts.z)
            return 1;
        else
            return 2;
    }

    CUDA_HOST_DEVICE
    float surface_area() const
    {
        vec3 exts = extents();
        return 2.0f * (exts.x * exts.y + exts.y * exts.z + exts.x * exts.z);
    }

    CUDA_HOST_DEVICE
    float volume() const
    {
        vec3 exts = extents();
        return exts.x * exts.y * exts.z;
    }

    CUDA_HOST_DEVICE
    bool contain(const vec3 &p) const
    {
        return min.x <= p.x && min.y <= p.y && min.z <= p.z && max.x >= p.x && max.y >= p.y && max.z >= p.z;
    }

    CUDA_HOST_DEVICE
    bool contain(const AABB3 &other) const
    {
        return min.x <= other.min.x && min.y <= other.min.y && min.z <= other.min.z && max.x >= other.max.x &&
               max.y >= other.max.y && max.z >= other.max.z;
    }

    CUDA_HOST_DEVICE
    const vec3 &operator[](uint32_t index) const
    {
        KSC_ASSERT(index <= 1);
        return (reinterpret_cast<const vec3 *>(this))[index];
    }

    CUDA_HOST_DEVICE
    vec3 &operator[](uint32_t index)
    {
        KSC_ASSERT(index <= 1);
        return (reinterpret_cast<vec3 *>(this))[index];
    }

    vec3 min = vec3(inf);
    vec3 max = vec3(-inf);
};

CUDA_HOST_DEVICE inline AABB3 transform_aabb(const Transform &t, const AABB3 &b)
{
    const vec3 *c = (const vec3 *)&b;
    AABB3 out;
    for (int i = 0; i < 8; ++i) {
        vec3 corner(c[i & 1].x, c[(i & 2) >> 1].y, c[(i & 4) >> 2].z);
        out.expand(t.point(corner));
    }
    return out;
}

CUDA_HOST_DEVICE
inline bool isect_ray_aabb(const Ray &ray, const AABB3 &bounds, const vec3 &inv_dir, const int dir_is_neg[3],
                           float thit[2] = nullptr, vec3 phit[2] = nullptr)
{
    // Check for ray intersection against $x$ and $y$ slabs
    // Need to avoid 0 multiplied by inf here...
    float tmin = (bounds[dir_is_neg[0]].x - ray.origin.x);
    if (tmin != 0.0f)
        tmin *= inv_dir.x;
    float tmax = (bounds[1 - dir_is_neg[0]].x - ray.origin.x);
    if (tmax != 0.0f)
        tmax *= inv_dir.x;
    float tymin = (bounds[dir_is_neg[1]].y - ray.origin.y);
    if (tymin != 0.0f)
        tymin *= inv_dir.y;
    float tymax = (bounds[1 - dir_is_neg[1]].y - ray.origin.y);
    if (tymax != 0.0f)
        tymax *= inv_dir.y;

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
    float tzmin = (bounds[dir_is_neg[2]].z - ray.origin.z);
    if (tzmin != 0.0f)
        tzmin *= inv_dir.z;
    float tzmax = (bounds[1 - dir_is_neg[2]].z - ray.origin.z);
    if (tzmax != 0.0f)
        tzmax *= inv_dir.z;

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

CUDA_HOST_DEVICE
inline bool isect_ray_aabb(const Ray &ray, const AABB3 &bounds, float thit[2] = nullptr, vec3 phit[2] = nullptr)
{
    vec3 inv_dir = 1.0f / ray.dir;
    int dir_is_neg[3];
    for (int i = 0; i < 3; ++i) {
        // Actually need to handle negative zeros?
        dir_is_neg[i] = ray.dir[i] < 0.0f || (ray.dir[i] == 0.0f && signbit(ray.dir[i]));
    }
    return isect_ray_aabb(ray, bounds, inv_dir, dir_is_neg, thit, phit);
}

} // namespace ksc