#pragma once
#include "ray.cuh"
#include "vecmath.cuh"

namespace ksc
{

struct RayTriHelper
{
    CUDA_HOST_DEVICE
    explicit RayTriHelper(const ksc::Ray &ray)
    {
        ksc::vec3 absDir = abs(ray.dir);
        if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
            kz = 0;
        } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
            kz = 1;
        } else {
            kz = 2;
        }
        kx = (kz + 1) % 3;
        ky = (kx + 1) % 3;
        // Swap kx and ky dimension to preserve winding direction of triangles.
        if (ray.dir[kz] < 0.0f) {
            ksc::_swap(kx, ky);
        }
        // Calculate shear constants.
        shear = ksc::vec3(ray.dir[kx], ray.dir[ky], 1.0f);
        shear /= ray.dir[kz];
    }

    CUDA_HOST_DEVICE
    ksc::vec3 permute(const ksc::vec3 &v) const { return ksc::vec3(v[kx], v[ky], v[kz]); }

    int kx, ky, kz;
    ksc::vec3 shear;
};

struct RayTriIsect
{
    ksc::vec3 coord;
    float tHit;
};

CUDA_HOST_DEVICE
inline bool isect_ray_tri(const ksc::Ray &ray, const RayTriHelper &help, const ksc::vec3 &v0, const ksc::vec3 &v1,
                          const ksc::vec3 &v2, RayTriIsect *isect = nullptr)
{
    // Calculate vertices relative to ray origin.
    const ksc::vec3 A = help.permute(v0 - ray.origin);
    const ksc::vec3 B = help.permute(v1 - ray.origin);
    const ksc::vec3 C = help.permute(v2 - ray.origin);

    // Perform shear and scale of vertices.
    const ksc::vec3 sx = ksc::vec3(A.x, B.x, C.x) - help.shear.x * ksc::vec3(A.z, B.z, C.z);
    const ksc::vec3 sy = ksc::vec3(A.y, B.y, C.y) - help.shear.y * ksc::vec3(A.z, B.z, C.z);

    // Calculate scaled barycentric coordinates.
    ksc::vec3 bc = cross(sy, sx);

    // Fallback to test against edges using double precision.
    if (bc.x == 0.0 || bc.y == 0.0 || bc.z == 0.0) {
        bc.x = (float)((double)sx[2] * (double)sy[1] - (double)sy[2] * (double)sx[1]);
        bc.y = (float)((double)sx[0] * (double)sy[2] - (double)sy[0] * (double)sx[2]);
        bc.z = (float)((double)sx[1] * (double)sy[0] - (double)sy[1] * (double)sx[0]);
    }

    // Perform edge tests. Moving this test before and at the end of the previous conditional gives higher performance.
    if ((bc.x < 0.0 || bc.y < 0.0 || bc.z < 0.0) && (bc.x > 0.0 || bc.y > 0.0 || bc.z > 0.0)) {
        return false;
    }

    // Calculate determinant.
    float det = bc.x + bc.y + bc.z;
    if (det == 0.0) {
        return false;
    }

    // Calculate scaled z-coordinates of vertices and use them to calculate the hit distance.
    const ksc::vec3 sz = help.shear.z * ksc::vec3(A.z, B.z, C.z);
    const float T = ksc::dot(bc, sz);

    // TODO: Should the >= and <= here be > and < ?
    if (det < 0.0 && (T >= 0.0 || T <= ray.tmax * det)) {
        return false;
    } else if (det > 0.0 && (T <= 0.0 || T >= ray.tmax * det)) {
        return false;
    }

    if (isect) {
        // Normalize barycentric coordinates, and T.
        const float invDet = 1.0f / det;
        isect->coord = bc * invDet;
        isect->tHit = T * invDet;
    }
    return true;
}

} // namespace ksc