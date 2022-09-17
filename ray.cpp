#include "ray.h"

void Intersection::compute_uv_partials(const Ray &ray)
{
    if (ray.has_ray_diffs()) {
        float d = frame.n.dot(p);
        float tx = -(frame.n.dot(ray.rx_origin) - d) / frame.n.dot(ray.rx_dir);
        if (std::isinf(tx) || std::isnan(tx))
            goto fail;
        vec3 px = ray.rx_origin + tx * ray.rx_dir;
        float ty = -(frame.n.dot(vec3(ray.ry_origin)) - d) / frame.n.dot(ray.ry_dir);
        if (std::isinf(ty) || std::isnan(ty))
            goto fail;
        vec3 py = ray.ry_origin + ty * ray.ry_dir;
        dpdx = px - p;
        dpdy = py - p;

        int dim[2];
        if (std::abs(frame.n.x()) > std::abs(frame.n.y()) && std::abs(frame.n.x()) > std::abs(frame.n.z())) {
            dim[0] = 1;
            dim[1] = 2;
        } else if (std::abs(frame.n.y()) > std::abs(frame.n.z())) {
            dim[0] = 0;
            dim[1] = 2;
        } else {
            dim[0] = 0;
            dim[1] = 1;
        }

        float A[2][2] = {{dpdu[dim[0]], dpdv[dim[0]]}, {dpdu[dim[1]], dpdv[dim[1]]}};
        float Bx[2] = {px[dim[0]] - p[dim[0]], px[dim[1]] - p[dim[1]]};
        float By[2] = {py[dim[0]] - p[dim[0]], py[dim[1]] - p[dim[1]]};
        if (!solve_linear_system_2x2(A, Bx, dudx, dvdx))
            dudx = dvdx = 0;
        if (!solve_linear_system_2x2(A, By, dudy, dvdy))
            dudy = dvdy = 0;
    } else {
    fail:
        dudx = dvdx = 0;
        dudy = dvdy = 0;
        dpdx = dpdy = vec3(0, 0, 0);
    }
}
