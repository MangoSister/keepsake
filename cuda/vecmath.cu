#include "vecmath.cuh"

namespace ksc
{
vec3 equal_area_square_to_sphere(vec2 p)
{
    KSC_ASSERT(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
    // Transform _p_ to $[-1,1]^2$ and compute absolute values
    float u = 2 * p.x - 1, v = 2 * p.y - 1;
    float up = abs(u), vp = abs(v);

    // Compute radius _r_ as signed distance from diagonal
    float signedDistance = 1 - (up + vp);
    float d = abs(signedDistance);
    float r = 1 - d;

    // Compute angle $\phi$ for square to sphere mapping
    float phi = (r == 0 ? 1 : (vp - up) / r + 1) * pi / 4;

    // Find $z$ coordinate for spherical direction
    float z = copysign(1 - sqr(r), signedDistance);

    // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
    float cosPhi = copysign(cos(phi), u);
    float sinPhi = copysign(sin(phi), v);
    return vec3(cosPhi * r * safe_sqrt(2 - sqr(r)), sinPhi * r * safe_sqrt(2 - sqr(r)), z);
}

vec2 equal_area_sphere_to_square(vec3 d)
{
    KSC_ASSERT(length_squared(d) > .999 && length_squared(d) < 1.001);
    float x = abs(d.x), y = abs(d.y), z = abs(d.z);

    // Compute the radius r
    float r = safe_sqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    float a = fmaxf(x, y), b = fminf(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const float t1 = 0.406758566246788489601959989e-5;
    const float t2 = 0.636226545274016134946890922156;
    const float t3 = 0.61572017898280213493197203466e-2;
    const float t4 = -0.247333733281268944196501420480;
    const float t5 = 0.881770664775316294736387951347e-1;
    const float t6 = 0.419038818029165735901852432784e-1;
    const float t7 = -0.251390972343483509333252996350e-1;
    float phi = evaluate_polynomial(b, t1, t2, t3, t4, t5, t6, t7);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    float v = phi * r;
    float u = r - v;

    if (d.z < 0) {
        // southern hemisphere -> mirror u,v
        ksc::swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = copysign(u, d.x);
    v = copysign(v, d.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return vec2(0.5f * (u + 1), 0.5f * (v + 1));
}

vec2 wrap_equal_area_square(vec2 uv)
{
    if (uv[0] < 0) {
        uv[0] = -uv[0];    // mirror across u = 0
        uv[1] = 1 - uv[1]; // mirror across v = 0.5
    } else if (uv[0] > 1) {
        uv[0] = 2 - uv[0]; // mirror across u = 1
        uv[1] = 1 - uv[1]; // mirror across v = 0.5
    }
    if (uv[1] < 0) {
        uv[0] = 1 - uv[0]; // mirror across u = 0.5
        uv[1] = -uv[1];    // mirror across v = 0;
    } else if (uv[1] > 1) {
        uv[0] = 1 - uv[0]; // mirror across u = 0.5
        uv[1] = 2 - uv[1]; // mirror across v = 1
    }
    return uv;
}

void equal_area_bilerp(vec2 uv, int N, array<vec2i, 4> &idx, array<float, 4> &weight)
{
    // ASSERT(is_pow2(N));
    float x = uv.x * N - 0.5f;
    float y = uv.y * N - 0.5f;
    int x0 = ::floor(x);
    int y0 = ::floor(y);
    float tx = x - x0;
    float ty = y - y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = clamp(x0, 0, N - 1);
    y0 = clamp(y0, 0, N - 1);
    x1 = clamp(x1, 0, N - 1);
    y1 = clamp(y1, 0, N - 1);

    float inv_tx = 1.0f - tx;
    float inv_ty = 1.0f - ty;
    weight[0] = inv_tx * inv_ty;
    weight[1] = tx * inv_ty;
    weight[2] = inv_tx * ty;
    weight[3] = tx * ty;

    // Need to handle wrap around on the boundary.

    bool m00 = (x0 ^ y0) & N;
    bool m01 = (x1 ^ y0) & N;
    bool m10 = (x0 ^ y1) & N;
    bool m11 = (x1 ^ y1) & N;

    int N_1 = N - 1;
    x0 = x0 & N_1;
    y0 = y0 & N_1;
    x1 = x1 & N_1;
    y1 = y1 & N_1;

    if (m00)
        idx[0] = vec2i(N_1 - x0, N_1 - y0);
    else
        idx[0] = vec2i(x0, y0);

    if (m01)
        idx[1] = vec2i(N_1 - x1, N_1 - y0);
    else
        idx[1] = vec2i(x1, y0);

    if (m10)
        idx[2] = vec2i(N_1 - x0, N_1 - y1);
    else
        idx[2] = vec2i(x0, y1);

    if (m11)
        idx[3] = vec2i(N_1 - x1, N_1 - y1);
    else
        idx[3] = vec2i(x1, y1);
}

} // namespace ksc