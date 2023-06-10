#pragma once
#include "maths.h"

// https://fileadmin.cs.lth.se/cs/Personal/Petrik_Clarberg/

/*
 * Contains implementations of the forward and inverse transform using
 * scalar code and approximations of the trigonometric operations.
 * Branching is avoided as far as possible. These implementations closely
 * follows the SIMD versions, but operates on single float values instead.
 *
 * The following functions are defined:
 *
 *     sph2sqr_fast - maps a 3D vector to a 2D point in the unit square
 *     sqr2sph_fast - maps a 2D point in the unit square to a 3D vector
 *
 * Written by Petrik Clarberg <petrik@cs.lth.se>, Lund University, 2007-2008.
 * This code is released as public domain for use free of charge for any
 * purpose, but without any kind of warranty.
 */

#include <cmath>

namespace ks
{
namespace sph_map
{
// Coefficients for minimax approximation of sin(x*pi/4), x=[0,2].
const float s1 = 0.7853975892066955566406250000000000f;
const float s2 = -0.0807407423853874206542968750000000f;
const float s3 = 0.0024843954015523195266723632812500f;
const float s4 = -0.0000341485538228880614042282104492f;

// Coefficients for minimax approximation of cos(x*pi/4), x=[0,2].
const float c1 = 0.9999932952821962577665326692990000f;
const float c2 = -0.3083711259464511647371969120320000f;
const float c3 = 0.0157862649459062213825197189573000f;
const float c4 = -0.0002983708648233575495551227373110f;

// Coefficients for 6th degree minimax approximation of atan(x)*2/pi, x=[0,1].
const float t1 = 0.406758566246788489601959989e-5f;
const float t2 = 0.636226545274016134946890922156f;
const float t3 = 0.61572017898280213493197203466e-2f;
const float t4 = -0.247333733281268944196501420480f;
const float t5 = 0.881770664775316294736387951347e-1f;
const float t6 = 0.419038818029165735901852432784e-1f;
const float t7 = -0.251390972343483509333252996350e-1f;

// Helper functions

/// Returns the sign bit of the float a.
inline unsigned int _sign(float a) { return (*((unsigned int *)&a)) & 0x80000000; }

/// Takes the absolute value of a by clearing the sign bit to 0.
inline void _abs(float &a) { *((int *)&a) &= 0x7fffffff; }

/// Flips the sign of a by XOR'ing with the sign bit in s.
inline void _flip(float &a, unsigned int s)
{
    unsigned int bits = *((unsigned int *)&a);
    *((int *)&a) = bits ^ s;
}

//  ------------------------------------------------------------------------
/// Transform a 2D position p=(u,v) in the unit square to a normalized 3D
/// vector on the unit sphere. Optimized scalar implementation.
//  ------------------------------------------------------------------------
inline vec3 square_to_sphere(const vec2 &p)
{
    // Transform p from [0,1] to [-1,1]
    float u = 2.f * p.x() - 1.f;
    float v = 2.f * p.y() - 1.f;

    // Store the sign bits of u,v for later use
    unsigned int sign_u = _sign(u);
    unsigned int sign_v = _sign(v);

    // Take the absolute values to move u,v to the first quadrant
    _abs(u);
    _abs(v);

    // Compute the radius based on the signed distance along the diagonal
    float sd = 1 - (u + v);
    float d = sd;
    _abs(d);
    float r = 1 - d;

    // Comute phi*2/pi based on u, v and r (avoid div-by-zero if r=0)
    float phi = r == 0.f ? 1.f : (v - u) / r + 1.f; // phi = [0,2)

    // Compute the z coordinate (flip sign based on signed distance)
    float r2 = r * r;
    float z = 1 - r2;
    _flip(z, _sign(sd));
    float sin_theta = r * std::sqrt(2.f - r2);

    // Approximate sin/cos
    float phi2 = phi * phi;
    float sp = s3 + s4 * phi2;
    sp = s2 + sp * phi2;
    sp = s1 + sp * phi2;
    float sin_phi = sp * phi;

    float cp = c3 + c4 * phi2;
    cp = c2 + cp * phi2;
    cp = c1 + cp * phi2;
    float cos_phi = cp;

    // Flip signs of sin/cos based on signs of u,v
    _flip(cos_phi, sign_u);
    _flip(sin_phi, sign_v);

    // Compute the x and y coordinates of the 3D vector
    float x = sin_theta * cos_phi;
    float y = sin_theta * sin_phi;

    return vec3(x, y, z);
}

//  ------------------------------------------------------------------------
/// Transforms a normalized 3D vector to a 2D position in the unit square.
/// Optimized scalar implementation using trigonometric approximations.
//  ------------------------------------------------------------------------
inline vec2 sphere_to_square(const vec3 &d)
{
    float x = d.x();
    float y = d.y();
    float z = d.z();

    // Take the absolute values of x,y,z
    _abs(x);
    _abs(y);
    _abs(z);

    // Compute the radius r
    float r = safe_sqrt(1.f - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    float a = std::max(x, y);
    float b = std::min(x, y);
    b = a == 0.f ? 0.f : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    float phi = t6 + t7 * b;
    phi = t5 + phi * b;
    phi = t4 + phi * b;
    phi = t3 + phi * b;
    phi = t2 + phi * b;
    phi = t1 + phi * b;

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1.f - phi;

    // Find (u,v) based on (r,phi)
    float v = phi * r;
    float u = r - v;

    if (d.z() < 0.f) // southern hemisphere -> mirror u,v
    {
        float tmp = u;
        u = 1.f - v;
        v = 1.f - tmp;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    _flip(u, _sign(d.x()));
    _flip(v, _sign(d.y()));

    // Transform (u,v) from [-1,1] to [0,1]
    u = 0.5f * (u + 1.f);
    v = 0.5f * (v + 1.f);

    return vec2(u, v);
}

inline void bilerp(const vec2 &pos, int N, std::array<vec2i, 4> &idx, vec4 &weight)
{
    // ASSERT(is_pow2(N));
    float x = pos.x() * N - 0.5f;
    float y = pos.y() * N - 0.5f;
    int x0 = std::floor(x);
    int y0 = std::floor(y);
    float tx = x - x0;
    float ty = y - y0;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::clamp(x0, 0, N - 1);
    y0 = std::clamp(y0, 0, N - 1);
    x1 = std::clamp(x1, 0, N - 1);
    y1 = std::clamp(y1, 0, N - 1);

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

} // namespace sph_map
} // namespace ks