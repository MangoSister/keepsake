implementing ks;

public static const float pi = 3.14159265359;
public static const float inv_pi = 0.31830988618;
public static const float half_pi = 1.57079632679;
public static const float two_pi = 6.28318530718;
public static const float fp32_before_one = nextafter(1.0f, float.minValue);
// smallest such that 1.0+FLT_EPSILON != 1.0
public static const float FLT_EPSILON = 1.192092896e-07F;
// 2^(-24); std::numeric_limits<Float>::epsilon() * 0.5;
public static const float fp32_machine_epsilon = FLT_EPSILON * 0.5;

public void swap<T>(inout T a, inout T b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

public bool is_zero<T, let N : int>(vector<T, N> v)
    where T : __BuiltinFloatingPointType { return dot(v, v) == T(0); }

// TODO: should probably report this...
public T sqr<T>(T x)
    where T : __BuiltinIntegerType { return x * x; }

[Differentiable]
public T sqr<T>(T x)
    where T : __BuiltinFloatingPointType { return x * x; }

[Differentiable]
public T length_sqr<T, let N : int>(vector<T, N> v)
    where T : __BuiltinFloatingPointType { return dot(v, v); }

[Differentiable]
public T safe_sqrt<T>(T x)
    where T : __BuiltinFloatingPointType { return sqrt(max(x, T(0))); }

[Differentiable]
public T safe_acos<T>(T x)
    where T : __BuiltinFloatingPointType { return acos(clamp(x, T(-1), T(1))); }

[Differentiable]
public T safe_pow<T>(T a, T b)
    where T : __BuiltinFloatingPointType
{
    if (a == T(0) && b == T(0))
        return T(0);
    return pow(a, b);
}

[Differentiable]
public T sinx_over_x<T>(T x)
    where T : __BuiltinFloatingPointType
{
    if (T(1) - x * x == T(1))
        return T(1);
    return sin(x) / x;
}

[Differentiable]
public T sigmoid<T>(T x)
    where T : __BuiltinFloatingPointType { return T(1) / (T(1) + exp(-x)); }

// No built-in inverse???
[Differentiable]
public float2x2 inverse(float2x2 A)
{
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return float2x2(A[1][1], -A[0][1], -A[1][0], A[0][0]) / det;
}

public matrix<T, N, N> make_diagonal<T, let N : int>(vector<T, N> diag)
    where T : IArithmetic
{
    matrix<T, N, N> m = matrix<T, N, N>(T(0));
    for (int i = 0; i < N; ++i) {
        m[i][i] = diag[i];
    }
    return m;
}

// Duff, Tom, et al. "Building an orthonormal basis, revisited." Journal of Computer Graphics Techniques Vol 6.1 (2017).
[Differentiable]
public void orthonormal_basis(float3 N, out float3 X, out float3 Y)
{
    // float sign = copysign(1.0f, N.z); // the instrinsic copysign is not differentiable...
    float sign = N.z >= 0.0f ? 1.0f : -1.0f;

    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    X = float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    Y = float3(b, sign + N.y * N.y * a, -N.y);
}

[Differentiable]
public float3x3 orthonormal_basis_to_local(float3 N)
{
    float3 X, Y;
    orthonormal_basis(N, X, Y);
    return float3x3(X, Y, N);
}

public float luminance(float3 rgb)
{
    static const float3 lum_weight = float3(0.212671f, 0.715160f, 0.072169f);
    return dot(rgb, lum_weight);
}

public struct Frame
{
    public __init(float3 t, float3 b, float3 n) { m = transpose(float3x3(t, b, n)); }

    public float3 to_local(float3 w) { return mul(w, m); }

    public float3 to_world(float3 w) { return mul(m, w); }

    public float3 tangent() { return float3(m[0][0], m[1][0], m[2][0]); }
    public float3 bitangent() { return float3(m[0][1], m[1][1], m[2][1]); }
    public float3 normal() { return float3(m[0][2], m[1][2], m[2][2]); }

    float3x3 m;
}

public Frame transform_frame(float3x3 m, Frame f)
{
    float3 t = normalize(mul(m, f.tangent()));

    float3 b = mul(m, f.bitangent());
    // Need to re-project if m is not rigid...when does this happen tho
    b = normalize(b - dot(b, t) * t);

    float3 n = normalize(cross(t, b));

    return Frame(t, b, n);
}

// Shirley-Chiu low distortion disk sampling.
public float2 sample_disk(float2 u)
{
    float a = 2.0f * u[0] - 1.0f;
    float b = 2.0f * u[1] - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        return float2(0.0);
    }
    float r, phi;
    if (a * a > b * b) {
        r = a;
        phi = 0.25f * pi * (b / a);
    } else {
        r = b;
        phi = half_pi - 0.25f * pi * (a / b);
    }
    return r * float2(cos(phi), sin(phi));
}

public float3 sample_cosine_hemisphere(float2 u)
{
    float2 d = sample_disk(u);
    float z = sqrt(max(0.0f, 1.0f - dot(d, d)));
    return float3(d.x, d.y, z);
    // pdf = z / pi;
}

// Ray tracing gem
// solid angle = 2pi * (1- cos_half_apex_angle)
public float3 sample_cone(float2 u, float cos_half_apex_angle)
{
    float cos_theta = (1 - u[0]) + u[0] * cos_half_apex_angle;
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = u[1] * 2 * pi;
    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;
    float z = cos_theta;
    return float3(x, y, z);
}

public float solid_angle_to_cone(float solid_angle) { return 1.0 - solid_angle / (2.0 * pi); }

public float cone_to_solid_angle(float cos_half_apex_angle) { return 2.0 * pi * (1 - cos_half_apex_angle); }

public int sample_small_distrib(IArray<float> data, float u, out float u_remap)
{
    int N = data.getCount();
    float sum_w = 0.0f;
    int last_positive = 0;
    for (int i = 0; i < N; ++i) {
        sum_w += data[i];
        if (data[i] > 0.0f) {
            last_positive = i;
        }
    }
    // ASSERT(sum_w > 0.0f);
    float inv_sum_w = 1.0f / sum_w;

    float cdf = 0.0f;
    int selected = -1;
    for (int i = 0; i < N; ++i) {
        float dcdf = data[i] * inv_sum_w;
        float cdf_next = cdf + dcdf;
        if (u < cdf_next) {
            selected = i;
            u_remap = (u - cdf) / (cdf_next - cdf);
            break;
        }
        cdf = cdf_next;
    }
    if (selected == -1) {
        selected = last_positive;
    }
    // ASSERT(data[selected] > 0.0f);
    return selected;
}

[Differentiable]
public float2 to_spherical(float3 dir)
{
    float theta = safe_acos(dir.z);
    float phi = atan2(dir.y, dir.x);
    if (phi < 0.0f) {
        phi += two_pi;
    }
    return float2(phi, theta);
}

[Differentiable]
public float3 to_cartesian(float2 sph)
{
    float phi = sph.x, theta = sph.y;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);
    return float3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

public float3 equal_area_square_to_sphere(float2 p)
{
    // ASSERT(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
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
    return float3(cosPhi * r * safe_sqrt(2 - sqr(r)), sinPhi * r * safe_sqrt(2 - sqr(r)), z);
}

public float2 equal_area_sphere_to_square(float3 d)
{
    // CUDA_ASSERT(length_squared(d) > .999 && length_squared(d) < 1.001);
    float x = abs(d.x), y = abs(d.y), z = abs(d.z);

    // Compute the radius r
    float r = safe_sqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    float a = max(x, y), b = min(x, y);
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

    // float phi = t6 + t7 * b;
    // phi = t5 + phi * b;
    // phi = t4 + phi * b;
    // phi = t3 + phi * b;
    // phi = t2 + phi * b;
    // phi = t1 + phi * b;
    float phi = fma(t7, b, t6);
    phi = fma(phi, b, t5);
    phi = fma(phi, b, t4);
    phi = fma(phi, b, t3);
    phi = fma(phi, b, t2);
    phi = fma(phi, b, t1);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    float v = phi * r;
    float u = r - v;

    if (d.z < 0) {
        // southern hemisphere -> mirror u,v
        // float tmp = u;
        // u = 1 - v;
        // v = 1 - tmp;
        swap(u, v);
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = copysign(u, d.x);
    v = copysign(v, d.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return float2(0.5f * (u + 1), 0.5f * (v + 1));
}

public float2 wrap_equal_area_square(float2 uv)
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

public void equal_area_bilerp(float2 uv, int N, out int2 idx[4], out float4 weight)
{
    // ASSERT(is_pow2(N));
    float x = uv.x * N - 0.5f;
    float y = uv.y * N - 0.5f;
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
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

    bool m00 = (bool)((x0 ^ y0) & N);
    bool m01 = (bool)((x1 ^ y0) & N);
    bool m10 = (bool)((x0 ^ y1) & N);
    bool m11 = (bool)((x1 ^ y1) & N);

    int N_1 = N - 1;
    x0 = x0 & N_1;
    y0 = y0 & N_1;
    x1 = x1 & N_1;
    y1 = y1 & N_1;

    if (m00)
        idx[0] = int2(N_1 - x0, N_1 - y0);
    else
        idx[0] = int2(x0, y0);

    if (m01)
        idx[1] = int2(N_1 - x1, N_1 - y0);
    else
        idx[1] = int2(x1, y0);

    if (m10)
        idx[2] = int2(N_1 - x0, N_1 - y1);
    else
        idx[2] = int2(x0, y1);

    if (m11)
        idx[3] = int2(N_1 - x1, N_1 - y1);
    else
        idx[3] = int2(x1, y1);
}

public float3 spherical_fibonacci(int i, int N)
{
    static const float golden_ratio = 1.61803398875f;
    float phi = two_pi * ((float)i / golden_ratio - floor((float)i / golden_ratio));
    float z = 1.0f - (float)(2 * i + 1) / (float)N;
    return to_cartesian(float2(phi, acos(z)));
}

public float rounding_error_gamma(int n) { return (n * fp32_machine_epsilon) / (1 - n * fp32_machine_epsilon); }

public bool isect_ray_aabb(float3 bmin, float3 bmax, float3 o, float3 d, float tmin, float tmax, out float hitt0,
                           out float hitt1)
{
    float t0 = tmin, t1 = tmax;
    [[ForceUnroll]]
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        float invRayDir = 1 / d[i];
        float tNear = (bmin[i] - o[i]) * invRayDir;
        float tFar = (bmax[i] - o[i]) * invRayDir;
        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) {
            swap(tNear, tFar);
        }
        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * rounding_error_gamma(3);

        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) {
            hitt0 = hitt1 = 0.0;
            return false;
        }
    }

    hitt0 = t0;
    hitt1 = t1;
    return true;
}

public bool isect_ray_aabb(float3 b[2], float3 o, float3 d, float raytMax, float3 invDir, int3 dirIsNeg)
{
    // Check for ray intersection against $x$ and $y$ slabs
    float tMin = (b[dirIsNeg[0]].x - o.x) * invDir.x;
    float tMax = (b[1 - dirIsNeg[0]].x - o.x) * invDir.x;
    float tyMin = (b[dirIsNeg[1]].y - o.y) * invDir.y;
    float tyMax = (b[1 - dirIsNeg[1]].y - o.y) * invDir.y;
    // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
    tMax *= 1 + 2 * rounding_error_gamma(3);
    tyMax *= 1 + 2 * rounding_error_gamma(3);

    if (tMin > tyMax || tyMin > tMax)
        return false;
    if (tyMin > tMin)
        tMin = tyMin;
    if (tyMax < tMax)
        tMax = tyMax;

    // Check for ray intersection against $z$ slab
    float tzMin = (b[dirIsNeg[2]].z - o.z) * invDir.z;
    float tzMax = (b[1 - dirIsNeg[2]].z - o.z) * invDir.z;
    // Update _tzMax_ to ensure robust bounds intersection
    tzMax *= 1 + 2 * rounding_error_gamma(3);

    if (tMin > tzMax || tzMin > tMax)
        return false;
    if (tzMin > tMin)
        tMin = tzMin;
    if (tzMax < tMax)
        tMax = tzMax;

    return (tMin < raytMax) && (tMax > 0);
}

// Wächter, Carsten, and Nikolaus Binder. "A fast and robust method for avoiding self-intersection." Ray Tracing Gems:
// High-Quality and Real-Time Rendering with DXR and Other APIs (2019): 77-85.
//
//  Normal points outward for rays exiting the surface, else is flipped.
float3 offset_ray(float3 p, float3 n)
{
    static const float origin = 1.0f / 32.0f;
    static const float float_scale = 1.0f / 65536.0f;
    static const float int_scale = 256.0f;

    int3 of_i = int3((int)(int_scale * n.x), (int)(int_scale * n.y), (int)(int_scale * n.z));
    float3 p_i = float3(asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return float3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
                  abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
                  abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

public RayDesc spawn_ray(float3 origin, float3 dir, float3 ng, float tnear, float tfar)
{
    RayDesc r;
    r.Origin = offset_ray(origin, dot(dir, ng) > 0.0f ? ng : -ng);
    r.Direction = dir;
    r.TMin = tnear;
    r.TMax = tfar;
    return r;
}

// https://shader-slang.com/slang/user-guide/a1-04-interop.html
public float spv_atomic_add(__ref float value, float amount)
{
    return spirv_asm
    {
        OpExtension "SPV_EXT_shader_atomic_float_add";
        OpCapability AtomicFloat32AddEXT;
        result:$$float = OpAtomicFAddEXT &value Device None $amount
    };
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// Also pbrt hair doc
// TODO: as of Haswell, the PEXT instruction could do all this in a
// single instruction.

// "Insert" a 0 bit after each of the 16 low bits of x
public uint part_1_by_1(uint x)
{
    x &= 0x0000ffff;                 // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
public uint part_1_by_2(uint x)
{
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

// Inverse of part_1_by_1 - "delete" all odd-indexed bits
public uint compact_1_by_1(uint x)
{
    x &= 0x55555555;                 // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

// Inverse of part_1_by_2 - "delete" all bits not at positions divisible by 3
public uint compact_1_by_2(uint x)
{
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

public uint encode_morton_2(uint2 v) { return (part_1_by_1(v.y) << 1) + part_1_by_1(v.x); }

public uint encode_morton_3(uint3 v) { return (part_1_by_2(v.z) << 2) + (part_1_by_2(v.y) << 1) + part_1_by_2(v.x); }

public uint2 decode_morton_2(uint32_t code)
{
    uint x = compact_1_by_1(code >> 0);
    uint y = compact_1_by_1(code >> 1);
    return uint2(x, y);
}

public uint3 decode_morton_3(uint32_t code)
{
    uint x = compact_1_by_2(code >> 0);
    uint y = compact_1_by_2(code >> 1);
    uint z = compact_1_by_2(code >> 2);
    return uint3(x, y, z);
}

// https://rosettacode.org/wiki/Cholesky_decomposition
public matrix<T, 3, 3> cholesky_decompose<T>(matrix<T, 3, 3> A)
    where T : __BuiltinFloatingPointType
{
    // Assume A is SPD (symmetric positive definite)
    T l11 = sqrt(A[0][0]);
    T l21 = T(1) / l11 * A[1][0];
    T l22 = sqrt(A[1][1] - sqr(l21));
    T l31 = T(1) / l11 * A[2][0];
    T l32 = T(1) / l22 * (A[2][1] - l31 * l21);
    T l33 = sqrt(A[2][2] - (sqr(l31) + sqr(l32)));
    matrix<T, 3, 3> L;
    // clang-format off
    L[0][0] = l11; L[0][1] = T(0);  L[0][2] = T(0);
    L[1][0] = l21; L[1][1] = l22;   L[1][2] = T(0);
    L[2][0] = l31; L[2][1] = l32;   L[2][2] = l33;
    // clang-format on
    return L;
}
