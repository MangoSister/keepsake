#pragma once
#include "math.cuh"
#include "vecmath.cuh"

namespace ksc
{

#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

CUDA_HOST_DEVICE inline uint64_t MixBits(uint64_t v)
{
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

struct RNG
{
    // RNG Public Methods
    CUDA_HOST_DEVICE
    RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}
    CUDA_HOST_DEVICE
    RNG(uint64_t seqIndex, uint64_t offset) { SetSequence(seqIndex, offset); }
    CUDA_HOST_DEVICE
    RNG(uint64_t seqIndex) { SetSequence(seqIndex); }

    CUDA_HOST_DEVICE
    void SetSequence(uint64_t sequenceIndex, uint64_t offset);
    CUDA_HOST_DEVICE
    void SetSequence(uint64_t sequenceIndex) { SetSequence(sequenceIndex, MixBits(sequenceIndex)); }

    template <typename T>
    CUDA_HOST_DEVICE T Uniform();

    template <typename T>
    CUDA_HOST_DEVICE typename std::enable_if_t<std::is_integral_v<T>, T> Uniform(T b)
    {
        T threshold = (~b + 1u) % b;
        while (true) {
            T r = Uniform<T>();
            if (r >= threshold)
                return r % b;
        }
    }

    CUDA_HOST_DEVICE
    void Advance(int64_t idelta);
    CUDA_HOST_DEVICE
    int64_t operator-(const RNG &other) const;

  private:
    // RNG Private Members
    uint64_t state, inc;
};

// RNG Inline Method Definitions
template <typename T>
inline T RNG::Uniform()
{
    return T::unimplemented;
}

template <>
inline uint32_t RNG::Uniform<uint32_t>();

template <>
inline uint32_t RNG::Uniform<uint32_t>()
{
    uint64_t oldstate = state;
    state = oldstate * PCG32_MULT + inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

template <>
inline uint64_t RNG::Uniform<uint64_t>()
{
    uint64_t v0 = Uniform<uint32_t>(), v1 = Uniform<uint32_t>();
    return (v0 << 32) | v1;
}

// template <>
// inline int32_t RNG::Uniform<int32_t>()
//{
//     // https://stackoverflow.com/a/13208789
//     uint32_t v = Uniform<uint32_t>();
//     if (v <= INT_MAX)
//         return int32_t(v);
//     CUDA_ASSERT(v >= INT_MIN);
//     return int32_t(v - INT_MIN) + INT_MIN;
// }

inline void RNG::SetSequence(uint64_t sequenceIndex, uint64_t seed)
{
    state = 0u;
    inc = (sequenceIndex << 1u) | 1u;
    Uniform<uint32_t>();
    state += seed;
    Uniform<uint32_t>();
}

template <>
inline float RNG::Uniform<float>()
{
    return min(ksc::float_before_one, Uniform<uint32_t>() * 0x1p-32f);
}

template <>
inline vec2 RNG::Uniform<vec2>()
{
    return vec2(Uniform<float>(), Uniform<float>());
}

template <>
inline vec3 RNG::Uniform<vec3>()
{
    return vec3(Uniform<float>(), Uniform<float>(), Uniform<float>());
}

template <>
inline vec4 RNG::Uniform<vec4>()
{
    return vec4(Uniform<float>(), Uniform<float>(), Uniform<float>(), Uniform<float>());
}

inline void RNG::Advance(int64_t idelta)
{
    uint64_t curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
    uint64_t accPlus = 0u, delta = (uint64_t)idelta;
    while (delta > 0) {
        if (delta & 1) {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    state = accMult * state + accPlus;
}

CUDA_HOST_DEVICE
inline int64_t RNG::operator-(const RNG &other) const
{
    CUDA_ASSERT(inc == other.inc);
    uint64_t curMult = PCG32_MULT, curPlus = inc, curState = other.state;
    uint64_t theBit = 1u, distance = 0u;
    while (state != curState) {
        if ((state & theBit) != (curState & theBit)) {
            curState = curState * curMult + curPlus;
            distance |= theBit;
        }
        CUDA_ASSERT((state & theBit) == (curState & theBit));
        theBit <<= 1;
        curPlus = (curMult + 1ULL) * curPlus;
        curMult *= curMult;
    }
    return (int64_t)distance;
}

CUDA_HOST_DEVICE inline float sample_exp(float u, float a) { return -log(1 - u) / a; }

CUDA_HOST_DEVICE inline vec2 sample_disk(const vec2 &u)
{
    float a = 2.0f * u[0] - 1.0f;
    float b = 2.0f * u[1] - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        return vec2::zero();
    }
    float r, phi;
    if (a * a > b * b) {
        r = a;
        phi = 0.25f * pi * (b / a);
    } else {
        r = b;
        phi = 0.5f * pi - 0.25f * pi * (a / b);
    }
    return r * vec2(cos(phi), sin(phi));
}

CUDA_HOST_DEVICE inline vec3 sample_uniform_sphere(vec2 u)
{
    float z = 1.0f - 2.0f * u.y;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = pi * 2.0f * u.x;
    float sin_phi = sin(phi);
    float cos_phi = cos(phi);
    return vec3(r * cos_phi, r * sin_phi, z);
}

CUDA_HOST_DEVICE inline vec3 sample_uniform_hemisphere_concentric(vec2 u, vec3 n)
{
    vec3 w = concentric_square_to_hemisphere(u);
    vec3 t, b;
    orthonormal_basis(n, t, b);
    return w.x * t + w.y * b + w.z * n;
}

CUDA_HOST_DEVICE inline vec3 sample_cosine_hemisphere(const vec2 &u)
{
    vec2 d = sample_disk(u);
    float z = sqrt(max(0.0f, 1.0f - length_squared(d)));
    return vec3(d.x, d.y, z);
    // pdf = z / pi;
}

CUDA_HOST_DEVICE inline vec3 sample_cosine_hemisphere(const vec2 &u, const vec3 &n)
{
    float a = 1.0f - 2.0f * u[0];
    float b = safe_sqrt(1.0f - a * a);
    // Avoid zero vector output in case the sample is in the opposite direction of n.
    a *= 0.999f;
    b *= 0.999f;

    float phi = 2.0f * pi * u[1];
    float x = n.x + b * cos(phi);
    float y = n.y + b * sin(phi);
    float z = n.z + a;
    return normalize(vec3(x, y, z));
}

// Sample a tetrahedron: http://vcg.isti.cnr.it/jgt/tetra.htm
// Only return barycentric coordiantes
CUDA_HOST_DEVICE inline vec4 sample_tetrahedron(float s, float t, float u)
{
    if (s + t > 1.0f) { // cut'n fold the cube into a prism
        s = 1.0f - s;
        t = 1.0f - t;
    }
    if (t + u > 1.0f) { // cut'n fold the prism into a tetrahedron
        float tmp = u;
        u = 1.0f - s - t;
        t = 1.0f - tmp;
    } else if (s + t + u > 1.0f) {
        float tmp = u;
        u = s + t + u - 1.0f;
        s = 1.0f - t - tmp;
    }
    float a = 1.0f - s - t - u; // a,s,t,u are the barycentric coordinates of the random point.
    return vec4(a, s, t, u);
}

// Return the actual point
CUDA_HOST_DEVICE inline vec3 sample_tetrahedron(const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v3,
                                                const vec3 &xi)
{
    float s = xi[0];
    float t = xi[1];
    float u = xi[2];
    vec4 coord = sample_tetrahedron(s, t, u);
    return v0 * coord[0] + v1 * coord[1] + v2 * coord[2] + v3 * coord[3];
}

CUDA_HOST_DEVICE inline int sample_small_discrete(cuda::std::span<const float> data, float u, float *u_remap = nullptr)
{
    int N = (int)data.size();
    float sum_w = 0.0f;
    int last_positive = 0;
    for (int i = 0; i < N; ++i) {
        sum_w += data[i];
        if (data[i] > 0.0f) {
            last_positive = i;
        }
    }
    CUDA_ASSERT(sum_w > 0.0f);
    float inv_sum_w = 1.0f / sum_w;

    float cdf = 0.0f;
    int selected = -1;
    for (int i = 0; i < N; ++i) {
        float dcdf = data[i] * inv_sum_w;
        float cdf_next = cdf + dcdf;
        if (u < cdf_next) {
            selected = i;
            if (u_remap) {
                *u_remap = (u - cdf) / (cdf_next - cdf);
            }
            break;
        }
        cdf = cdf_next;
    }
    if (selected == -1) {
        selected = last_positive;
    }
    CUDA_ASSERT(data[selected] > 0.0f);
    return selected;
}

// https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
// h = (sqrt(1 - u) sin(2 \pi v), sqrt(1 - u) cos(2 \pi v), sqrt(u) sin(2 \pi w), sqrt(u) cos(2 \pi w))
CUDA_HOST_DEVICE inline quat sample_uniform_rotation(float u, float v, float w)
{
    ksc::quat q;
    q.w = std::sqrt(1 - u) * std::sin(2 * ksc::pi * v);
    q.v.x = std::sqrt(1 - u) * std::cos(2 * ksc::pi * v);
    q.v.y = std::sqrt(u) * std::sin(2 * ksc::pi * w);
    q.v.z = std::sqrt(u) * std::cos(2 * ksc::pi * w);
    return q;
}

// Gammell, Jonathan D., and Timothy D. Barfoot. "The probability density function of a transformation-based
// hyperellipsoid sampling technique." arXiv preprint arXiv:1404.1347 (2014).
CUDA_HOST_DEVICE
inline ksc::vec3 sample_uniform_ellipsoid(const ksc::mat3 &L, ksc::vec3 center, ksc::vec2 u, ksc::vec3 *sph = nullptr)
{
    ksc::vec3 p = ksc::sample_uniform_sphere(u);
    if (sph) {
        *sph = p;
    }
    return L * p + center;
}

CUDA_HOST_DEVICE
inline ksc::vec3 sample_uniform_ellipsoid_equal_area(const ksc::mat3 &L, ksc::vec3 center, ksc::vec2 u,
                                                     ksc::vec3 *sph = nullptr)
{
    ksc::vec3 p = ksc::equal_area_square_to_sphere(u);
    if (sph) {
        *sph = p;
    }
    return L * p + center;
}

} // namespace ksc