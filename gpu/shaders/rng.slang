implementing ks;

public struct RNG
{
    static const uint64_t PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL;
    static const uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    static const uint64_t PCG32_MULT = 0x5851f42d4c957f2dULL;

    public __init()
    {
        state = PCG32_DEFAULT_STATE;
        inc = PCG32_DEFAULT_STREAM;
    }

    public __init(uint64_t seq_index, uint64_t offset) { set_seq(seq_index, offset); }

    public __init(uint64_t seq_index) { set_seq(seq_index); }

    [mutating]
    public void set_seq(uint64_t sequence_index, uint64_t offset)
    {
        state = 0u;
        inc = (sequence_index << 1u) | 1u;
        next_u32();
        state += offset;
        next_u32();
    }

    [mutating]
    public void set_seq(uint64_t sequence_index) { set_seq(sequence_index, mix_bits(sequence_index)); }

    [mutating]
    public uint next_u32()
    {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint rot = (uint)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    [mutating]
    public uint64_t next_u64()
    {
        uint64_t v0 = next_u32(), v1 = next_u32();
        return (v0 << 32) | v1;
    }

    [mutating]
    public float next() { return min(fp32_before_one, next_u32() * 0x1p-32f); }

    [mutating]
    public float next_f32() { return next(); }

    [mutating]
    public double next_f64() { return min<double>((double)fp32_before_one, (double)next_u64() * (double)0x1p-64); }

    [mutating]
    public float2 next2d() { return float2(next(), next()); }

    [mutating]
    public float3 next3d() { return float3(next(), next(), next()); }

    [mutating]
    public float4 next4d() { return float4(next(), next(), next(), next()); }

    [mutating]
    public void advance(int64_t idelta)
    {
        uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u;
        uint64_t acc_plus = 0u, delta = (uint64_t)idelta;
        while (delta > 0) {
            if ((delta & 1) > 0) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        state = acc_mult * state + acc_plus;
    }

    public uint64_t state;
    public uint64_t inc;
}

public uint xxhash32_1(uint p)
{
    static const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

public uint xxhash32_2(uint2 p)
{
    static const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

public uint xxhash32_3(uint3 p)
{
    static const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

public uint xxhash32_4(uint4 p)
{
    static const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

// See latest boost for more information about hash mixing/combining
// https://www.boost.org/doc/libs/1_86_0/boost/intrusive/detail/hash_mix.hpp
public uint mix_bits_u32(uint x)
{
    static const uint m1 = 0x21f0aaad;
    static const uint m2 = 0x735a2d97;

    x ^= x >> 16;
    x *= m1;
    x ^= x >> 15;
    x *= m2;
    x ^= x >> 15;

    return x;
}

// (lastest boost now uses a different variant)
// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
public uint64_t mix_bits(uint64_t v)
{
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185ULL;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44dULL;
    v ^= (v >> 33);
    return v;
}

public void hash_combine(inout uint64_t seed, uint64_t value) { seed = mix_bits(seed + 0x9e3779b9 + value); }

public void hash_combine_u32(inout uint seed, uint value) { seed = mix_bits_u32(seed + 0x9e3779b9 + value); }

// TODO: low discrepency sequences

public struct PTRenderSampler
{
    public __init() {}

    public __init(uint2 full_resolution, uint2 pixel, uint samples_per_pixel, uint sample_index, uint32_t seed)
    {
        rng = RNG((uint64_t)xxhash32_3(uint3(pixel.x, pixel.y, seed)));
        // Assume each sample uses <65536 dims...
        rng.advance(sample_index * 65536ull);
    }
    public RNG rng;
}