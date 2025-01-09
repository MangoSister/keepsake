implementing ks;

public struct LightSample
{
    public __init()
    {
        Le_beta = float3(0.0);
        pdf_wi = 0.0;
        wi = float3(0.0);
        wi_dist = 0.0;
    }

    public bool success() { return pdf_wi > 0.0; }

    public float3 Le_beta;
    public float pdf_wi;
    public float3 wi;
    public float wi_dist;
}

public interface ILight
{
    bool delta_position();
    bool delta_direction();
    bool delta();

    LightSample sample(float3 p_shade, float2 u, LightSystem system);
    float pdf(float3 p_shade, float3 wi, float wi_dist, LightSystem system);
}

struct DirectionalLight : ILight
{
    __init(float3 _L, float3 _dir)
    {
        L = _L;
        dir = _dir;
    }
    bool delta_position() { return false; }
    bool delta_direction() { return true; }
    bool delta() { return delta_direction() || delta_position(); }

    LightSample sample(float3 p_shade, float2 u, LightSystem system)
    {
        LightSample s;
        s.Le_beta = L;
        s.pdf_wi = 1.0f;
        s.wi = dir;
        s.wi_dist = float.maxValue;
        return s;
    }

    float pdf(float3 p_shade, float3 wi, float wi_dist, LightSystem system) { return 0.0f; }

    float3 L;
    float3 dir;
}

struct PointLight : ILight
{
    __init(float3 _I, float3 _pos)
    {
        I = _I;
        pos = _pos;
    }

    bool delta_position() { return true; }
    bool delta_direction() { return false; }
    bool delta() { return delta_direction() || delta_position(); }

    LightSample sample(float3 p_shade, float2 u, LightSystem system)
    {
        LightSample s;
        s.wi = pos - p_shade;
        float l2 = length_sqr(s.wi);
        if (l2 == 0.0f) {
            return LightSample();
        }
        float l = sqrt(l2);
        s.Le_beta = I / l2;
        s.pdf_wi = 1.0f;
        s.wi /= l;
        s.wi_dist = l;
        return s;
    }

    float pdf(float3 p_shade, float3 wi, float wi_dist, LightSystem system) { return 0.0f; }

    float3 I;
    float3 pos;
}

public struct EqualAreaSkyLight : ILight
{
    public __init(uint tex_idx_)
    {
        tex_idx = tex_idx_;
        to_world = Quaternion();
        strength = 1.0;
    }

    public bool delta_position() { return false; }
    public bool delta_direction() { return false; }
    public bool delta() { return delta_direction() || delta_position(); }

    uint get_map_res(Sampler2D<float4> map)
    {
        uint width, height, num_levels;
        map.GetDimensions(0, width, height, num_levels);
        unused(height);
        unused(num_levels);
        // width should be the same as height
        return width;
    }

    public LightSample sample(float3 p_shade, float2 u, LightSystem system)
    {
        Sampler2D<float4> map = system.textures[tex_idx];
        AliasTable pmf = system.pmfs[tex_idx];
        uint map_res = get_map_res(map);

        AliasTableSample a = pmf.sample(u.x);
        // unravel index to 2D
        int iy = a.index / map_res;
        int ix = a.index % map_res;

        float2 p_square = (float2(ix, iy) + demux_float(u.y)) / map_res;

        LightSample s;
        s.pdf_wi = a.pmf * sqr(map_res) / (4.0f * pi);
        s.wi = equal_area_square_to_sphere(p_square);
        // wi = to_world.direction(wi_local);
        s.wi_dist = float.maxValue;
        s.Le_beta = eval(p_square, system) / s.pdf_wi;
        return s;
    }

    public float pdf(float3 p_shade, float3 wi, float wi_dist, LightSystem system)
    {
        Sampler2D<float4> map = system.textures[tex_idx];
        AliasTable pmf = system.pmfs[tex_idx];
        uint map_res = get_map_res(map);

        // vec3 wi_local = to_world.inv_direction(wi);
        float2 p_square = equal_area_sphere_to_square(wi);
        int iy = clamp((int)floor(p_square.y * map_res), 0, map_res - 1);
        int ix = clamp((int)floor(p_square.x * map_res), 0, map_res - 1);
        int idx = iy * map_res + ix;
        return pmf.pmf(idx) * sqr(map_res) / (4.0f * pi);
    }

    public float3 eval(float3 wi, LightSystem system)
    {
        // float3 wi_local = to_world.inv_direction(wi);
        float2 p_square = equal_area_sphere_to_square(wi);
        return eval(p_square, system);
    }

    float3 eval(float2 p_square, LightSystem system)
    {
        Sampler2D<float4> map = system.textures[tex_idx];
        uint map_res = get_map_res(map);

        int2 idx[4];
        float4 weight;
        equal_area_bilerp(p_square, map_res, idx, weight);

        float3 Le = float3(0);
        for (int i = 0; i < 4; ++i) {
            // should be nearest because we will manully perform bilerp.
            float2 st = ((float2)idx[i] + 0.5) / float2(map_res, map_res);
            Le += weight[i] * map.SampleLevel(st, 0).rgb;
        }
        Le *= strength;

        return Le;
    }

    // Equal-area parameterization
    Quaternion to_world; // TODO
    float strength;
    uint tex_idx;
}

public enum LightFlagBits : int
{
    Directional = (1 << 0),
    Point = (1 << 1),
    Sky = (1 << 2),
}

typedef uint32_t LightFlag;

public struct LightHeader
{
    public float3 field1;
    public LightFlag flag;
    public float3 field2;
    public int ext;
}

public struct LightSystemUniforms
{
    public uint num_skylights;
}

public struct LightSystemGlobal
{
    public LightSystemUniforms uniforms;
    // Skylights are always at front.
    public StructuredBuffer<LightHeader, ScalarDataLayout> headers;
    public AliasTable pmf;
}

public struct LightTextureArray
{
    public __subscript(int i)->Sampler2D<float4>
    {
        get { return t[i]; }
    }

    public Sampler2D<float4> t[];
}

public struct LightPMFArray
{
    public __subscript(int i)->AliasTable
    {
        get {
            return AliasTable(a[i]);
        }
    }

    public StructuredBuffer<AliasTableBin, ScalarDataLayout> a[];
}

public struct LightSystem
{
    public uint num_lights()
    {
        uint N, stride;
        global.headers.GetDimensions(N, stride);
        unused(stride);
        return N;
    }

    public ILight get_light(int index)
    {
        LightHeader h = global.headers[index];
        if ((bool)(h.flag & (LightFlagBits::Directional))) {
            return DirectionalLight(h.field1, h.field2);
        } else if ((bool)(h.flag & (LightFlagBits::Point))) {
            return PointLight(h.field1, h.field2);
        } else if ((bool)(h.flag & (LightFlagBits::Sky))) {
            return get_skylight(index);
        } else {
            return DirectionalLight(float3(1, 0, 1), float3(0, 0, 1));
        }
    }

    public uint num_skylights() { return global.uniforms.num_skylights; }

    public EqualAreaSkyLight get_skylight(int index)
    {
        LightHeader h = global.headers[index];
        return EqualAreaSkyLight(h.ext);
    }

    public ILight sample(float u, out float pr)
    {
        AliasTableSample alias_sample = global.pmf.sample(u);
        pr = alias_sample.pmf;
        return get_light(alias_sample.index);
    }

    public float probability(uint32_t light_index) { return global.pmf.pmf(light_index); }

    public ParameterBlock<LightSystemGlobal> global;
    public ParameterBlock<LightTextureArray> textures;
    public ParameterBlock<LightPMFArray> pmfs;
}
