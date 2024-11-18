implementing ks;

public static const float pi = 3.14159265359;
public static const float inv_pi = 0.31830988618;
public static const float half_pi = 1.57079632679;
public static const float two_pi = 6.28318530718;
public static const float fp32_before_one = nextafter(1.0f, float.minValue);

public float sqr(float x) { return x * x; }

public float safe_sqrt(float x) { return sqrt(max(x, 0.0)); }

public float safe_pow(float a, float b)
{
    if (a == 0.0 && b == 0.0)
        return 0.0;
    return pow(a, b);
}

// Duff, Tom, et al. "Building an orthonormal basis, revisited." Journal of Computer Graphics Techniques Vol 6.1 (2017).
public void orthonormal_basis(float3 N, out float3 X, out float3 Y)
{
    float sign = copysign(1.0f, N.z);
    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    X = float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    Y = float3(b, sign + N.y * N.y * a, -N.y);
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

float2 sample_disk(float2 u)
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

float3 sample_cosine_hemisphere(float2 u)
{
    float2 d = sample_disk(u);
    float z = sqrt(max(0.0f, 1.0f - dot(d, d)));
    return float3(d.x, d.y, z);
    // pdf = z / pi;
}

int sample_small_distrib(IArray<float> data, float u, out float u_remap)
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