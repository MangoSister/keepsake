// shaders.slang

struct GPUMeshDataAddresses
{
    uint64_t vertices_address;
    uint64_t texcoords_address;
    uint64_t vertex_normals_address;
    uint64_t indices_address;
};

struct RayDiff
{
    float3 rx_origin;
    float3 rx_dir;
    float3 ry_origin;
    float3 ry_dir;
};

struct Frame
{
    __init(float3 t, float3 b, float3 n) { m = transpose(float3x3(t, b, n)); }

    float3 to_local(float3 w) { return mul(w, m); }

    float3 to_world(float3 w) { return mul(m, w); }

    float3 tangent() { return float3(m[0][0], m[1][0], m[2][0]); }
    float3 bitangent() { return float3(m[0][1], m[1][1], m[2][1]); }
    float3 normal() { return float3(m[0][2], m[1][2], m[2][2]); }

    float3x3 m;
}

Frame transform_frame(float3x3 m, Frame f)
{
    float3 t = normalize(mul(m, f.tangent()));

    float3 b = mul(m, f.bitangent());
    // Need to re-project if m is not rigid...when does this happen tho
    b = normalize(b - dot(b, t) * t);

    float3 n = normalize(cross(t, b));

    return Frame(t, b, n);
}

struct Intersection
{
    // Attempts to alleviate the usual shading normal / normal map problems
    // by forcing the vector to stay in the same hemisphere before/after transform.
    float3 sh_vector_to_local(float3 world)
    {
        bool hw = dot(frame.normal(), world) >= 0.0;
        float3 local = sh_frame.to_local(world);
        bool hl = local.z >= 0.0;
        if (hw != hl) {
            local.z *= -1.0;
        }
        return local;
    }

    float3 sh_vector_to_world(float3 local)
    {
        bool hl = local.z >= 0.0;
        float3 world = sh_frame.to_world(local);
        bool hw = dot(frame.normal(), world) >= 0.0;
        if (hl != hw) {
            world -= 2.0 * dot(frame.normal(), world) * frame.normal();
        }
        return world;
    }

    Frame frame;
    Frame sh_frame;
    float3 dpdu;
    float3 dpdv;
    float3 p;
    float2 uv;
    float dudx, dvdx, dudy, dvdy;
};

Intersection compute_intersection(GPUMeshDataAddresses mesh_data, uint prim_id,
                                  BuiltInTriangleIntersectionAttributes attr, float3 ray_dir, float3x4 object_to_world,
                                  float3x4 world_to_object, bool two_sided)
{
    // TODO: optimization, for example some normalizations are not necessary.

    Intersection it;

    // TODO: can I have ConstBufferPointer<float3> with scalar layout????
    ConstBufferPointer<uint> indices = ConstBufferPointer<uint>::fromUInt(mesh_data.indices_address);
    ConstBufferPointer<float> vertices = ConstBufferPointer<float>::fromUInt(mesh_data.vertices_address);
    ConstBufferPointer<float> texcoords = ConstBufferPointer<float>::fromUInt(mesh_data.texcoords_address);
    ConstBufferPointer<float> vertex_normals = ConstBufferPointer<float>::fromUInt(mesh_data.vertex_normals_address);

    float3 barycentrics =
        float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    uint3 ind = uint3(indices[3 * prim_id], indices[3 * prim_id + 1], indices[3 * prim_id + 2]);

    float3 p[3];
    p[0] = float3(vertices[3 * ind[0]], vertices[3 * ind[0] + 1], vertices[3 * ind[0] + 2]);
    p[1] = float3(vertices[3 * ind[1]], vertices[3 * ind[1] + 1], vertices[3 * ind[1] + 2]);
    p[2] = float3(vertices[3 * ind[2]], vertices[3 * ind[2] + 1], vertices[3 * ind[2] + 2]);

    // Numerically, calculating intersection point based on surface parameterization
    // is much better than based on ray equation. (At least that's the case for CPU ray tracing...)
    it.p = p[0] * barycentrics.x + p[1] * barycentrics.y + p[2] * barycentrics.z;

    float2 uv[3];
    if (texcoords.isValid()) {
        uv[0] = float2(texcoords[2 * ind[0]], texcoords[2 * ind[0] + 1]);
        uv[1] = float2(texcoords[2 * ind[1]], texcoords[2 * ind[1] + 1]);
        uv[2] = float2(texcoords[2 * ind[2]], texcoords[2 * ind[2] + 1]);
    } else {
        uv[0] = float2(0, 0);
        uv[1] = float2(1, 0);
        uv[2] = float2(1, 1);
    }
    it.uv = uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;

    float2 duv02 = uv[0] - uv[2];
    float2 duv12 = uv[1] - uv[2];
    float3 dp02 = p[0] - p[2];
    float3 dp12 = p[1] - p[2];
    float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
    bool uv_degenerate = abs(determinant) < 1e-8f;
    if (!uv_degenerate) {
        float inv_det = 1.0 / determinant;
        it.dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * inv_det;
        it.dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * inv_det;
    }
    // TODO: pbrt also checks and handles degenerate dpdu/dpdv at this point.

    // re-orthogonalize for geometry frame (ng is respected).
    float3 ng = normalize(cross(p[1] - p[0], p[2] - p[1]));
    // TODO: pbrt also checks and handles degenerate ng (by switching to double).
    float3 b = normalize(cross(ng, it.dpdu));
    float3 t = normalize(cross(b, ng));
    if (two_sided) {
        float3 wo_object = mul(world_to_object, float4(ray_dir, 0.0));
        if (dot(wo_object, ng) < 0.0) {
            ng = -ng;
            it.dpdu = -it.dpdu;
            t = -t;
        }
    }

    it.frame = Frame(t, b, ng);

    if (!vertex_normals.isValid()) {
        it.sh_frame = it.frame;
    } else {
        float3 vn0 = float3(vertex_normals[3 * ind[0]], vertex_normals[3 * ind[0] + 1], vertex_normals[3 * ind[0] + 2]);
        float3 vn1 = float3(vertex_normals[3 * ind[1]], vertex_normals[3 * ind[1] + 1], vertex_normals[3 * ind[1] + 2]);
        float3 vn2 = float3(vertex_normals[3 * ind[2]], vertex_normals[3 * ind[2] + 1], vertex_normals[3 * ind[2] + 2]);
        // Computing the normal at hit position
        float3 vn = normalize(vn0 * barycentrics.x + vn1 * barycentrics.y + vn2 * barycentrics.z);
        if (any(isnan(vn))) {
            // Revert to face normal if vertex normal is bad.
            it.sh_frame = it.frame;
        } else {
            // Forcing vertex normal to be on the same side as face normal.
            if (dot(vn, it.frame.normal()) < 0.0) {
                vn = -vn;
            }
            // re-orthogonalize for geometry frame (vn is respected).
            float3 vb = normalize(cross(vn, it.dpdu));
            float3 vt = normalize(cross(vb, vn));

            it.sh_frame = Frame(vt, vb, vn);
        }
    }

    // Transform to world space.
    it.p = mul(object_to_world, float4(it.p, 1.0));
    it.dpdu = mul(object_to_world, float4(it.dpdu, 0.0));
    it.dpdv = mul(object_to_world, float4(it.dpdv, 0.0));
    it.frame = transform_frame((float3x3)object_to_world, it.frame);
    it.sh_frame = transform_frame((float3x3)object_to_world, it.sh_frame);

    // TODO: Use compute_uv_partials() to compute partials after transform
    it.dudx = it.dvdx = 0;
    it.dudy = it.dvdy = 0;

    return it;
}

float2 compute_intersection_lite(GPUMeshDataAddresses mesh_data, uint prim_id,
                                 BuiltInTriangleIntersectionAttributes attr)
{
    // TODO: can I have ConstBufferPointer<float3> with scalar layout????
    ConstBufferPointer<uint> indices = ConstBufferPointer<uint>::fromUInt(mesh_data.indices_address);
    ConstBufferPointer<float> texcoords = ConstBufferPointer<float>::fromUInt(mesh_data.texcoords_address);

    float3 barycentrics =
        float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    uint3 ind = uint3(indices[3 * prim_id], indices[3 * prim_id + 1], indices[3 * prim_id + 2]);

    float2 uv[3];
    if (texcoords.isValid()) {
        uv[0] = float2(texcoords[2 * ind[0]], texcoords[2 * ind[0] + 1]);
        uv[1] = float2(texcoords[2 * ind[1]], texcoords[2 * ind[1] + 1]);
        uv[2] = float2(texcoords[2 * ind[2]], texcoords[2 * ind[2] + 1]);
    } else {
        uv[0] = float2(0, 0);
        uv[1] = float2(1, 0);
        uv[2] = float2(1, 1);
    }
    float2 uv_interp = uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;

    return uv_interp;
}

struct BSDFSample
{
    __init()
    {
        beta = float3(0.0);
        pdf_wi = 0.0;
        wi = float3(0.0);
    }
    bool success() { return pdf_wi > 0.0; }

    float3 beta;
    float pdf_wi;
    float3 wi;
};

interface IBSDFClosure
{
    // return cosine-weighted bsdf: f*cos(theta_i)
    float3 eval(float3 wo, float3 wi);
    BSDFSample sample(float3 wo, float u_lobe, float2 u_wi);
}

interface IMaterial
{
    IBSDFClosure eval_closure(float2 uv, Sampler2D material_textures_2d[]);
}

interface INormalMap
{
    void apply_normal_map(inout Intersection it, Sampler2D material_textures_2d[]);
}

interface IOpacityMap
{
    // TODO: should opacity map support mipmapping (sub-pixel correlation)?
    bool apply_opacity_map(float2 uv, float rnd, Sampler2D material_textures_2d[]);
    // pbrt trick: just hash ray origin and dir to get a random number.
    bool apply_opacity_map(float2 uv, float3 ro, float3 rd, Sampler2D material_textures_2d[]);
}

static const float pi = 3.14159265359;
static const float inv_pi = 0.31830988618;
static const float half_pi = 1.57079632679;
static const float two_pi = 6.28318530718;
static const float fp32_before_one = nextafter(1.0f, float.minValue);

float sqr(float x) { return x * x; }

float safe_sqrt(float x) { return sqrt(max(x, 0.0)); }

float safe_pow(float a, float b)
{
    if (a == 0.0 && b == 0.0)
        return 0.0;
    return pow(a, b);
}

// Duff, Tom, et al. "Building an orthonormal basis, revisited." Journal of Computer Graphics Techniques Vol 6.1 (2017).
void orthonormal_basis(float3 N, out float3 X, out float3 Y)
{
    float sign = copysign(1.0f, N.z);
    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    X = float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    Y = float3(b, sign + N.y * N.y * a, -N.y);
}

float luminance(float3 rgb)
{
    static const float3 lum_weight = float3(0.212671f, 0.715160f, 0.072169f);
    return dot(rgb, lum_weight);
}

float fresnel_dielectric(float cos_theta_i, float eta)
{
    float sin_theta_t_2 = eta * eta * (1.0 - cos_theta_i * cos_theta_i);

    // Total internal reflection
    if (sin_theta_t_2 > 1.0)
        return 1.0;

    float cos_theta_t = safe_sqrt(1.0 - sin_theta_t_2);

    float rs = (eta * cos_theta_t - cos_theta_i) / (eta * cos_theta_t + cos_theta_i);
    float rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

    return 0.5 * (rs * rs + rp * rp);
}

float fresnel_schlick(float cos_theta_i)
{
    float schlick = saturate(1.0 - cos_theta_i);
    float schlick2 = sqr(schlick);
    schlick = schlick2 * schlick2 * schlick;
    return schlick;
}

// TODO: Slang also has built-in reflect/refract functions but with different conventions!!!!!
// Consolidate both versions!

// eta = eta_i / eta_t
// Make sure wi and n are on the same side.
bool refract_ks(float3 wi, float3 n, float eta, out float3 wt)
{
    float NdotI = dot(n, wi);
    float k = 1.0f - eta * eta * (1.0f - sqr(NdotI));
    if (k < 0.0f) {
        wt = float3(0.0);
        return false;
    }
    wt = -eta * wi + (eta * NdotI - sqrt(k)) * n;
    return true;
}

float3 reflect_ks(float3 w, float3 n) { return 2.0f * dot(n, w) * n - w; }

float GGX_D(float alpha_x, float alpha_y, float3 wm)
{
    if (wm.z <= 0.0) {
        return 0.0;
    }

    if (alpha_x == alpha_y) {
        float alpha = alpha_x;
        float a2 = alpha * alpha;
        float t = 1.0 + (a2 - 1.0) * wm.z * wm.z;
        return a2 / (pi * t * t);
    } else {
        float hx = wm.x / alpha_x;
        float hy = wm.y / alpha_y;
        float t = sqr(hx) + sqr(hy) + sqr(wm.z);
        return 1.0 / (pi * alpha_x * alpha_y * sqr(t));
    }
}

float GGX_smith_lambda(float alpha_x, float alpha_y, float3 w)
{
    if (w.z >= 1.0 || w.z <= -1.0) {
        return 0.0;
    }
    float alpha;
    if (alpha_x == alpha_y) {
        alpha = alpha_x;
    } else {
        float inv_sin_theta2 = 1.0 / (1.0 - w.z * w.z);
        float cos_phi2 = w.x * w.x * inv_sin_theta2;
        float sin_phi2 = w.y * w.y * inv_sin_theta2;
        alpha = sqrt(cos_phi2 * alpha_x * alpha_x + sin_phi2 * alpha_y * alpha_y);
    }
    float alpha2 = alpha * alpha;
    float NdotV2 = w.z * w.z;
    float t = (1.0 - NdotV2) * alpha2 / NdotV2;
    return 0.5 * (-1.0 + sqrt(1.0 + t));
}

float GGX_smith_G1(float alpha_x, float alpha_y, float3 w)
{
    return 1.0f / (1.0f + GGX_smith_lambda(alpha_x, alpha_y, w));
}

float GGX_smith_G2(float alpha_x, float alpha_y, float3 wo, float3 wi)
{
    return 1.0 / (1.0 + GGX_smith_lambda(alpha_x, alpha_y, wo) + GGX_smith_lambda(alpha_x, alpha_y, wi));

    // TODO: we need a beta function implementation on GPU.

    // See Heitz et al. 16, Appendix A.
    // lambda(w) = lambda(-w)
    // if (wo.z * wi.z >= 0.0f) {
    //     return 1.0 / (1.0 + lambda(wo) + lambda(wi));
    // } else {
    //     // beta function is symmetric
    //     return (float)std::beta(1.0 + lambda(wo), 1.0 + lambda(wi));
    // }
}

float3 GGX_sample_vndf(float alpha_x, float alpha_y, float3 wo, float2 u)
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(float3(alpha_x * wo.x, alpha_y * wo.y, wo.z));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    float r = sqrt(u.x);
    float phi = two_pi * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    float3 Ne = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z)));
    return Ne;
}

/* Given cosine between rays, return probability density that a photon bounces
 * to that direction. The g parameter controls how different it is from the
 * uniform sphere. g=0 uniform diffuse-like, g=1 close to sharp single ray. */
float single_peaked_henyey_greenstein(float cos_theta, float g)
{
    return ((1.0 - g * g) / safe_pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5)) * (inv_pi * 0.25);
};

float3 sample_henyey_greenstein(float3 D, float g, float randu, float randv, out float pdf)
{
    /* match pdf for small g */
    float cos_theta;
    bool isotropic = abs(g) < 1e-3f;

    if (isotropic) {
        cos_theta = (1.0f - 2.0f * randu);
        pdf = inv_pi * 0.25f;

    } else {
        float k = (1.0f - g * g) / (1.0f - g + 2.0f * g * randu);
        cos_theta = (1.0f + g * g - k * k) / (2.0f * g);
        pdf = single_peaked_henyey_greenstein(cos_theta, g);
    }

    float sin_theta = safe_sqrt(1.0f - cos_theta * cos_theta);
    float phi = two_pi * randv;
    float3 dir = float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    float3 T, B;
    orthonormal_basis(D, T, B);
    dir = dir.x * T + dir.y * B + dir.z * D;

    return dir;
}

inline float2 sample_disk(float2 u)
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

// TODO: In current implementation many terms can be cancelled/collapsed/shared/approximated.
// maybe we should consider performance more seriously now on GPU...

struct PrincipleBSDFClosure : IBSDFClosure
{
    static const float diffuse_trans_strategy_switch = 0.05f;

    float3 eval(float3 wo, float3 wi)
    {
        if (wo.z == 0.0 || wi.z == 0.0) {
            return float3(0.0, 0.0, 0.0);
        }
        float3 f = float3(0.0, 0.0, 0.0);
        if (wo.z > 0.0 && wi.z > 0.0) {
            f += eval_diffuse(wo, wi);
            f += eval_metallic_specular(wo, wi);
        } else if (wo.z > 0.0 && wi.z < 0.0) {
            f += eval_diffuse_transmission(wo, wi);
        }
        f += eval_dielectric_specular(wo, wi);
        return f;
    }

    float3 eval_diffuse(float3 wo, float3 wi)
    {
        float lobe_weight = (1.0 - metallic) * (1.0 - specular_trans) * (1.0 - diffuse_trans);
        if (lobe_weight == 0.0) {
            return float3(0.0, 0.0, 0.0);
        }

        float3 wh = normalize(wo + wi);
        float Fd90 = 0.5 + 2.0 * roughness * wh.z;
        float disney_retro = lerp(1.0, Fd90, fresnel_schlick(wi.z)) * lerp(1.0, Fd90, fresnel_schlick(wo.z));
        return lobe_weight * basecolor * inv_pi * disney_retro * wi.z;
    }

    float3 eval_metallic_specular(float3 wo, float3 wi)
    {
        float lobe_weight = metallic;
        if (lobe_weight == 0.0f) {
            return float3(0.0, 0.0, 0.0);
        }

        float3 wh = normalize(wo + wi);
        float D = GGX_D(ax, ay, wh);
        float G = GGX_smith_G2(ax, ay, wo, wi);
        float3 Fr = lerp(basecolor, float3(1.0, 1.0, 1.0), fresnel_schlick(dot(wo, wh)));
        float3 f = lobe_weight * D * G * Fr / (4.0 * wo.z);
        // ASSERT(f.allFinite() && (f >= 0.0f).all());

        return f;
    }

    float dielectric_specular_adjust(float wo_dot_wh)
    {
        float R0 = sqr((1.0 - ior) / (1.0 + ior));
        float schlick = fresnel_schlick(wo_dot_wh);
        return lerp(specular_r0_mul * R0, 1.0, schlick) / lerp(R0, 1.0, schlick);
    }

    float3 eval_dielectric_specular(float3 wo, float3 wi)
    {
        float lobe_weight = (1.0 - metallic);
        if (lobe_weight == 0.0) {
            return float3(0.0, 0.0, 0.0);
        }

        bool is_reflect = wo.z * wi.z >= 0.0;
        float eta = wo.z >= 0.0 ? ior : (1.0 / ior);
        float3 wh;
        if (is_reflect) {
            wh = normalize(wo + wi);
        } else {
            wh = normalize(wo + wi * eta);
            lobe_weight *= specular_trans;
        }

        if (wh.z < 0.0)
            wh = -wh;

        float D = GGX_D(ax, ay, wh);
        float G = GGX_smith_G2(ax, ay, wo, wi);
        float abs_wo_dot_wh = abs(dot(wo, wh));
        float Fr = fresnel_dielectric(abs_wo_dot_wh, 1.0 / eta);

        float3 f;
        if (is_reflect) {
            float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh);
            f = float3(D * G * (Fr * Fr_adj) / (4.0 * abs(wo.z)));
        } else {
            float denom = sqr(dot(wo, wh) + eta * dot(wi, wh));
            float specular = D * G * (1.0 - Fr) * dot(wo, wh) * dot(wi, wh) / (denom * wo.z);
            specular = abs(specular); // Flip all the negative signs.
            // TODO: if we ever need BDPT:
            // https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
            // https://github.com/mitsuba-renderer/mitsuba/blob/master/src/bsdfs/roughdielectric.cpp
            /* Missing term in the original paper: account for the solid angle
               compression when tracing radiance -- this is necessary for
               bidirectional methods */
            // float factor = (mode == TransportMode::Radiance) ? eta : 1;
            // f *= sqr(1.0 / eta) * sqr(factor);
            f = float3(specular);
        }
        f *= lobe_weight;
        // ASSERT(f.allFinite() && (f >= 0.0f).all());
        return f;
    }

    float3 eval_diffuse_transmission(float3 wo, float3 wi)
    {
        float lobe_weight = (1.0 - metallic) * (1.0 - specular_trans) * diffuse_trans;
        if (lobe_weight == 0.0) {
            return float3(0.0, 0.0, 0.0);
        }

        float fwd_scatter = single_peaked_henyey_greenstein(dot(wo, -wi), diffuse_trans_fwd);
        return lobe_weight * basecolor * inv_pi * (-wi.z) * fwd_scatter;
    }

    float4 lobe_sample_weights(float3 wo)
    {
        float lum_basecolor = luminance(basecolor);

        float wd = wo.z > 0.0f ? (1.0f - metallic) * (1.0f - specular_trans) * lum_basecolor * inv_pi : 0.0f;
        float weight_diffuse = wd * (1.0f - diffuse_trans);
        float weight_diffuse_transmission = wd * diffuse_trans;
        // wh isn't available at this point...
        float weight_metallic_specular =
            wo.z > 0.0f ? metallic * lerp(lum_basecolor, 1.0f, fresnel_schlick(wo.z)) : 0.0f;
        float eta = wo.z >= 0.0f ? ior : (1.0f / ior);
        // TODO: incorporate specular_r0_mul here? in general need a better heuristic...
        float weight_dielectric_specular = (1.0f - metallic) * fresnel_dielectric(abs(wo.z), 1.0f / eta);

        float sum =
            weight_diffuse + weight_diffuse_transmission + weight_metallic_specular + weight_dielectric_specular;
        if (sum == 0.0f) {
            return float4(0.0);
        }
        float4 weights =
            float4(weight_diffuse, weight_diffuse_transmission, weight_metallic_specular, weight_dielectric_specular);
        weights /= sum;

        // ASSERT(weights.allFinite() && (weights.array() >= 0.0f).all());
        return weights;
    }

    float3 sample_diffuse(float2 u, out float pdf)
    {
        float3 wi = sample_cosine_hemisphere(u);
        pdf = wi.z * inv_pi;
        return wi;
    }

    float3 sample_diffuse_transmission(float3 wo, float2 u, out float pdf)
    {
        // Switch to cosine sampling when approaching isotropic.
        if (diffuse_trans_fwd > diffuse_trans_strategy_switch) {
            return sample_henyey_greenstein(-wo, diffuse_trans_fwd, u[0], u[1], pdf);
        } else {
            float3 wi = sample_cosine_hemisphere(u);
            pdf = wi.z * inv_pi;
            wi.z = -wi.z;
            return wi;
        }
    }

    float3 sample_metallic_specular(float3 wo, float2 u, out float pdf)
    {
        if (wo.z == 0.0f) {
            pdf = 0.0f;
            return float3(0.0);
        }

        float3 wh = GGX_sample_vndf(ax, ay, sign(wo.z) * wo, u);
        float3 wi = reflect_ks(wo, wh);
        // side check
        if (wo.z * wi.z < 0.0f) {
            pdf = 0.0f;
            return float3(0.0);
        }

        float D = GGX_D(ax, ay, wh);
        float G1 = GGX_smith_G1(ax, ay, wo);
        pdf = D * G1 / (4.0f * abs(wo.z));
        // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        return wi;
    }

    float3 sample_dielectric_specular(float3 wo, float u_lobe, float2 u_wi, out float pdf)
    {
        if (wo.z == 0.0f) {
            pdf = 0.0f;
            return float3(0.0);
        }

        float3 wh = GGX_sample_vndf(ax, ay, sign(wo.z) * wo, u_wi);
        float D = GGX_D(ax, ay, wh);
        float G1 = GGX_smith_G1(ax, ay, wo);

        float eta = wo.z >= 0.0f ? ior : (1.0f / ior);
        float abs_wo_dot_wh = abs(dot(wo, wh));
        float Fr = fresnel_dielectric(abs(dot(wo, wh)), 1.0f / eta);
        float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh);
        float Fr_refl = Fr * Fr_adj;
        float Fr_refr = (1.0f - Fr) * specular_trans;
        if (Fr_refr == 0.0f) {
            Fr_refl = 1.0f;
        }

        float3 wi;
        if (u_lobe < Fr_refl / (Fr_refl + Fr_refr)) {
            // sample reflection
            wi = reflect_ks(wo, wh);
            // side check
            if (wo.z * wi.z < 0.0f) {
                pdf = 0.0f;
                return float3(0.0);
            }
            pdf = D * G1 / (4.0f * abs(wo.z));
            pdf *= Fr_refl / (Fr_refl + Fr_refr);
            // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        } else {
            // sample refraction
            if (!refract_ks(wo, sign(wo.z) * wh, 1.0f / eta, wi)) {
                // total internal reflection
                pdf = 0.0f;
                return float3(0.0);
            }
            // side check
            if (wo.z * wi.z > 0.0f) {
                pdf = 0.0f;
                return float3(0.0);
            }
            float denom = sqr(dot(wo, wh) + eta * dot(wi, wh));
            float jacobian = eta * eta * abs(dot(wi, wh)) / denom;
            pdf = D * G1 * abs(dot(wo, wh)) / abs(wo.z) * jacobian;
            pdf *= Fr_refr / (Fr_refl + Fr_refr);
            // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        }

        return wi;
    }

    float pdf_diffuse(float3 wo, float3 wi) { return wi.z * inv_pi; }

    float pdf_diffuse_transmission(float3 wo, float3 wi)
    {
        if (diffuse_trans_fwd > diffuse_trans_strategy_switch) {
            return single_peaked_henyey_greenstein(dot(wo, -wi), diffuse_trans_fwd);
        } else {
            return -wi.z * inv_pi;
        }
    }

    float pdf_metallic_specular(float3 wo, float3 wi)
    {
        float3 wh = normalize(wo + wi);
        float D = GGX_D(ax, ay, wh);
        float G1 = GGX_smith_G1(ax, ay, wo);
        float pdf = D * G1 / (4.0f * abs(wo.z));
        // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        return pdf;
    }

    float pdf_dielectric_specular(float3 wo, float3 wi)
    {
        bool reflect = wo.z * wi.z >= 0.0f;
        float eta = wo.z >= 0.0f ? ior : (1.0f / ior);
        float3 wh;
        if (reflect) {
            wh = normalize(wo + wi);
        } else {
            wh = normalize(wo + wi * eta);
        }

        if (wh.z < 0.0f)
            wh = -wh;

        float D = GGX_D(ax, ay, wh);
        float G1 = GGX_smith_G1(ax, ay, wo);
        float abs_wo_dot_wh = abs(dot(wo, wh));
        float Fr = fresnel_dielectric(abs(dot(wo, wh)), 1.0f / eta);
        float Fr_adj = dielectric_specular_adjust(abs_wo_dot_wh);
        float Fr_refl = Fr * Fr_adj;
        float Fr_refr = (1.0f - Fr) * specular_trans;
        if (Fr_refr == 0.0f) {
            Fr_refl = 1.0f;
        }

        float pdf;
        if (reflect) {
            pdf = D * G1 / (4.0f * abs(wo.z));
            pdf *= Fr_refl / (Fr_refl + Fr_refr);
            // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        } else {
            float denom = sqr(dot(wo, wh) + eta * dot(wi, wh));
            float jacobian = eta * eta * abs(dot(wi, wh)) / denom;
            pdf = D * G1 * abs(dot(wo, wh)) / abs(wo.z) * jacobian;
            pdf *= Fr_refr / (Fr_refl + Fr_refr);
            // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        }
        return pdf;
    }

    BSDFSample sample(float3 wo, float u_lobe, float2 u_wi)
    {
        if (wo.z == 0.0f) {
            return BSDFSample();
        }

        float4 sample_weights = lobe_sample_weights(wo);
        if (dot(sample_weights, sample_weights) == 0.0) {
            return BSDFSample();
        }
        float u_lobe_remap;
        int lobe = sample_small_distrib(sample_weights, u_lobe, u_lobe_remap);
        // 0: diffuse
        // 1: diffuse transmission
        // 2: metallic specular
        // 3: dielectric specular
        float4 pdf_lobe = float4(0.0);

        BSDFSample s;
        if (lobe == 0) {
            s.wi = sample_diffuse(u_wi, pdf_lobe[0]);
            if (dot(s.wi, s.wi) == 0.0f || pdf_lobe[0] == 0.0f) {
                s.pdf_wi = 0.0f;
                return s;
            }
            // pdf_lobe[1] = 0
            if (wo.z > 0.0f && s.wi.z > 0.0f)
                pdf_lobe[2] = pdf_metallic_specular(wo, s.wi);
            pdf_lobe[3] = pdf_dielectric_specular(wo, s.wi);
        } else if (lobe == 1) {
            s.wi = sample_diffuse_transmission(wo, u_wi, pdf_lobe[1]);
            if (dot(s.wi, s.wi) == 0.0f || pdf_lobe[1] == 0.0f) {
                s.pdf_wi = 0.0f;
                return s;
            }
            // pdf_lobe[0] = 0
            // pdf_lobe[2] = 0
            pdf_lobe[3] = pdf_dielectric_specular(wo, s.wi);
        } else if (lobe == 2) {
            s.wi = sample_metallic_specular(wo, u_wi, pdf_lobe[2]);
            if (dot(s.wi, s.wi) == 0.0f || pdf_lobe[2] == 0.0f) {
                s.pdf_wi = 0.0f;
                return s;
            }
            if (wo.z > 0.0f && s.wi.z > 0.0f)
                pdf_lobe[0] = pdf_diffuse(wo, s.wi);
            // pdf_lobe[1] = 0
            pdf_lobe[3] = pdf_dielectric_specular(wo, s.wi);
        } else {
            s.wi = sample_dielectric_specular(wo, u_lobe_remap, u_wi, pdf_lobe[3]);
            if (dot(s.wi, s.wi) == 0.0f || pdf_lobe[3] == 0.0f) {
                s.pdf_wi = 0.0f;
                return s;
            }
            if (wo.z > 0.0f && s.wi.z > 0.0f) {
                pdf_lobe[0] = pdf_diffuse(wo, s.wi);
                pdf_lobe[2] = pdf_metallic_specular(wo, s.wi);
            }
            if (wo.z > 0.0f && s.wi.z < 0.0f) {
                pdf_lobe[1] = pdf_diffuse_transmission(wo, s.wi);
            }
        }

        s.pdf_wi = dot(pdf_lobe, sample_weights);
        // ASSERT(std::isfinite(pdf) && pdf >= 0.0f);
        s.beta = eval(wo, s.wi) / s.pdf_wi;
        return s;
    }

    float3 basecolor;
    float roughness;
    float ax;
    float ay;
    float metallic;
    float ior;
    float specular_r0_mul;
    float specular_trans;
    float diffuse_trans;
    float diffuse_trans_fwd;
};

static const uint gpu_material_empty_map_id = uint(~0);

struct GPUMaterialPrincipledBSDFDataBlock : IMaterial, INormalMap, IOpacityMap
{
    static float2 transform_uv(float2 uv, float4 scale_offset) { return uv * scale_offset.xy + scale_offset.zw; }

    PrincipleBSDFClosure eval_closure(float2 uv, Sampler2D material_textures_2d[])
    {
        PrincipleBSDFClosure c;

        c.basecolor = basecolor_constant_or_scale;
        if (basecolor_map_id != gpu_material_empty_map_id) {
            c.basecolor *=
                material_textures_2d[basecolor_map_id].SampleLevel(transform_uv(uv, basecolor_uv_scale_offset), 0).rgb;
        }

        c.roughness = roughness_constant_or_scale;
        if (roughness_map_id != gpu_material_empty_map_id) {
            c.roughness *=
                material_textures_2d[roughness_map_id].SampleLevel(transform_uv(uv, roughness_uv_scale_offset), 0).r;
        }
        c.ax = c.ay = clamp(c.roughness * c.roughness, 0.001, 1.0);

        c.metallic = metallic_constant_or_scale;
        if (metallic_map_id != gpu_material_empty_map_id) {
            c.metallic *=
                material_textures_2d[metallic_map_id].SampleLevel(transform_uv(uv, metallic_uv_scale_offset), 0).r;
        }

        c.ior = ior_constant_or_scale;
        if (ior_map_id != gpu_material_empty_map_id) {
            c.ior *= material_textures_2d[ior_map_id].SampleLevel(transform_uv(uv, ior_uv_scale_offset), 0).r;
        }

        c.specular_r0_mul = specular_r0_mul_constant_or_scale;
        if (specular_r0_mul_map_id != gpu_material_empty_map_id) {
            c.specular_r0_mul *= material_textures_2d[specular_r0_mul_map_id]
                                     .SampleLevel(transform_uv(uv, specular_r0_mul_uv_scale_offset), 0)
                                     .r;
        }

        c.specular_trans = specular_trans_constant_or_scale;
        if (specular_trans_map_id != gpu_material_empty_map_id) {
            c.specular_trans *= material_textures_2d[specular_trans_map_id]
                                    .SampleLevel(transform_uv(uv, specular_trans_uv_scale_offset), 0)
                                    .r;
        }

        c.diffuse_trans = diffuse_trans_constant_or_scale;
        if (diffuse_trans_map_id != gpu_material_empty_map_id) {
            c.diffuse_trans *= material_textures_2d[diffuse_trans_map_id]
                                   .SampleLevel(transform_uv(uv, diffuse_trans_uv_scale_offset), 0)
                                   .r;
        }

        c.diffuse_trans_fwd = diffuse_trans_fwd_constant_or_scale;
        if (diffuse_trans_fwd_map_id != gpu_material_empty_map_id) {
            c.diffuse_trans_fwd *= material_textures_2d[diffuse_trans_fwd_map_id]
                                       .SampleLevel(transform_uv(uv, diffuse_trans_fwd_uv_scale_offset), 0)
                                       .r;
        }

        return c;
    }

    void apply_normal_map(inout Intersection it, Sampler2D material_textures_2d[])
    {
        if (normal_map_id != gpu_material_empty_map_id) {
            float3 normal = material_textures_2d[normal_map_id].SampleLevel(it.uv, 0).xyz * 2.0 - 1.0;
            normal = lerp(float3(0, 0, 1), normal, normal_strength);
            normal = normalize(normal);

            float3 n_perturb = it.sh_frame.to_world(normal);
            float bump_deriv = -normal.x / normal.z;
            float3 t_perturb = normalize(it.sh_frame.tangent() + bump_deriv * it.sh_frame.normal());
            // re-orthogonalize (n_perturb is respected).
            float3 b_perturb = normalize(cross(n_perturb, t_perturb));
            t_perturb = normalize(cross(b_perturb, n_perturb));
            it.sh_frame = Frame(t_perturb, b_perturb, n_perturb);
        }
    }

    // TODO: should opacity map support mipmapping (sub-pixel correlation)?
    bool apply_opacity_map(float2 uv, float rnd, Sampler2D material_textures_2d[])
    {
        float opacity = opacity_constant_or_scale;
        if (opacity_map_id != gpu_material_empty_map_id) {
            opacity *= material_textures_2d[opacity_map_id].SampleLevel(uv, 0).r;
        }
        opacity = saturate(opacity);
        if (opacity <= 0.0f) {
            return false;
        } else if (opacity >= 1.0f) {
            return true;
        } else {
            return rnd < opacity;
        }
    }

    // pbrt trick: just hash ray origin and dir to get a random number.
    bool apply_opacity_map(float2 uv, float3 ro, float3 rd, Sampler2D material_textures_2d[])
    {
        float opacity = opacity_constant_or_scale;
        if (opacity_map_id != gpu_material_empty_map_id) {
            opacity *= material_textures_2d[opacity_map_id].SampleLevel(uv, 0).r;
        }
        opacity = saturate(opacity);
        if (opacity <= 0.0f) {
            return false;
        } else if (opacity >= 1.0f) {
            return true;
        } else {
            uint h = xxhash32(reinterpret<uint3>(ro));
            hash_combine_u32(h, xxhash32(reinterpret<uint3>(rd)));
            float rnd = h * 0x1p-32f;
            return rnd < opacity;
        }
    }

    float4 basecolor_uv_scale_offset;
    float4 emissive_uv_scale_offset;
    float4 roughness_uv_scale_offset;
    float4 metallic_uv_scale_offset;
    float4 ior_uv_scale_offset;
    float4 specular_r0_mul_uv_scale_offset;
    float4 specular_trans_uv_scale_offset;
    float4 diffuse_trans_uv_scale_offset;
    float4 diffuse_trans_fwd_uv_scale_offset;
    float4 normal_uv_scale_offset;
    float4 opacity_uv_scale_offset;
    // 44 * sizeof(uint32_t)

    float3 basecolor_constant_or_scale;
    uint basecolor_map_id;
    // 48 * sizeof(uint32_t)

    float3 emissive_constant_or_scale;
    uint emissive_map_id;
    // 52 * sizeof(uint32_t)

    float roughness_constant_or_scale;
    uint roughness_map_id;
    float metallic_constant_or_scale;
    uint metallic_map_id;
    float ior_constant_or_scale;
    uint ior_map_id;
    float specular_r0_mul_constant_or_scale;
    uint specular_r0_mul_map_id;
    float specular_trans_constant_or_scale;
    uint specular_trans_map_id;
    float diffuse_trans_constant_or_scale;
    uint diffuse_trans_map_id;
    float diffuse_trans_fwd_constant_or_scale;
    uint diffuse_trans_fwd_map_id;
    float normal_strength;
    uint normal_map_id;
    float opacity_constant_or_scale;
    uint opacity_map_id;
    // 70 * sizeof(uint32_t)

    // padding
    uint padding[2];
    // 72 * sizeof(uint32_t)
};

// See latest boost for more information about hash mixing/combining
// https://www.boost.org/doc/libs/1_86_0/boost/intrusive/detail/hash_mix.hpp
inline uint32_t mix_bits_u32(uint32_t x)
{
    static const uint32_t m1 = 0x21f0aaad;
    static const uint32_t m2 = 0x735a2d97;

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
uint64_t mix_bits(uint64_t v)
{
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185ULL;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44dULL;
    v ^= (v >> 33);
    return v;
}

inline void hash_combine(inout uint64_t seed, uint64_t value) { seed = mix_bits(seed + 0x9e3779b9 + value); }

inline void hash_combine_u32(inout uint32_t seed, uint32_t value) { seed = mix_bits_u32(seed + 0x9e3779b9 + value); }

struct RNG
{
    static const uint64_t PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL;
    static const uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    static const uint64_t PCG32_MULT = 0x5851f42d4c957f2dULL;

    __init()
    {
        state = PCG32_DEFAULT_STATE;
        inc = PCG32_DEFAULT_STREAM;
    }

    __init(uint64_t seq_index, uint64_t offset) { set_seq(seq_index, offset); }

    __init(uint64_t seq_index) { set_seq(seq_index); }

    [mutating]
    void set_seq(uint64_t sequence_index, uint64_t offset)
    {
        state = 0u;
        inc = (sequence_index << 1u) | 1u;
        next_u32();
        state += offset;
        next_u32();
    }

    [mutating]
    void set_seq(uint64_t sequence_index) { set_seq(sequence_index, mix_bits(sequence_index)); }

    [mutating]
    uint32_t next_u32()
    {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    [mutating]
    uint64_t next_u64()
    {
        uint64_t v0 = next_u32(), v1 = next_u32();
        return (v0 << 32) | v1;
    }

    [mutating]
    float next() { return min(fp32_before_one, next_u32() * 0x1p-32f); }

    [mutating]
    float next_f32() { return next(); }

    [mutating]
    double next_f64() { return min<double>((double)fp32_before_one, (double)next_u64() * (double)0x1p-64); }

    [mutating]
    float2 next2d() { return float2(next(), next()); }

    [mutating]
    float3 next3d() { return float3(next(), next(), next()); }

    [mutating]
    float4 next4d() { return float4(next(), next(), next(), next()); }

    [mutating]
    void advance(int64_t idelta)
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

    uint64_t state;
    uint64_t inc;
}

inline uint32_t xxhash32(uint32_t p)
{
    const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(uint2 p)
{
    static const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.y + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(uint3 p)
{
    static const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.z + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

inline uint32_t xxhash32(uint4 p)
{
    static const uint32_t PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    static const uint32_t PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint32_t h32 = p.w + PRIME32_5 + p.x * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.y * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 += p.z * PRIME32_3;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

struct PTRenderSampler
{
    __init(uint2 full_resolution, uint2 pixel, uint samples_per_pixel, uint sample_index, uint32_t seed)
    {
        rng = RNG((uint64_t)xxhash32(uint3(pixel.x, pixel.y, seed)));
        // Assume each sample uses <65536 dims...
        rng.advance(sample_index * 65536ull);
    }
    RNG rng;
}

interface IPixelFilter
{
    float eval(float2 offset);
    float2 sample(float2 u);
    float radius();
}

struct BoxPixelFilter : IPixelFilter
{
    __init(float width = 1.0f) { r = 0.5 * width; }

    float eval(float2 offset)
    {
        if (offset.x < -r || offset.x > r || offset.y < -r || offset.y > r) {
            return 0.0f;
        }
        return 1.0f;
    }

    float2 sample(float2 u) { return lerp(float2(-r), float2(r), u); }

    float radius() { return r; }

    float r;
}

struct PerspectiveCamera
{
    RayDesc spawn_ray(float2 film_pos, float tmin, float tmax)
    {
        float2 d = film_pos * 2.0 - 1.0;
        float4 target = mul(proj_to_camera, float4(d.x, d.y, 1, 1));

        RayDesc ray;
        ray.Origin = mul(camera_to_world, float4(0, 0, 0, 1)).xyz;
        ray.Direction = normalize(mul((float3x3)camera_to_world, target.xyz));
        ray.TMin = tmin;
        ray.TMax = tmax;
        return ray;
    }

    float4x4 world_to_proj;
    float4x4 camera_to_world;
    float4x4 proj_to_camera;
};

// TODO: use reflection for bindings.
struct SceneParams
{
    RaytracingAccelerationStructure scene_bvh;
    StructuredBuffer<uint> subscene_offsets;
    StructuredBuffer<GPUMeshDataAddresses, ScalarDataLayout> mesh_data_addresses;
    StructuredBuffer<uint> material_indices;
    StructuredBuffer<GPUMaterialPrincipledBSDFDataBlock> material_blocks;
    Sampler2D<float4> material_textures_2d[];
};

struct GlobalParams
{
    PerspectiveCamera camera;
    uint bounces;
    uint full_render_width;
    uint full_render_height;
    uint crop_start_x;
    uint crop_start_y;
    uint spp;
    uint spp_interval_start;
    uint spp_interval_end;
    uint rng_seed;
    uint padding[7]; // pad to vec4 boundary for now...

    RWTexture2D image;
};

/*
`ParameterBlock` is a Slang generic type for binding uniform parameters.
It is similar to `ConstantBuffer` in HLSL, and `ParameterBlock` can include not only constant parameters but also
descriptors such as Texture2D or StructuredBuffer. `ParameterBlock` is designed specifically for d3d/vulkan/metal, so
that parameters are laid out more naturally on these platforms. For Vulkan, when a ParameterBlock doesn't contain nested
parameter block fields, it always maps to a single descriptor set, with a dedicated set number and every resources is
placed into the set with binding index starting from 0.

When both ordinary data fields and resource typed fields exist in a parameter block, all ordinary data fields will be
grouped together into a uniform buffer and appear as a binding 0 of the resulting descriptor set.
*/
ParameterBlock<SceneParams> scene_params;
ParameterBlock<GlobalParams> global_params;

struct RayPayload
{
    __init()
    {
        L = float3(0.0);
        hit = 0.0;
        beta = float3(1.0);
        pdf_wi_bsdf = 0.0;
        next_ray = RayDesc();
    }

    PTRenderSampler sampler;

    float3 L;
    float hit;
    float3 beta;
    float pdf_wi_bsdf;
    RayDesc next_ray;
};

struct ShadowRayPayload
{
    float v;
};

[shader("raygeneration")]
void ray_gen_shader()
{
    uint bounces = global_params.bounces;
    uint full_render_width = global_params.full_render_width;
    uint full_render_height = global_params.full_render_height;
    uint crop_start_x = global_params.crop_start_x;
    uint crop_start_y = global_params.crop_start_y;
    uint spp = global_params.spp;
    uint spp_interval_start = global_params.spp_interval_start;
    uint spp_interval_end = global_params.spp_interval_end;
    uint rng_seed = global_params.rng_seed;

    uint2 pixel = uint2(crop_start_x + DispatchRaysIndex().x, crop_start_y + DispatchRaysIndex().y);
    uint2 full_render_size = uint2(full_render_width, full_render_height);

    BoxPixelFilter pixel_filter;
    float2 pixel_center = (float2)(pixel) + float2(0.5, 0.5);

    float3 pixel_value = float3(0.0);
    for (uint s = spp_interval_start; s < spp_interval_end; ++s) {
        RayPayload payload;
        payload.sampler = PTRenderSampler(full_render_size, pixel, spp, s, rng_seed);
        float2 pixel_sample_offset = spp <= 1 ? float2(0.0) : pixel_filter.sample(payload.sampler.rng.next2d());
        float2 pixel_sample_pos = (pixel_center + pixel_sample_offset) / (float2)(full_render_size);

        RayDesc ray = global_params.camera.spawn_ray(pixel_sample_pos, 0.0, float.maxValue);

        for (int bounce = 0; bounce < bounces + 1; ++bounce) {
            uint ray_flags = RAY_FLAG_NONE;
            if (bounce == bounces) {
                ray_flags |= RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
            }
            TraceRay(scene_params.scene_bvh, // acceleration structure
                     ray_flags,              // rayFlags
                     ~0,                     // cullMask
                     0,                      // sbtRecordOffset
                     0,                      // sbtRecordStride
                     0,                      // missIndex
                     ray,                    // ray
                     payload                 // payload
            );

            // Reaches max bounce || Miss || BSDF sampling failed
            if ((bounce == bounces) || (payload.hit == 0) || (payload.hit > 0 && payload.pdf_wi_bsdf == 0.0)) {
                break;
            }
            // Russian roulette
            float max_beta = max3(payload.beta.r, payload.beta.g, payload.beta.b);
            if (max_beta < 1.0f && bounce >= 1) {
                float q = max(0.0f, 1.0f - max_beta);
                if (payload.sampler.rng.next() < q)
                    break;
                payload.beta /= 1.0f - q;
                // thread_monitor_check(beta.allFinite());
            }
            ray = payload.next_ray;
        }

        // DEBUG. catch weird values...
        // if (!all(isfinite(payload.L)) || !all(payload.L >= 0)) {
        //     pixel_value = float3((float)samples_per_pixel, 0, (float)samples_per_pixel);
        //     break;
        // }
        pixel_value += payload.L.rgb;
    }

    //  (old * spp_interval_start + new ) / spp_interval_end
    pixel_value *= (1.0f / (float)spp_interval_end);
    if (spp_interval_start > 0) {
        float3 old_pixel_value = global_params.image[pixel].rgb;
        pixel_value += old_pixel_value * (float)spp_interval_start / (float)spp_interval_end;
    }
    global_params.image[pixel] = float4(pixel_value, 1.0);
}

[shader("miss")]
void miss_shader(inout RayPayload payload)
{
    // payload.L = float3(1, 0, 1);

    // TODO: Unidirectional (bsdf sampling) strategy
    // if (bounce == 0 && include_background)
    float3 Le = float3(0);
    payload.L += payload.beta * Le;
}

[shader("miss")]
void shadow_miss_shader(inout ShadowRayPayload payload) { payload.v = 1.0; }

[shader("closesthit")]
void closest_hit_shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    GPUMeshDataAddresses mesh_data = scene_params.mesh_data_addresses[InstanceID()];

    Intersection it = compute_intersection(mesh_data, PrimitiveIndex(), attr, WorldRayDirection(), ObjectToWorld(),
                                           WorldToObject(), false);
    // TODO: Area lights / self emission

    uint mat_id = scene_params.subscene_offsets[InstanceID()] + GeometryIndex();
    GPUMaterialPrincipledBSDFDataBlock mat = scene_params.material_blocks[mat_id];

    mat.apply_normal_map(it, scene_params.material_textures_2d);

    PrincipleBSDFClosure closure = mat.eval_closure(it.uv, scene_params.material_textures_2d);

    // Vector toward the light
    float3 wi = normalize(float3(0, 0, 1));
    float3 L = float3(1.0, 1.0, 1.0);
    float light_distance = 100000.0;

    float3 wo = -WorldRayDirection();
    float3 wo_local = it.sh_vector_to_local(wo);
    float3 wi_local = it.sh_vector_to_local(wi);
    float3 f = closure.eval(wo_local, wi_local);

    float3 Ld = float3(0.0f, 0.0f, 0.0f);
    // TODO: NEE
    if (any(f > 0.0)) {
        RayDesc ray;
        ray.Origin = it.p;
        ray.Direction = wi;
        ray.TMin = 0.001;
        ray.TMax = 10000.0;
        ShadowRayPayload shadowPayload = { 0.0 };

        uint shadow_ray_flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        // uint flags =
        //     RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
        TraceRay(scene_params.scene_bvh, // acceleration structure
                 shadow_ray_flags,       // rayFlags
                 ~0,                     // cullMask
                 0,                      // sbtRecordOffset
                 0,                      // sbtRecordStride
                 1,                      // missIndex
                 ray,                    // ray
                 shadowPayload           // payload
        );
        Ld += f * L * shadowPayload.v;
    }

    payload.hit = 1.0;
    payload.L += payload.beta * Ld;

    float u_bsdf_lobe = payload.sampler.rng.next();
    float2 u_bsdf_wi = payload.sampler.rng.next2d();
    BSDFSample bsdf_sample = closure.sample(wo_local, u_bsdf_lobe, u_bsdf_wi);
    payload.pdf_wi_bsdf = bsdf_sample.pdf_wi;
    if (bsdf_sample.success()) {
        payload.beta *= bsdf_sample.beta;
        // TODO: robust offset
        payload.next_ray.Direction = it.sh_vector_to_world(bsdf_sample.wi);
        payload.next_ray.Origin = it.p;
        payload.next_ray.TMin = 0.001;
        payload.next_ray.TMax = float.maxValue;
    }
}

[shader("anyhit")]
void any_hit_shader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // Now only for handling opacity map.

    GPUMeshDataAddresses mesh_data = scene_params.mesh_data_addresses[InstanceID()];

    float2 uv = compute_intersection_lite(mesh_data, PrimitiveIndex(), attr);

    uint mat_id = scene_params.subscene_offsets[InstanceID()] + GeometryIndex();
    GPUMaterialPrincipledBSDFDataBlock mat = scene_params.material_blocks[mat_id];
    if (!mat.apply_opacity_map(uv, WorldRayOrigin(), WorldRayDirection(), scene_params.material_textures_2d)) {
        IgnoreHit();
    }
}
