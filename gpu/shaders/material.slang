implementing ks;

public struct BSDFSample
{
    public __init()
    {
        beta = float3(0.0);
        pdf_wi = 0.0;
        wi = float3(0.0);
    }
    public bool success() { return pdf_wi > 0.0; }

    public float3 beta;
    public float pdf_wi;
    public float3 wi;
};

public interface IBSDFClosure
{
    // return cosine-weighted bsdf: f*cos(theta_i)
    float3 eval(float3 wo, float3 wi);
    BSDFSample sample(float3 wo, float u_lobe, float2 u_wi);
    float pdf(float3 wo, float3 wi);
}

public interface IMaterial
{
    IBSDFClosure eval_closure(float2 uv, Sampler2D material_textures_2d[]);
}

public interface INormalMap
{
    void apply_normal_map(inout Intersection it, Sampler2D material_textures_2d[]);
}

public interface IOpacityMap
{
    // TODO: should opacity map support mipmapping (sub-pixel correlation)?
    bool apply_opacity_map(float2 uv, float rnd, Sampler2D material_textures_2d[]);
    // pbrt trick: just hash ray origin and dir to get a random number.
    bool apply_opacity_map(float2 uv, float3 ro, float3 rd, Sampler2D material_textures_2d[]);
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

// TODO: In current implementation many terms can be cancelled/collapsed/shared/approximated.
// maybe we should consider performance more seriously now on GPU...

public struct PrincipleBSDFClosure : IBSDFClosure
{
    static const float diffuse_trans_strategy_switch = 0.05f;

    public float3 eval(float3 wo, float3 wi)
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

    public BSDFSample sample(float3 wo, float u_lobe, float2 u_wi)
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

    public float pdf(float3 wo, float3 wi)
    {
        if (wo.z == 0.0f || wi.z == 0.0f) {
            return 0.0f;
        }

        float4 weights = lobe_sample_weights(wo);
        // 0: diffuse
        // 1: diffuse transmission
        // 2: metallic specular
        // 3: dielectric specular
        if (is_zero(weights)) {
            return 0.0f;
        }

        float4 pdf_lobe = float4(0.0);
        if (wo.z > 0.0f && wi.z > 0.0f) {
            pdf_lobe[0] = pdf_diffuse(wo, wi);
            pdf_lobe[2] = pdf_metallic_specular(wo, wi);
        }
        if (wo.z > 0.0f && wi.z < 0.0f) {
            pdf_lobe[1] = pdf_diffuse_transmission(wo, wi);
        }
        pdf_lobe[3] = pdf_dielectric_specular(wo, wi);

        return dot(pdf_lobe, weights);
    }

    public float3 basecolor;
    public float roughness;
    public float ax;
    public float ay;
    public float metallic;
    public float ior;
    public float specular_r0_mul;
    public float specular_trans;
    public float diffuse_trans;
    public float diffuse_trans_fwd;
};

public struct GPUMaterialPrincipledBSDFDataBlock : IMaterial, INormalMap, IOpacityMap
{
    static float2 transform_uv(float2 uv, float4 scale_offset) { return uv * scale_offset.xy + scale_offset.zw; }

    public IBSDFClosure eval_closure(float2 uv, Sampler2D material_textures_2d[])
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

    public void apply_normal_map(inout Intersection it, Sampler2D material_textures_2d[])
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
    public bool apply_opacity_map(float2 uv, float rnd, Sampler2D material_textures_2d[])
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
    public bool apply_opacity_map(float2 uv, float3 ro, float3 rd, Sampler2D material_textures_2d[])
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
            uint h = xxhash32_3(reinterpret<uint3>(ro));
            hash_combine_u32(h, xxhash32_3(reinterpret<uint3>(rd)));
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