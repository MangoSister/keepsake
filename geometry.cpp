#include "geometry.h"
#include "embree_util.h"

Geometry::~Geometry()
{
    if (rtcgeom) {
        rtcReleaseGeometry(rtcgeom);
        rtcgeom = nullptr;
    }
}

void MeshData::transform(const Transform &t)
{
    for (int i = 0; i < (int)vertices.size() - 1; i += 3) {
        vec3 v(vertices[i + 0], vertices[i + 1], vertices[i + 2]);
        v = t.point(v);
        vertices[i + 0] = v.x();
        vertices[i + 1] = v.y();
        vertices[i + 2] = v.z();
    }
}

void MeshGeometry::create_rtc_geom(const EmbreeDevice &device)
{
    rtcgeom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    ASSERT((data->vertices.size() - 1) % 3 == 0);
    ASSERT(data->indices.size() % 3 == 0);
    rtcSetSharedGeometryBuffer(rtcgeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, data->vertices.data(), 0,
                               sizeof(float[3]), (data->vertices.size() - 1) / 3);
    rtcSetSharedGeometryBuffer(rtcgeom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, data->indices.data(), 0,
                               sizeof(uint32_t[3]), data->indices.size() / 3);
    if (data->has_texcoord()) {
        ASSERT((data->texcoords.size() - 2) % 2 == 0);
        rtcSetGeometryVertexAttributeCount(rtcgeom, 1);
        rtcSetSharedGeometryBuffer(rtcgeom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT2,
                                   data->texcoords.data(), 0, sizeof(float[2]), (data->texcoords.size() - 2) / 2);
    }

    rtcSetGeometryUserData(rtcgeom, (void *)this);

    rtcCommitGeometry(rtcgeom);
}

Intersection MeshGeometry::compute_intersection(const RTCRayHit &rayhit) const
{
    Intersection it;
    it.thit = rayhit.ray.tfar;

    vec3 ray_dir = vec3(rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z).normalized();
    vec3 wo = -ray_dir;
    vec3 ng = vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z).normalized();

    it.uv[0] = rayhit.hit.u;
    it.uv[1] = rayhit.hit.v;

    // All input buffers and output arrays must be padded to 16 bytes,
    // as the implementation uses 16-byte SSE instructions to read and write into these buffers.

    // Numerically, calculating intersection point based on surface parameterization
    // is much better than based on ray equation.
    vec4 P;
    vec4 dPdu, dPdv;
    rtcInterpolate1(rtcgeom, rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v, RTC_BUFFER_TYPE_VERTEX, 0, P.data(),
                    dPdu.data(), dPdv.data(), 3);
    it.p = P.head(3);
    it.dpdu = dPdu.head(3);
    it.dpdv = dPdv.head(3);

    // TODO: refactor hard-coded buffer slot.
    if (data->has_texcoord()) {
        vec4 tc;
        rtcInterpolate0(rtcgeom, rayhit.hit.primID, rayhit.hit.u, rayhit.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                        tc.data(), 2);
        it.uv = tc.head(2);

        // Need to manually re-calculate dpdu/dpdv due to based on supplied texture coordinates.
        vec3 p[3];
        vec2 uv[3];
        for (int v = 0; v < 3; ++v) {
            int idx = data->indices[3 * rayhit.hit.primID + v];
            p[v] = data->get_pos(idx);
            uv[v] = data->get_texcoord(idx);
        }
        vec2 duv02 = uv[0] - uv[2];
        vec2 duv12 = uv[1] - uv[2];
        vec3 dp02 = p[0] - p[2];
        vec3 dp12 = p[1] - p[2];
        float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
        bool uv_degenerate = std::abs(determinant) < 1e-8f;
        if (!uv_degenerate) {
            float inv_det = 1.0f / determinant;
            it.dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * inv_det;
            it.dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * inv_det;
        }
    }

    // re-normalize for geometry frame (ng is respected).
    vec3 b = ng.cross(it.dpdu).normalized();
    vec3 t = b.cross(ng).normalized();

    if (data->twosided) {
        if (wo.dot(ng) < 0.0f) {
            ng = -ng;
            it.dpdu = -it.dpdu;
            t = -t;
        }
    }

    it.frame = Frame(t, b, ng);
    // TODO: shading frame from vertex normals.
    it.sh_frame = it.frame;
    return it;
}

void SphereGeometry::create_rtc_geom(const EmbreeDevice &device)
{
    rtcgeom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    // The vertex buffer stores each control vertex in the form of a single precision position and radius stored in
    // (x, y, z, r)order in memory(RTC_FORMAT_FLOAT4 format).
    // The number of vertices is inferred from the size of this buffer.
    rtcSetSharedGeometryBuffer(rtcgeom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, data.data(), 0, sizeof(float[4]),
                               data.size());
    rtcSetGeometryUserData(rtcgeom, (void *)this);

    rtcCommitGeometry(rtcgeom);
}

Intersection SphereGeometry::compute_intersection(const RTCRayHit &rayhit) const
{
    Intersection it;
    it.thit = rayhit.ray.tfar;
    vec3 n = vec3(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z).normalized();
    // snap
    const vec4 &prim = data[rayhit.hit.primID];
    it.p = prim.head(3) + prim[3] * n;

    // NOTE: Embree doesn't calculate uv/partials for sphere. Need to do it manually.
    float phi, theta;
    // singularity at north/south pole
    if (n.x() == 0 && n.y() == 0)
        n.x() = 1e-5f;
    to_spherical(n, phi, theta);
    it.uv[0] = phi / two_pi;
    it.uv[1] = 1.0f - theta * inv_pi;
    it.dpdu = vec3(-two_pi * n.y(), two_pi * n.x(), 0.0f);
    it.dpdv = -pi * vec3(n.z() * std::cos(phi), n.z() * std::sin(phi), -std::sin(theta));
    // for sphere, dpdu and dpdv are orthogonal already.
    vec3 t = it.dpdu.normalized();
    vec3 b = it.dpdv.normalized();
    it.frame = Frame(t, b);
    it.sh_frame = it.frame;

    return it;
}
