implementing ks;

// TODO: use reflection for bindings.
public struct SceneParams
{
    public RaytracingAccelerationStructure scene_bvh;
    public StructuredBuffer<uint> subscene_offsets;
    public StructuredBuffer<GPUMeshDataAddresses, ScalarDataLayout> mesh_data_addresses;
    public StructuredBuffer<uint> material_indices;
    public StructuredBuffer<GPUMaterialPrincipledBSDFDataBlock> material_blocks;
    public Sampler2D<float4> material_textures_2d[];
};

public static const uint gpu_material_empty_map_id = uint(~0);

public struct GPUMeshDataAddresses
{
    uint64_t vertices_address;
    uint64_t texcoords_address;
    uint64_t vertex_normals_address;
    uint64_t indices_address;
};

public struct RayDiff
{
    float3 rx_origin;
    float3 rx_dir;
    float3 ry_origin;
    float3 ry_dir;
};

public struct Intersection
{
    // Attempts to alleviate the usual shading normal / normal map problems
    // by forcing the vector to stay in the same hemisphere before/after transform.
    public float3 sh_vector_to_local(float3 world)
    {
        bool hw = dot(frame.normal(), world) >= 0.0;
        float3 local = sh_frame.to_local(world);
        bool hl = local.z >= 0.0;
        if (hw != hl) {
            local.z *= -1.0;
        }
        return local;
    }

    public float3 sh_vector_to_world(float3 local)
    {
        bool hl = local.z >= 0.0;
        float3 world = sh_frame.to_world(local);
        bool hw = dot(frame.normal(), world) >= 0.0;
        if (hl != hw) {
            world -= 2.0 * dot(frame.normal(), world) * frame.normal();
        }
        return world;
    }

    public Frame frame;
    public Frame sh_frame;
    public float3 dpdu;
    public float3 dpdv;
    public float3 p;
    public float2 uv;
    public float dudx, dvdx, dudy, dvdy;
};

public Intersection compute_intersection(GPUMeshDataAddresses mesh_data, uint prim_id,
                                         BuiltInTriangleIntersectionAttributes attr, float3 ray_dir,
                                         float3x4 object_to_world, float3x4 world_to_object, bool two_sided)
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

public float2 compute_intersection_lite(GPUMeshDataAddresses mesh_data, uint prim_id,
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
