#pragma once
#include "aabb.cuh"
#include "memory.cuh"
#include "vecmath.cuh"

namespace ksc
{

struct SBVHBuildOption
{
    uint32_t maxPrimsInNode = 4;
    uint32_t bucketCount = 12;
    float travWeight = 1.0f;
    // Control spatial splitting.
    float alpha = 1e-5f;
};

struct alignas(32) SBVHNode
{
    AABB3 bound;
    uint32_t splitAxis : 2;         // Interior only
    uint32_t rightChildOffset : 30; // Interior only
    uint32_t primOffset;

    CUDA_HOST_DEVICE
    bool is_leaf() const { return rightChildOffset == 0; }
};
static_assert(sizeof(SBVHNode) == 32, "Unexpected SBVHNode size.");

enum class SBVHCulling : uint8_t
{
    CullBackface,
    CullFrontFace,
    None,
};

struct SBVHIsectOption
{
    SBVHCulling culling;
};

struct SBVHIsectRecord
{
    CUDA_HOST_DEVICE
    bool valid() const { return thit >= 0.0f; }

    vec3 coord = vec3(0.0f);
    float thit = -inf;
    uint32_t tri_idx = uint32_t(~0);
};

struct Primitive
{
    uint32_t shapeIndex;
};

struct SBVH
{
    SBVH() = default;
    SBVH(const SBVHBuildOption &option, span<const vec3> vertices, span<const uint32_t> indices);
    void printStats() const;
    CUDA_HOST_DEVICE
    SBVHIsectRecord intersect(const Ray &ray, const SBVHIsectOption &option) const;
    CUDA_HOST_DEVICE
    bool intersectBool(const Ray &ray, const SBVHIsectOption &option) const;
    CUDA_HOST_DEVICE
    AABB3 bound() const { return nodes[0].bound; }

    CUDA_HOST_DEVICE
    uint32_t vertex_count() const { return (uint32_t)vertices.size; };
    CUDA_HOST_DEVICE
    uint32_t tri_count() const { return (uint32_t)indices.size / 3; };
    CUDA_HOST_DEVICE
    array<vec3, 3> tri_verts(size_t tri_idx) const
    {
        return {vertices[indices[tri_idx * 3 + 0]], vertices[indices[tri_idx * 3 + 1]],
                vertices[indices[tri_idx * 3 + 2]]};
    }
    CUDA_HOST_DEVICE
    AABB3 tri_bound(size_t tri_idx) const
    {
        AABB3 b;
        b.expand(vertices[indices[tri_idx * 3 + 0]]);
        b.expand(vertices[indices[tri_idx * 3 + 1]]);
        b.expand(vertices[indices[tri_idx * 3 + 2]]);
        return b;
    }

    CudaManagedArray<vec3> vertices;
    CudaManagedArray<uint32_t> indices;
    CudaManagedArray<Primitive> primitives;
    CudaManagedArray<SBVHNode> nodes;
};

} // namespace ksc