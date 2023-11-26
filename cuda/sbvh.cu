#include "../memory_util.h"
#include "ray_tri.cuh"
#include "sbvh.cuh"
#include <algorithm>

using ksc::AABB3;
using ksc::Primitive;
using ksc::SBVHBuildOption;
using ksc::SBVHNode;

struct alignas(64) SBVHBuildNode
{
    void initLeaf(uint32_t first, uint32_t n, const AABB3 &b)
    {
        primOffset = first;
        primCount = n;
        bound = b;
        children[0] = children[1] = 0;
    }
    void initInterior(uint32_t axis, uint32_t first, uint32_t n, uint32_t leftIndex, const AABB3 &leftBound,
                      uint32_t rightIndex, const AABB3 &rightBound)
    {
        primOffset = first;
        primCount = n;
        children[0] = leftIndex;
        children[1] = rightIndex;
        bound = join(leftBound, rightBound);
        splitAxis = axis;
    }

    bool isLeaf() const { return children[0] == 0 && children[1] == 0; }

    AABB3 bound;
    uint32_t children[2];
    uint32_t splitAxis;
    uint32_t primOffset;
    uint32_t primCount;

    uint8_t pad[20]; // padding.
};
static_assert(sizeof(SBVHBuildNode) == 64, "Unexpected SBVHBuildNode size.");

struct alignas(32) SBVHPrimRef
{
    SBVHPrimRef() = default;
    SBVHPrimRef(uint32_t originalIndex, const AABB3 &partialBound)
        : originalIndex(originalIndex), partialBound(partialBound)
    {}

    uint32_t originalIndex;
    // This is not always the same as the full bound
    // -- can be a partial bound due to spatial splitting.
    AABB3 partialBound;

    uint8_t pad[4];
};
static_assert(sizeof(SBVHPrimRef) == 32, "Unexpected SBVHPrimitiveInfo size.");

struct BuildContext
{
    SBVHBuildOption option;
    std::vector<Primitive> unorderedPrims;
    std::vector<SBVHBuildNode> buildNodePool;
    std::vector<SBVHPrimRef> refScratchBuffer;
    std::vector<Primitive> orderedPrims;
    uint32_t totalNodeCount = 0;
    float rootSurfaceArea = 0.0f;
};

static uint32_t recursiveBuild(BuildContext &ctx, std::vector<SBVHPrimRef> &refs);
static uint32_t flatten(uint32_t buildIndex, uint32_t &flatIndex, SBVHNode *nodes, const BuildContext &ctx);

struct SplitEstimate
{
    AABB3 leftBound, rightBound;
    float cost = 0.0f;
    uint32_t bucket = 0;
};

static SplitEstimate estimateObjectSplit(const std::vector<SBVHPrimRef> &refs, const BuildContext &ctx,
                                         const AABB3 &bounds, const AABB3 &centroidBounds, uint32_t dim,
                                         bool &skipSpatialSplit)
{
    struct ObjectSplitBucketInfo
    {
        uint32_t count = 0;
        AABB3 bounds;
    };
    uint32_t bucketCount = ctx.option.bucketCount;
    VLA(buckets, ObjectSplitBucketInfo, bucketCount);
    for (uint32_t i = 0; i < (uint32_t)refs.size(); ++i) {
        uint32_t b = (uint32_t)std::floor(bucketCount * centroidBounds.offset(refs[i].partialBound.center(), dim));
        b = std::min(b, bucketCount - 1);
        ++buckets[b].count;
        buckets[b].bounds.expand(refs[i].partialBound);
    }
    // Compute costs for splitting after each bucket.
    // Can optimize the double loop.
    VLA(cost, float, bucketCount - 1);
    VLA(leftBound, AABB3, bucketCount - 1);
    VLA(rightBound, AABB3, bucketCount - 1);
    for (uint32_t i = 0; i < bucketCount - 1; ++i) {
        uint32_t count0 = 0, count1 = 0;
        for (uint32_t j = 0; j <= i; ++j) {
            leftBound[i].expand(buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for (uint32_t j = i + 1; j < bucketCount; ++j) {
            rightBound[i].expand(buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] =
            ctx.option.travWeight +
            (count0 * leftBound[i].surface_area() + count1 * rightBound[i].surface_area()) / bounds.surface_area();
    }
    // Find bucket to split at that minimizes SAH metric.
    SplitEstimate est;
    est.cost = cost[0];
    est.bucket = 0;
    est.leftBound = leftBound[0];
    est.rightBound = rightBound[0];
    for (uint32_t i = 1; i < bucketCount - 1; ++i) {
        if (cost[i] < est.cost) {
            est.cost = cost[i];
            est.bucket = i;
            est.leftBound = leftBound[i];
            est.rightBound = rightBound[i];
        }
    }

    AABB3 isect = intersect(est.leftBound, est.rightBound);
    float isectSurfaceArea = isect.is_empty() ? 0.0f : isect.surface_area();
    skipSpatialSplit = isectSurfaceArea / ctx.rootSurfaceArea <= ctx.option.alpha;
    return est;
}

static SplitEstimate estimateSpatialSplit(const std::vector<SBVHPrimRef> &refs, const BuildContext &ctx,
                                          const AABB3 &bounds, uint32_t dim)
{
    struct SpatialSplitBucketInfo
    {
        uint32_t enterCount = 0;
        uint32_t exitCount = 0;
        AABB3 bounds;
    };
    uint32_t bucketCount = ctx.option.bucketCount;
    VLA(buckets, SpatialSplitBucketInfo, bucketCount);
    VLA(clipPlanes, float, bucketCount + 1);
    for (uint32_t b = 0; b <= bucketCount; ++b) {
        clipPlanes[b] = std::lerp(bounds.min[dim], bounds.max[dim], (float)(b) / (float)bucketCount);
    }
    for (uint32_t i = 0; i < (uint32_t)refs.size(); ++i) {
        uint32_t enter = bucketCount - 1;
        uint32_t exit = 0;
        for (uint32_t b = 0; b < bucketCount; ++b) {
            if (refs[i].partialBound.min[dim] > clipPlanes[b + 1])
                continue;
            if (refs[i].partialBound.max[dim] < clipPlanes[b])
                break;
            exit = std::max(exit, b);
            enter = std::min(enter, b);
            AABB3 chopped = refs[i].partialBound;
            chopped.min[dim] = std::max(chopped.min[dim], clipPlanes[b]);
            chopped.max[dim] = std::min(chopped.max[dim], clipPlanes[b + 1]);
            buckets[b].bounds.expand(chopped);
        }
        KSC_ASSERT(enter <= exit);
        ++buckets[enter].enterCount;
        ++buckets[exit].exitCount;
    }
    // Compute costs for splitting after each bucket.
    // Can optimize the double loop.
    VLA(cost, float, bucketCount - 1);
    VLA(leftBound, AABB3, bucketCount - 1);
    VLA(rightBound, AABB3, bucketCount - 1);
    for (uint32_t b = 0; b < bucketCount - 1; ++b) {
        uint32_t count0 = 0, count1 = 0;
        for (uint32_t j = 0; j <= b; ++j) {
            leftBound[b].expand(buckets[j].bounds);
            count0 += buckets[j].enterCount;
        }
        for (uint32_t j = b + 1; j < bucketCount; ++j) {
            rightBound[b].expand(buckets[j].bounds);
            count1 += buckets[j].exitCount;
        }
        cost[b] =
            ctx.option.travWeight +
            (count0 * leftBound[b].surface_area() + count1 * rightBound[b].surface_area()) / bounds.surface_area();
    }
    // Find bucket to split at that minimizes SAH metric.
    SplitEstimate est;
    est.cost = cost[0];
    est.bucket = 0;
    est.leftBound = leftBound[0];
    est.rightBound = rightBound[0];
    for (uint32_t i = 1; i < bucketCount - 1; ++i) {
        if (cost[i] < est.cost) {
            est.cost = cost[i];
            est.bucket = i;
            est.leftBound = leftBound[i];
            est.rightBound = rightBound[i];
        }
    }

    return est;
}

struct SplitResult
{
    AABB3 leftBound, rightBound;
    uint32_t mid = 0;
};

static SplitResult doObjectSplit(std::vector<SBVHPrimRef> &refs, const SplitEstimate &est, const AABB3 &centroidBounds,
                                 uint32_t dim, const BuildContext &ctx)
{
    auto mid = std::partition(refs.begin(), refs.end(), [&](const SBVHPrimRef &p) {
        uint32_t b = (uint32_t)std::floor(ctx.option.bucketCount * centroidBounds.offset(p.partialBound.center(), dim));
        b = std::min(b, ctx.option.bucketCount - 1);
        KSC_ASSERT(b < ctx.option.bucketCount);
        return b <= est.bucket;
    });
    SplitResult res;
    res.leftBound = est.leftBound;
    res.rightBound = est.rightBound;
    res.mid = (uint32_t)std::distance(refs.begin(), mid);
    return res;
}

static SplitResult doSpatialSplit(std::vector<SBVHPrimRef> &refs, const SplitEstimate &est, const AABB3 &bounds,
                                  uint32_t dim, BuildContext &ctx)
{
    float clipPlane =
        std::lerp(bounds.min[dim], bounds.max[dim], (float)(est.bucket + 1) / (float)ctx.option.bucketCount);
    // left: [0, leftEnd).
    // right: [leftEnd, rightStart).
    // straddling: [rightStart, rightEnd).
    // rightEnd starts to be N, but can grow.
    AABB3 actualLeftBound, actualRightBound;
    int leftEnd = 0;
    int rightStart = (int)refs.size();
    int rightEnd = (int)refs.size();
    for (int i = leftEnd; i < rightStart; i++) {
        if (refs[i].partialBound.max[dim] <= clipPlane) {
            actualLeftBound.expand(refs[i].partialBound);
            std::swap(refs[i], refs[leftEnd++]);
        } else if (refs[i].partialBound.min[dim] >= clipPlane) {
            actualRightBound.expand(refs[i].partialBound);
            std::swap(refs[i--], refs[--rightStart]);
        }
    }

    ctx.refScratchBuffer.clear();
    ctx.refScratchBuffer.reserve(rightStart - leftEnd);

    int n1 = rightStart;
    int n2 = (int)refs.size() - leftEnd;
    float fullSplitCost = est.leftBound.surface_area() * n1 + est.rightBound.surface_area() * n2;
    while (leftEnd < rightStart) {
        // Test reference un-splitting.
        SBVHPrimRef &currRef = refs[leftEnd];
        float moveLeftCost =
            join(est.leftBound, currRef.partialBound).surface_area() * n1 + est.rightBound.surface_area() * (n2 - 1);
        float moveRightCost =
            est.leftBound.surface_area() * (n1 - 1) + join(est.rightBound, currRef.partialBound).surface_area() * n2;
        float minCost = std::min(fullSplitCost, std::min(moveLeftCost, moveRightCost));
        if (minCost == moveLeftCost) {
            actualLeftBound.expand(currRef.partialBound);
            ++leftEnd;
        } else if (minCost == moveRightCost) {
            actualRightBound.expand(currRef.partialBound);
            std::swap(currRef, refs[--rightStart]);
        } else {
            KSC_ASSERT(minCost == fullSplitCost);
            SBVHPrimRef newRef = currRef;
            newRef.partialBound.min[dim] = clipPlane;
            currRef.partialBound.max[dim] = clipPlane;

            actualLeftBound.expand(currRef.partialBound);
            actualRightBound.expand(newRef.partialBound);

            ++leftEnd;
            ctx.refScratchBuffer.push_back(newRef);
            ++rightEnd;
        }
    }
    if (ctx.refScratchBuffer.size()) {
        // or just refs.end()?
        refs.insert(refs.begin() + (rightEnd - ctx.refScratchBuffer.size()), ctx.refScratchBuffer.begin(),
                    ctx.refScratchBuffer.end());
    }
    SplitResult res;
    res.leftBound = actualLeftBound;
    res.rightBound = actualRightBound;
    res.mid = rightStart;
    KSC_ASSERT(res.mid > 0 && res.mid < (uint32_t)rightEnd);
    return res;
}

static uint32_t recursiveBuild(BuildContext &ctx, std::vector<SBVHPrimRef> &refs)
{
    uint32_t nodeIndex = ctx.totalNodeCount++;
    ctx.buildNodePool.resize(ctx.totalNodeCount);

    uint32_t refCount = (uint32_t)refs.size();
    // Compute bounds of all primitives in BVH node.
    AABB3 bounds;
    for (uint32_t i = 0; i < refCount; ++i) {
        bounds.expand(refs[i].partialBound);
    }
    if (ctx.rootSurfaceArea == 0.0f) {
        ctx.rootSurfaceArea = bounds.surface_area();
    }

    auto asLeaf = [&]() {
        uint32_t firstPrim = (uint32_t)ctx.orderedPrims.size();
        for (uint32_t i = 0; i < refCount; ++i) {
            uint32_t index = refs[i].originalIndex;
            ctx.orderedPrims.push_back(ctx.unorderedPrims[index]);
        }
        ctx.buildNodePool[nodeIndex].initLeaf(firstPrim, refCount, bounds);
        return nodeIndex;
    };

    if (refCount == 1) {
        return asLeaf();
    } else {
        // Compute bound of primitive centroids, choose split dimension.
        AABB3 centroidBounds;
        for (uint32_t i = 0; i < refCount; ++i) {
            centroidBounds.expand(refs[i].partialBound.center());
        }
        uint32_t dim = centroidBounds.largest_axis();
        // Super degeneracy...no bound at all.
        if (centroidBounds.max[dim] == centroidBounds.min[dim]) {
            return asLeaf();
        } else {
            // No split.
            float leafCost = (float)refCount;
            // Estimate SAH object split.
            bool skipSpatialSplit;
            SplitEstimate osEst, ssEst;
            osEst = estimateObjectSplit(refs, ctx, bounds, centroidBounds, dim, skipSpatialSplit);
            // Estimate SAH spatial split.
            if (!skipSpatialSplit) {
                ssEst = estimateSpatialSplit(refs, ctx, bounds, dim);
            }
            if (refCount <= ctx.option.maxPrimsInNode && leafCost <= osEst.cost &&
                (skipSpatialSplit || leafCost <= ssEst.cost)) {
                return asLeaf();
            }
            SplitResult split;
            if (skipSpatialSplit || osEst.cost <= ssEst.cost) {
                split = doObjectSplit(refs, osEst, centroidBounds, dim, ctx);
            } else {
                split = doSpatialSplit(refs, ssEst, bounds, dim, ctx);
            }

            std::vector<SBVHPrimRef> &leftChildRefs = refs;
            std::vector<SBVHPrimRef> rightChildRefs(refs.begin() + split.mid, refs.end());
            leftChildRefs.resize(split.mid);
            uint32_t orderedPrimStart = (uint32_t)ctx.orderedPrims.size();
            uint32_t leftChildIndex = recursiveBuild(ctx, leftChildRefs);
            uint32_t rightChildIndex = recursiveBuild(ctx, rightChildRefs);
            uint32_t orderedPrimEnd = (uint32_t)ctx.orderedPrims.size();
            ctx.buildNodePool[nodeIndex].initInterior(dim, orderedPrimStart, orderedPrimEnd - orderedPrimStart,
                                                      leftChildIndex, split.leftBound, rightChildIndex,
                                                      split.rightBound);
            return nodeIndex;
        }
    }
}

static uint32_t flatten(uint32_t buildIndex, uint32_t &flatIndex, SBVHNode *nodes, const BuildContext &ctx)
{
    SBVHNode &node = nodes[flatIndex];
    const SBVHBuildNode &buildNode = ctx.buildNodePool[buildIndex];
    node.bound = buildNode.bound;
    node.primOffset = buildNode.primOffset;
    uint32_t thisFlatIndex = flatIndex++;
    if (!buildNode.isLeaf()) { // If interior node.
        // Create interior flattened BVH node
        // linearNode->primCount = 0;
        node.splitAxis = buildNode.splitAxis;
        flatten(buildNode.children[0], flatIndex, nodes, ctx);
        node.rightChildOffset = flatten(buildNode.children[1], flatIndex, nodes, ctx);
        KSC_ASSERT(node.rightChildOffset < (uint32_t)ctx.buildNodePool.size());
    } else {
        node.rightChildOffset = 0;
    }
    return thisFlatIndex;
}

namespace ksc
{

SBVH::SBVH(const SBVHBuildOption &option, span<const vec3> vertices, span<const uint32_t> indices)
    : vertices(vertices), indices(indices)
{
    build(option);
}

void SBVH::build(const SBVHBuildOption &option)
{
    KSC_ASSERT(indices.size % 3 == 0);
    uint32_t initPrimCount = tri_count();

    BuildContext ctx;
    ctx.option = option;
    ctx.option.maxPrimsInNode = std::min(ctx.option.maxPrimsInNode, 8u);
    ctx.option.bucketCount = std::min(ctx.option.bucketCount, 16u);
    ctx.option.travWeight = clamp(ctx.option.travWeight, 0.0f, 10.0f);
    ctx.option.alpha = saturate(ctx.option.alpha);

    ctx.unorderedPrims.reserve(initPrimCount);
    for (uint32_t shapeIndex = 0; shapeIndex < initPrimCount; ++shapeIndex) {
        ctx.unorderedPrims.push_back({shapeIndex});
    }

    std::vector<SBVHPrimRef> refs(ctx.unorderedPrims.size());
    for (uint32_t i = 0; i < (uint32_t)ctx.unorderedPrims.size(); ++i) {
        uint32_t shapeIndex = ctx.unorderedPrims[i].shapeIndex;
        AABB3 bound = tri_bound(shapeIndex);
        refs[i] = SBVHPrimRef(i, bound);
    }

    ctx.orderedPrims.reserve(ctx.unorderedPrims.size());
    ctx.totalNodeCount = 0;
    ctx.buildNodePool.reserve(std::min((uint32_t)1024, 2 * initPrimCount - 1));

    recursiveBuild(ctx, refs);
    nodes = CudaManagedArray<SBVHNode>(ctx.totalNodeCount);
    uint32_t flatIndex = 0;
    flatten(0, flatIndex, nodes.ptr.get(), ctx);
    KSC_ASSERT(ctx.totalNodeCount == flatIndex);

    primitives = CudaManagedArray<Primitive>(ctx.orderedPrims.size());
    std::memcpy(primitives.ptr.get(), ctx.orderedPrims.data(), sizeof(Primitive) * primitives.size);
}

void SBVH::printStats() const
{
    AABB3 b = bound();
    printf("----------------------------------------\n");
    printf("SBVH stats:\n");
    printf("bound: [%.4f, %.4f, %.4f] -> [%.4f, %.4f, %.4f]\n", b.min.x, b.min.y, b.min.x, b.max.x, b.max.y, b.max.x);
    printf("node count: %llu\n", nodes.size);
    printf("----------------------------------------\n");
}

struct TraversalStackEntry
{
    uint32_t nodeIndex = 0;
    uint32_t primCount = 0;
};

struct TraversalStack
{
    static constexpr uint8_t maxStackDepth = 64;
    TraversalStackEntry entries[maxStackDepth];
    uint8_t stackSize = 0;

    CUDA_HOST_DEVICE
    void push(uint32_t nodeIndex, uint32_t primCount)
    {
        KSC_ASSERT(stackSize < maxStackDepth);
        uint32_t index = stackSize++;
        entries[index].nodeIndex = nodeIndex;
        entries[index].primCount = primCount;
    }

    CUDA_HOST_DEVICE
    TraversalStackEntry pop()
    {
        KSC_ASSERT(stackSize > 0);
        uint32_t index = --stackSize;
        return entries[index];
    }

    CUDA_HOST_DEVICE
    bool empty() const { return stackSize == 0; }
};

struct IsectQuery
{
    CUDA_HOST_DEVICE
    IsectQuery(const Ray &ray, const SBVHIsectOption &option)
        : ray(ray), rb(RayBoundHelper(ray)), rt(RayTriHelper(ray)), option(option)
    {}

    CUDA_HOST_DEVICE
    bool intersect(const AABB3 &bound) const { return isect_ray_aabb(ray, bound, rb); }
    CUDA_HOST_DEVICE
    bool intersect(const array<vec3, 3> &tri, RayTriIsect *isect = nullptr) const
    {
        float check = dot(ray.dir, cross(tri[1] - tri[0], tri[2] - tri[1]));
        if (check >= 0.0f && option.culling == SBVHCulling::CullBackface) {
            return false;
        } else if (check <= 0.0f && option.culling == SBVHCulling::CullFrontFace) {
            return false;
        }
        return isect_ray_tri(ray, rt, tri[0], tri[1], tri[2], isect);
    }

    Ray ray;
    RayBoundHelper rb;
    RayTriHelper rt;
    const SBVHIsectOption &option;
};

SBVHIsectRecord SBVH::intersect(const Ray &ray, const SBVHIsectOption &option) const
{
    SBVHIsectRecord record;

    IsectQuery query(ray, option);
    TraversalStack stack;
    TraversalStackEntry curr = {0, (uint32_t)primitives.size};
    while (true) {
        const SBVHNode *node = &nodes[curr.nodeIndex];
        if (!query.intersect(node->bound)) {
            if (stack.empty())
                break;
            curr = stack.pop();
            continue;
        }
        if (node->is_leaf()) {
            for (uint32_t i = 0; i < curr.primCount; ++i) {
                const Primitive &prim = primitives[i + node->primOffset];
                RayTriIsect isect;
                if (query.intersect(tri_verts(prim.shapeIndex), &isect) && isect.tHit < query.ray.tmax) {
                    query.ray.tmax = isect.tHit;
                    record.coord = isect.coord;
                    record.thit = isect.tHit;
                    record.tri_idx = prim.shapeIndex;
                }
            }
            if (stack.empty())
                break;
            curr = stack.pop();
        } else {
            // const SBVHNode *left = &nodes[curr.nodeIndex + 1];
            const SBVHNode *right = &nodes[node->rightChildOffset];
            KSC_ASSERT(right->primOffset > node->primOffset);
            uint32_t leftPrimCount = right->primOffset - node->primOffset;
            KSC_ASSERT(leftPrimCount < curr.primCount);
            uint32_t rightPrimCount = curr.primCount - leftPrimCount;
            KSC_ASSERT(leftPrimCount + rightPrimCount == curr.primCount);
            if (query.rb.dir_is_neg[node->splitAxis]) {
                stack.push(curr.nodeIndex + 1, leftPrimCount);
                curr = {node->rightChildOffset, rightPrimCount};
            } else {
                stack.push(node->rightChildOffset, rightPrimCount);
                curr = {curr.nodeIndex + 1, leftPrimCount};
            }
        }
    }

    return record;
}

bool SBVH::intersectBool(const Ray &ray, const SBVHIsectOption &option) const
{
    IsectQuery query(ray, option);
    TraversalStack stack;
    TraversalStackEntry curr = {0, (uint32_t)primitives.size};
    while (true) {
        const SBVHNode *node = &nodes[curr.nodeIndex];
        if (!query.intersect(node->bound)) {
            if (stack.empty())
                break;
            curr = stack.pop();
            continue;
        }
        if (node->is_leaf()) {
            for (uint32_t i = 0; i < curr.primCount; ++i) {
                const Primitive &prim = primitives[i + node->primOffset];
                if (query.intersect(tri_verts(prim.shapeIndex))) {
                    return true;
                }
            }
            if (stack.empty())
                break;
            curr = stack.pop();
        } else {
            // const SBVHNode *left = &nodes[curr.nodeIndex + 1];
            const SBVHNode *right = &nodes[node->rightChildOffset];
            KSC_ASSERT(right->primOffset > node->primOffset);
            uint32_t leftPrimCount = right->primOffset - node->primOffset;
            KSC_ASSERT(leftPrimCount < curr.primCount);
            uint32_t rightPrimCount = curr.primCount - leftPrimCount;
            if (query.rb.dir_is_neg[node->splitAxis]) {
                stack.push(curr.nodeIndex + 1, leftPrimCount);
                curr = {node->rightChildOffset, rightPrimCount};
            } else {
                stack.push(node->rightChildOffset, rightPrimCount);
                curr = {curr.nodeIndex + 1, leftPrimCount};
            }
        }
    }

    return false;
}

} // namespace ksc