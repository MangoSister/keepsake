#pragma once

// Parallel KD-tree building based on
// Kun Zhou, Qiming Hou, Rui Wang, Baining Guo:
// Real-time KD-tree construction on graphics hardware. ACM Trans.Graph.27(5) : 126(2008)

// Other relevant papers:
// Choi, Byn, et al. "Parallel SAH kD tree construction." HPG 2010.
// Wu, Zhefeng, Fukai Zhao, and Xinguo Liu. "SAH KD-tree construction on GPU." HPG 2011.

#include "aabb.cuh"
#include "basic.cuh"
#include "memory_low_level.h"
#include "vecmath.cuh"
#include <thrust/device_vector.h>

namespace ksc
{

struct SplitPlane
{
    uint32_t axis;
    float pos;
};

enum LargeNodeChildType : uint32_t
{
    Large,
    Small,
};

struct LargeNodeChildRef
{
    LargeNodeChildType type : 1;
    uint32_t index : 31;
};

struct LargeNodeChildInfo
{
    SplitPlane split;
    LargeNodeChildRef children[2];
};

struct LargeNodeArray
{
    thrust::device_vector<uint32_t> prim_ids;             // per-prim but sorted based on nodes. can have duplicates
    thrust::device_vector<uint32_t> node_prim_count_psum; // per-node
    thrust::device_vector<AABB3> node_loose_bounds;       // per-node

    thrust::device_vector<uint32_t> node_chunk_count_psum;     // per-node
    thrust::device_vector<AABB3> node_tight_bounds;            // per-node
    thrust::device_vector<LargeNodeChildInfo> node_child_info; // per-node
    thrust::device_vector<AABB3> non_empty_bounds;             // per-node
    thrust::device_vector<uint8_t> bad_flags;                  // per-node

    thrust::device_vector<AABB3> chunk_bounds;         // per-chunk
    thrust::device_vector<uint32_t> chunk_to_node_map; // per-chunk

    //
    thrust::device_vector<uint32_t> subtree_sizes;
};

struct SmallNodeChildInfo
{
    SplitPlane split;
    uint32_t children[2];
};

struct SAHSplitCandidate
{
    uint64_t left_mask;
    uint64_t right_mask;
    uint32_t split_axis;
    float split_pos;
};

struct SmallRootArray
{
    thrust::device_vector<uint32_t> prim_ids;             // per-prim but sorted based on nodes. can have duplicates
    thrust::device_vector<uint32_t> node_prim_count_psum; // per-node
    thrust::device_vector<AABB3> node_loose_bounds;
    thrust::device_vector<uint8_t> bad_flags;
    thrust::device_vector<uint8_t> depths;
    thrust::device_vector<SAHSplitCandidate> sah_split_candidates; // per-node
};

struct SmallNodeArray
{
    // all per-node
    thrust::device_vector<AABB3> node_loose_bounds;
    thrust::device_vector<uint32_t> small_root_ids;
    thrust::device_vector<uint64_t> prim_masks;
    //
    thrust::device_vector<uint32_t> sah_splits;
    thrust::device_vector<uint32_t> child_offsets;
    //
    thrust::device_vector<uint32_t> subtree_sizes;
};

// TODO: by default thrust is blocking. Check: thrust::cuda::par_nosync or other ways to control thrust
// synchronization https://github.com/NVIDIA/thrust/pull/1568

struct CompactKdTreeNode
{
    CUDA_HOST_DEVICE void init_leaf(uint32_t n_prims)
    {
        //
        flags = 3 | (n_prims << 2);
    }

    CUDA_HOST_DEVICE void init_interior(uint32_t axis, uint32_t above_child, float s)
    {
        split = s;
        flags = axis | (above_child << 2);
    }

    CUDA_HOST_DEVICE float split_pos() const { return split; }
    CUDA_HOST_DEVICE uint32_t n_primitives() const { return flags >> 2; }
    CUDA_HOST_DEVICE uint32_t split_axis() const { return flags & 3; }
    CUDA_HOST_DEVICE bool is_leaf() const { return (flags & 3) == 3; }
    CUDA_HOST_DEVICE uint32_t above_child() const { return flags >> 2; }

    union {
        float split;         // Interior
        uint32_t first_prim; // Leaf
    };
    uint32_t flags;
};
static_assert(sizeof(CompactKdTreeNode) == 8);
static_assert(alignof(CompactKdTreeNode) == 4);
CONSTEXPR_VAL uint32_t node_header_size_u32 = sizeof(CompactKdTreeNode) / sizeof(uint32_t);

struct ParallelKdTreeBuildStats
{
#ifdef CPP_CODE_ONLY
    std::string to_string() const;
#endif
    size_t compact_strorage_bytes;
    uint32_t max_depth;
    uint32_t upper_max_depth;
    uint32_t lower_max_depth;
    uint32_t n_nodes;
    uint32_t n_small_roots;
    uint32_t n_leaves;
    uint32_t n_prim_refs;
};

struct ParallelKdTreeBuildInput
{
    uint32_t num_prims;
    CudaShareableLowLevelMemory prim_bounds_storage;

    uint32_t max_leaf_prims = 4;
    // Intersect cost is fixed to 1.
    float traversal_cost = 0.2f;
    ParallelKdTreeBuildStats *stats = nullptr;
};

struct ParallelKdTree
{
    void build(const ParallelKdTreeBuildInput &input);
    LargeNodeArray init_build(const ParallelKdTreeBuildInput &input);
    LargeNodeArray large_node_step(const ParallelKdTreeBuildInput &input, LargeNodeArray &large_nodes, uint8_t depth,
                                   SmallRootArray &small_roots);
    void prepare_small_roots(const ParallelKdTreeBuildInput &input, SmallRootArray &small_roots);
    SmallNodeArray small_node_step(const ParallelKdTreeBuildInput &input, SmallNodeArray &small_nodes,
                                   const SmallRootArray &small_roots, uint8_t depth,
                                   thrust::device_ptr<uint32_t> max_depth);

    void compact(const ParallelKdTreeBuildInput &input, std::vector<LargeNodeArray> &upper_tree,
                 const SmallRootArray &small_roots, std::vector<SmallNodeArray> &lower_tree,
                 thrust::device_ptr<uint32_t> n_leaves, thrust::device_ptr<uint32_t> prim_ref_storage);

    AABB3 get_total_bound() const; // Blocking.

    // TODO: interop memory
    // thrust::device_vector<uint32_t> nodes_storage;
    CudaShareableLowLevelMemory total_bound;
    CudaShareableLowLevelMemory nodes_storage;
};

} // namespace ksc