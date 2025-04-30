#pragma once

// Parallel KD-tree building based on
// Kun Zhou, Qiming Hou, Rui Wang, Baining Guo:
// Real-time KD-tree construction on graphics hardware. ACM Trans.Graph.27(5) : 126(2008)

// Other relevant papers:
// Choi, Byn, et al. "Parallel SAH kD tree construction." HPG 2010.
// Wu, Zhefeng, Fukai Zhao, and Xinguo Liu. "SAH KD-tree construction on GPU." HPG 2011.

#include "aabb.cuh"
#include "vecmath.cuh"
#include <thrust/device_vector.h>

namespace ksc
{

struct ParallelKdTreeBuildInput
{
    const thrust::device_vector<AABB3> &bounds;
};

struct SplitPlane
{
    uint32_t axis;
    float t;
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

    thrust::device_vector<AABB3> chunk_bounds;         // per-chunk
    thrust::device_vector<uint32_t> chunk_to_node_map; // per-chunk
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
};

// TODO: by default thrust is blocking. Check: thrust::cuda::par_nosync or other ways to control thrust
// synchronization https://github.com/NVIDIA/thrust/pull/1568

struct ParallelKdTree
{
    void build(const ParallelKdTreeBuildInput &input);
    LargeNodeArray init_build(const ParallelKdTreeBuildInput &input);
    LargeNodeArray large_node_step(const ParallelKdTreeBuildInput &input, LargeNodeArray &large_nodes,
                                   SmallRootArray &small_roots);
    void prepare_small_roots(const ParallelKdTreeBuildInput &input, SmallRootArray &small_roots);
    SmallNodeArray small_node_step(SmallNodeArray &small_nodes, const SmallRootArray &small_roots);
};

} // namespace ksc