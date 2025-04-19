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

struct ParallelKdTreeNode
{
    int foo;
};

struct ParallelKdTreeBuildInput
{
    thrust::device_vector<AABB3> bounds;
};

struct ParallelKdTree
{
    void build(const ParallelKdTreeBuildInput &input);

    //void build_large_nodes_stage(NodeChunkArray &active, NodeChunkArray &next);

    thrust::device_vector<ParallelKdTreeNode> nodes;
    thrust::device_vector<uint32_t> prim_ids;
};

} // namespace ksc