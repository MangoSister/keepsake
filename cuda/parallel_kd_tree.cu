#include "api_error.h"
#include "device_util.cuh"
#include "parallel_kd_tree.cuh"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cub/block/block_reduce.cuh>
// #include <cooperative_groups.h>
// namespace cg = cooperative_groups;

namespace ksc
{

CONSTEXPR_VAL uint32_t CHUNK_SIZE = 256;
CONSTEXPR_VAL uint32_t LARGE_NODE_THRESHOLD = 64;

struct LargeNodeArray
{
    thrust::device_vector<uint32_t> prim_ids;

    thrust::device_vector<AABB3> node_bounds;
    thrust::device_vector<uint32_t> node_prim_count_psum;
    thrust::device_vector<uint32_t> node_chunk_count_psum;

    thrust::device_vector<AABB3> chunk_bounds;
};

template <typename Predicate>
CUDA_HOST_DEVICE inline uint32_t find_interval(uint32_t sz, const Predicate &pred)
{
    int size = (int)sz - 2, first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update _first_ and _size_
        uint32_t half = (uint32_t)size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    return (uint32_t)clamp((int)first - 1, 0, (int)sz - 2);
}

// Grid size: (num_prims, 1, 1)
// Block size: (CHUNK_SIZE, 1, 1)
// Each chunk is one block.
__global__ void compute_chunk_bounds(const AABB3 *__restrict__ prim_bounds, const uint32_t *__restrict__ prim_ids,
                                     uint32_t num_nodes, const uint32_t *__restrict__ node_prim_count_psum,
                                     const uint32_t *__restrict__ node_chunk_count_psum,
                                     AABB3 *__restrict__ chunk_bounds)
{
    // CUDA_ASSERT(blockDim.x == CHUNK_SIZE);
    //
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t prim_id = prim_ids[gid];

    // am I in the last chunk of a node?
    // which node am i in? prefix sum node count + binary search
    uint32_t node_idx = find_interval(num_nodes, [&](uint32_t i) { return blockIdx.x >= node_chunk_count_psum[i]; });
    uint32_t num_valid = CHUNK_SIZE;
    if (blockIdx.x + 1 == node_chunk_count_psum[node_idx + 1]) {
        uint32_t node_prim_count = node_prim_count_psum[node_idx + 1] - node_prim_count_psum[node_idx];
        num_valid = node_prim_count - (node_prim_count / CHUNK_SIZE) * CHUNK_SIZE;
    }

    AABB3 b = prim_bounds[prim_id];

    using BlockReduce = cub::BlockReduce<int, CHUNK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    AABB3 b_reduced;
    b_reduced.min.x = BlockReduce(temp_storage).Reduce(b.min.x, cuda::std::less<float>{}, num_valid);
    __syncthreads();
    b_reduced.min.y = BlockReduce(temp_storage).Reduce(b.min.y, cuda::std::less<float>{}, num_valid);
    __syncthreads();
    b_reduced.min.z = BlockReduce(temp_storage).Reduce(b.min.z, cuda::std::less<float>{}, num_valid);
    __syncthreads();

    b_reduced.max.x = BlockReduce(temp_storage).Reduce(b.max.x, cuda::std::greater<float>{}, num_valid);
    __syncthreads();
    b_reduced.max.y = BlockReduce(temp_storage).Reduce(b.max.y, cuda::std::greater<float>{}, num_valid);
    __syncthreads();
    b_reduced.max.z = BlockReduce(temp_storage).Reduce(b.max.z, cuda::std::greater<float>{}, num_valid);
    __syncthreads();

    // The return value is undefined in threads other than thread0.
    if (threadIdx.x == 0) {
        chunk_bounds[blockIdx.x] = b_reduced;
    }
}

//__global__ void compute_chunk_bounds(const AABB3 *__restrict__ prim_bounds, const uint32_t *__restrict__ prim_ids,
//                                     uint32_t num_nodes, const uint32_t *__restrict__ node_prim_count_psum,
//                                     const uint32_t *__restrict__ node_chunk_count_psum,
//                                     AABB3 *__restrict__ chunk_bounds)

void ParallelKdTree::build(const ParallelKdTreeBuildInput &input)
{
    uint32_t num_prims = (uint32_t)input.bounds.size();
    LargeNodeArray large_nodes;
    uint32_t num_nodes = (uint32_t)large_nodes.node_bounds.size();

    run_kernel_1d<CHUNK_SIZE>(compute_chunk_bounds, 0, (cudaStream_t)(0), num_prims,
                              thrust::raw_pointer_cast(input.bounds.data()),
                              thrust::raw_pointer_cast(large_nodes.prim_ids.data()), num_nodes,
                              thrust::raw_pointer_cast(large_nodes.node_prim_count_psum.data()),
                              thrust::raw_pointer_cast(large_nodes.node_chunk_count_psum.data()),
                              thrust::raw_pointer_cast(large_nodes.chunk_bounds.data()));
}

} // namespace ksc