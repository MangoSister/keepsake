#include "api_error.h"
#include "device_util.cuh"
#include "parallel_kd_tree.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
// #include <cooperative_groups.h>
// namespace cg = cooperative_groups;

namespace ksc
{

CONSTEXPR_VAL uint32_t CHUNK_SIZE = 256;
CONSTEXPR_VAL uint32_t LARGE_NODE_THRESHOLD = 64;
CONSTEXPR_VAL float EMPTY_SPACE_RATIO = 0.25f;

__global__ void count_node_chunks(uint32_t num_nodes, const uint32_t *__restrict__ node_prim_count_psum,
                                  uint32_t *__restrict__ node_chunk_count)
{
    uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= num_nodes) {
        return;
    }

    uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
    node_chunk_count[node_id] = (node_prim_count + (CHUNK_SIZE - 1)) / CHUNK_SIZE;
}

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

template <typename T>
struct minimum
{
    [[nodiscard]] __host__ __device__ inline T operator()(T a, T b) const { return min(a, b); }
};

template <typename T>
struct maximum
{
    [[nodiscard]] __host__ __device__ inline T operator()(T a, T b) const { return max(a, b); }
};

struct is_zero
{
    [[nodiscard]] __host__ __device__ inline bool operator()(uint32_t x) { return x == 0; }
};

struct fix_small_flag
{
    [[nodiscard]] __host__ __device__ inline uint32_t operator()(uint32_t psum, uint32_t value) const
    {
        if (!value) {
            return 0;
        } else {
            return psum;
        }
    }
};

// Grid size: (num_prims, 1, 1)
// Block size: (CHUNK_SIZE, 1, 1)
// Each chunk is one block.
__global__ void compute_chunk_bounds(uint32_t num_prims, const AABB3 *__restrict__ prim_bounds,
                                     const uint32_t *__restrict__ prim_ids, uint32_t num_nodes,
                                     const uint32_t *__restrict__ node_prim_count_psum,
                                     const uint32_t *__restrict__ node_chunk_count_psum,
                                     AABB3 *__restrict__ chunk_bounds, uint32_t *__restrict__ chunk_to_node_map)
{
    // CUDA_ASSERT(blockDim.x == CHUNK_SIZE);
    //
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t chunk_id = blockIdx.x;

    // Which node am i in? prefix sum node count + binary search
    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    uint32_t num_valid = CHUNK_SIZE;
    // Am I in the last chunk of a node?
    if (chunk_id + 1 == node_chunk_count_psum[node_id + 1]) {
        uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
        num_valid = node_prim_count - (node_prim_count / CHUNK_SIZE) * CHUNK_SIZE;
    }

    AABB3 b;
    if (gid < num_prims) {
        uint32_t prim_id = prim_ids[gid];
        b = prim_bounds[prim_id];
    }

    using BlockReduce = cub::BlockReduce<float, CHUNK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    AABB3 b_reduced;
    b_reduced.min.x = BlockReduce(temp_storage).Reduce(b.min.x, minimum<float>{}, num_valid);
    __syncthreads();
    b_reduced.min.y = BlockReduce(temp_storage).Reduce(b.min.y, minimum<float>{}, num_valid);
    __syncthreads();
    b_reduced.min.z = BlockReduce(temp_storage).Reduce(b.min.z, minimum<float>{}, num_valid);
    __syncthreads();

    b_reduced.max.x = BlockReduce(temp_storage).Reduce(b.max.x, maximum<float>{}, num_valid);
    __syncthreads();
    b_reduced.max.y = BlockReduce(temp_storage).Reduce(b.max.y, maximum<float>{}, num_valid);
    __syncthreads();
    b_reduced.max.z = BlockReduce(temp_storage).Reduce(b.max.z, maximum<float>{}, num_valid);
    __syncthreads();

    // The return value is undefined in threads other than thread0.
    if (threadIdx.x == 0) {
        chunk_bounds[chunk_id] = b_reduced;
        chunk_to_node_map[chunk_id] = node_id;
        // printf("%f\n", chunk_bounds[chunk_id].max.x);
    }
}

__global__ void split_large_nodes(uint32_t num_nodes, const AABB3 *__restrict__ node_loose_bounds,
                                  const AABB3 *__restrict__ node_tight_bounds,
                                  LargeNodeChildInfo *__restrict__ child_info,
                                  AABB3 *__restrict__ next_node_loose_bounds)
{
    uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= num_nodes) {
        return;
    }

    AABB3 loose_bound = node_loose_bounds[node_id];
    vec3 loose_ext = loose_bound.extents();
    AABB3 tight_bound = node_tight_bounds[node_id];
    vec3 tight_ext = tight_bound.extents();
    AABB3 valid_bound = loose_bound;
    for (uint32_t d = 0; d < 3; ++d) {
        if (tight_ext[d] / loose_ext[d] < 1.0f - EMPTY_SPACE_RATIO) {
            valid_bound.min[d] = tight_bound.min[d];
            valid_bound.max[d] = tight_bound.max[d];
        }
    }
    uint32_t axis = valid_bound.largest_axis();

    float split_pos = 0.5f * (valid_bound.min[axis] + valid_bound.max[axis]);
    float t_split = (split_pos - loose_bound.min[axis]) / loose_ext[axis];
    child_info[node_id].split.axis = axis;
    child_info[node_id].split.t = t_split;

    next_node_loose_bounds[node_id * 2] = loose_bound;
    next_node_loose_bounds[node_id * 2].max[axis] = split_pos;

    next_node_loose_bounds[node_id * 2 + 1] = loose_bound;
    next_node_loose_bounds[node_id * 2 + 1].min[axis] = split_pos;
}

struct join_aabb : public thrust::binary_function<AABB3, AABB3, AABB3>
{
    CUDA_HOST_DEVICE AABB3 operator()(AABB3 b0, AABB3 b1) { return join(b0, b1); }
};

__global__ void partition_prims_count(uint32_t num_prims, const AABB3 *__restrict__ prim_bounds,
                                      const uint32_t *__restrict__ prim_ids, uint32_t num_nodes,
                                      const uint32_t *__restrict__ node_chunk_count_psum,
                                      const AABB3 *__restrict__ node_loose_bounds,
                                      const LargeNodeChildInfo *__restrict__ child_info,
                                      uint32_t *__restrict__ next_node_prim_count)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_prims) {
        return;
    }
    uint32_t prim_id = prim_ids[gid];
    uint32_t chunk_id = blockIdx.x;

    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    SplitPlane split = child_info[node_id].split;
    AABB3 node_loose_bound = node_loose_bounds[node_id];
    float split_pos = _lerp(node_loose_bound.min[split.axis], node_loose_bound.max[split.axis], split.t);

    AABB3 b = prim_bounds[prim_id];
    uint32_t next_node_id_lc = node_id * 2;
    uint32_t next_node_id_rc = node_id * 2 + 1;
    if (b.max[split.axis] <= split_pos) {
        // prim_ids[gid].tag = PartitionTag::Left;
        atomicAdd(&next_node_prim_count[next_node_id_lc], 1);

    } else if (b.min[split.axis] >= split_pos) {
        // prim_ids[gid].tag = PartitionTag::Right;
        atomicAdd(&next_node_prim_count[next_node_id_rc], 1);
    } else {
        // prim_ids[gid].tag = PartitionTag::Both;
        atomicAdd(&next_node_prim_count[next_node_id_lc], 1);
        atomicAdd(&next_node_prim_count[next_node_id_rc], 1);
    }
}

__global__ void mark_small_nodes(uint32_t num_nodes_next, const uint32_t *__restrict__ next_node_prim_count,
                                 uint32_t *__restrict__ small_flags, uint32_t *__restrict__ small_flags_invert)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_nodes_next) {
        return;
    }
    if (next_node_prim_count[gid] <= LARGE_NODE_THRESHOLD) {
        small_flags[gid] = 1;
        small_flags_invert[gid] = 0;
    } else {
        small_flags[gid] = 0;
        small_flags_invert[gid] = 1;
    }
}

__global__ void store_child_refs(uint32_t num_nodes, const uint32_t *__restrict__ small_flags,
                                 const uint32_t *__restrict__ small_flags_invert,
                                 LargeNodeChildInfo *__restrict__ node_child_info, uint32_t curr_small_node_count)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_nodes) {
        return;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t c = 2 * gid + i;
        if (small_flags_invert[c] > 0) {
            node_child_info[gid].children[i].type = LargeNodeChildType::Large;
            node_child_info[gid].children[i].index = small_flags_invert[c] - 1;
        } else {
            node_child_info[gid].children[i].type = LargeNodeChildType::Small;
            node_child_info[gid].children[i].index = small_flags[c] - 1 + curr_small_node_count;
        }
    }
}

__global__ void
partition_prims_assign(uint32_t num_prims, const AABB3 *__restrict__ prim_bounds, const uint32_t *__restrict__ prim_ids,
                       uint32_t num_nodes, const uint32_t *__restrict__ node_chunk_count_psum,
                       const AABB3 *__restrict__ node_loose_bounds, const LargeNodeChildInfo *__restrict__ child_info,
                       const uint32_t *__restrict__ small_flags, const uint32_t *__restrict__ small_flags_invert,
                       uint32_t *__restrict__ next_prim_ids_small, uint32_t *__restrict__ next_prim_ids_large,
                       uint32_t *__restrict__ assign_offsets_small, uint32_t *__restrict__ assign_offsets_large)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_prims) {
        return;
    }
    uint32_t prim_id = prim_ids[gid];
    uint32_t chunk_id = blockIdx.x;

    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    SplitPlane split = child_info[node_id].split;
    AABB3 node_loose_bound = node_loose_bounds[node_id];
    float split_pos = _lerp(node_loose_bound.min[split.axis], node_loose_bound.max[split.axis], split.t);

    AABB3 b = prim_bounds[prim_id];

    bool left = false, right = false;
    if (b.max[split.axis] <= split_pos) {
        left = true;
    } else if (b.min[split.axis] >= split_pos) {
        right = true;
    } else {
        left = right = true;
    }

    if (left) {
        uint32_t lc = node_id * 2;
        if (small_flags[lc] > 0) {
            uint32_t next_small_node_id = small_flags[lc] - 1;
            uint32_t pos = atomicAdd(&assign_offsets_small[next_small_node_id], 1);
            next_prim_ids_small[pos] = prim_id;
        } else {
            uint32_t next_large_node_id = small_flags_invert[lc] - 1;
            uint32_t pos = atomicAdd(&assign_offsets_large[next_large_node_id], 1);
            next_prim_ids_large[pos] = prim_id;
        }
    }
    if (right) {
        uint32_t rc = node_id * 2 + 1;
        if (small_flags[rc] > 0) {
            uint32_t next_small_node_id = small_flags[rc] - 1;
            uint32_t pos = atomicAdd(&assign_offsets_small[next_small_node_id], 1);
            next_prim_ids_small[pos] = prim_id;
        } else {
            uint32_t next_large_node_id = small_flags_invert[rc] - 1;
            uint32_t pos = atomicAdd(&assign_offsets_large[next_large_node_id], 1);
            next_prim_ids_large[pos] = prim_id;
        }
    }
    // uint32_t next_node_id_lc = node_id * 2;
    // uint32_t next_node_id_rc = node_id * 2 + 1;
    // if (b.max[split.axis] <= split_pos) {
    //     // prim_ids[gid].tag = PartitionTag::Left;
    //     uint32_t pos = atomicAdd(&assign_offsets[next_node_id_lc], 1);
    //     next_prim_ids[pos] = prim_id;
    // } else if (b.min[split.axis] >= split_pos) {
    //     // prim_ids[gid].tag = PartitionTag::Right;
    //     uint32_t pos = atomicAdd(&assign_offsets[next_node_id_rc], 1);
    //     next_prim_ids[pos] = prim_id;
    // } else {
    //     // prim_ids[gid].tag = PartitionTag::Both;
    //     uint32_t pos1 = atomicAdd(&assign_offsets[next_node_id_lc], 1);
    //     uint32_t pos2 = atomicAdd(&assign_offsets[next_node_id_rc], 1);
    //     next_prim_ids[pos1] = prim_id;
    //     next_prim_ids[pos2] = prim_id;
    // }
}

void ParallelKdTree::build(const ParallelKdTreeBuildInput &input)
{
    std::vector<LargeNodeArray> upper_tree{init_build(input)};
    SmallNodeArray small_roots;
    // Small node stage
    while (true) {
        LargeNodeArray &curr = upper_tree.back();
        LargeNodeArray next = large_node_step(input, curr, small_roots);
        if (!next.node_loose_bounds.empty()) {
            upper_tree.push_back(std::move(next));
        } else {
            break;
        }
    }
    // Small node stage
    prepare_small_roots(input, small_roots);
}

LargeNodeArray ParallelKdTree::init_build(const ParallelKdTreeBuildInput &input)
{
    uint32_t num_prims = (uint32_t)input.bounds.size();
    LargeNodeArray large_nodes;
    large_nodes.prim_ids.resize(num_prims);
    thrust::sequence(large_nodes.prim_ids.begin(), large_nodes.prim_ids.end());

    // Allocate one more to automatically get the total count from prefix sum.
    large_nodes.node_prim_count_psum.resize(2);
    thrust::fill_n(large_nodes.node_prim_count_psum.begin(), 1, num_prims);
    thrust::exclusive_scan(large_nodes.node_prim_count_psum.begin(), large_nodes.node_prim_count_psum.end(),
                           large_nodes.node_prim_count_psum.begin()); // support in-place prefix sum.
    return large_nodes;
}

LargeNodeArray ParallelKdTree::large_node_step(const ParallelKdTreeBuildInput &input, LargeNodeArray &large_nodes,
                                               SmallNodeArray &global_small_nodes)
{
    uint32_t num_prims = (uint32_t)input.bounds.size();
    uint32_t num_nodes = large_nodes.node_prim_count_psum.size() - 1;

    large_nodes.node_chunk_count_psum.resize(num_nodes + 1);
    run_kernel_1d(count_node_chunks, 0, (cudaStream_t)(0), num_nodes, num_nodes,
                  large_nodes.node_prim_count_psum.data().get(), large_nodes.node_chunk_count_psum.data().get());
    thrust::exclusive_scan(large_nodes.node_chunk_count_psum.begin(), large_nodes.node_chunk_count_psum.end(),
                           large_nodes.node_chunk_count_psum.begin()); // support in-place prefix sum.

    uint32_t num_chunks = 0;
    thrust::copy_n(large_nodes.node_chunk_count_psum.begin() + num_nodes, 1, &num_chunks);
    large_nodes.chunk_bounds.resize(num_chunks);
    large_nodes.chunk_to_node_map.resize(num_chunks);
    run_kernel_1d<CHUNK_SIZE>(compute_chunk_bounds, 0, (cudaStream_t)(0), num_prims, num_prims,
                              input.bounds.data().get(), large_nodes.prim_ids.data().get(), num_nodes,
                              large_nodes.node_prim_count_psum.data().get(),
                              large_nodes.node_chunk_count_psum.data().get(), large_nodes.chunk_bounds.data().get(),
                              large_nodes.chunk_to_node_map.data().get());
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());

    // Perform segmented reduction on per-chunk reduction result to compute per-node tight bounding box
    large_nodes.node_tight_bounds.resize(num_nodes);
    thrust::device_vector<uint32_t> segmented_reduce_output_keys;
    segmented_reduce_output_keys.resize(large_nodes.node_tight_bounds.size());
    // https://nvidia.github.io/cccl/thrust/api/function_group__reductions_1ga12501ca04b9b21b101d6a299d899d7af.html
    thrust::reduce_by_key(large_nodes.chunk_to_node_map.begin(), large_nodes.chunk_to_node_map.end(),
                          large_nodes.chunk_bounds.begin(), segmented_reduce_output_keys.begin(),
                          large_nodes.node_tight_bounds.begin(), cuda::std::equal_to<uint32_t>{}, join_aabb{});
    if (num_nodes == 1) {
        // Initialize: same as tight bound.
        large_nodes.node_loose_bounds = large_nodes.node_tight_bounds;
    }

    large_nodes.node_child_info.resize(num_nodes);

    uint32_t num_nodes_next = num_nodes * 2;
    thrust::device_vector<AABB3> next_node_loose_bounds(num_nodes_next);

    // thrust::device_vector<SplitPlane> split_planes(num_nodes);
    run_kernel_1d(split_large_nodes, 0, (cudaStream_t)(0), num_nodes, num_nodes,
                  large_nodes.node_loose_bounds.data().get(), large_nodes.node_tight_bounds.data().get(),
                  large_nodes.node_child_info.data().get(), next_node_loose_bounds.data().get());
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());

    // Allocate one more to automatically get the total count from prefix sum.
    thrust::device_vector<uint32_t> next_node_prim_count_psum(num_nodes_next + 1, 0);

    run_kernel_1d<CHUNK_SIZE>(partition_prims_count, 0, (cudaStream_t)(0), num_prims, num_prims,
                              input.bounds.data().get(), large_nodes.prim_ids.data().get(), num_nodes,
                              large_nodes.node_chunk_count_psum.data().get(),
                              large_nodes.node_loose_bounds.data().get(), large_nodes.node_child_info.data().get(),
                              next_node_prim_count_psum.data().get());
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());

    // TODO: DEBUG next level large/small split logic below

    thrust::device_vector<uint32_t> small_flags(num_nodes_next);
    thrust::device_vector<uint32_t> small_flags_invert(num_nodes_next);
    run_kernel_1d(mark_small_nodes, 0, (cudaStream_t)(0), num_nodes_next, num_nodes_next,
                  next_node_prim_count_psum.data().get(), small_flags.data().get(), small_flags_invert.data().get());
    thrust::device_vector<uint32_t> small_flags_copy = small_flags;
    thrust::device_vector<uint32_t> small_flags_invert_copy = small_flags_invert;

    thrust::inclusive_scan(small_flags.begin(), small_flags.end(), small_flags.begin());
    thrust::inclusive_scan(small_flags_invert.begin(), small_flags_invert.end(), small_flags_invert.begin());

    uint32_t num_small_nodes_next = 0;
    thrust::copy_n(small_flags.rbegin(), 1, &num_small_nodes_next);
    uint32_t num_large_nodes_next = num_nodes_next - num_small_nodes_next;
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());

    thrust::transform(small_flags.begin(), small_flags.end(), small_flags_copy.begin(), small_flags.begin(),
                      fix_small_flag{});
    thrust::transform(small_flags_invert.begin(), small_flags_invert.end(), small_flags_invert_copy.begin(),
                      small_flags_invert.begin(), fix_small_flag{});
    // After transform, small_flag should either be 0 for large nodes, or the 1-based node indices for small nodes.
    // vice-verse for small_flags_invert

    // Record parent/child references
    uint32_t curr_small_node_count = global_small_nodes.node_loose_bounds.size();
    run_kernel_1d(store_child_refs, 0, (cudaStream_t)(0), num_nodes, num_nodes, small_flags.data().get(),
                  small_flags_invert.data().get(), large_nodes.node_child_info.data().get(), curr_small_node_count);
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());
    //{
    //    // DEBUG
    //    thrust::host_vector<LargeNodeChildInfo> debug = large_nodes.node_child_info;
    //    std::vector<LargeNodeChildInfo> v{debug.begin(), debug.end()};
    //    LargeNodeChildInfo a = v[0];
    //    // LargeNodeChildInfo b = v[1];
    //    LargeNodeChildInfo c = v.back();
    //    c = v.back();
    //}

    thrust::device_vector<uint32_t> next_node_prim_count_psum_small(num_small_nodes_next + 1);
    thrust::device_vector<uint32_t> next_node_prim_count_psum_large(num_large_nodes_next + 1);
    // Need stable partition to preserve the large/small node indices order
    thrust::stable_partition_copy(next_node_prim_count_psum.begin(), next_node_prim_count_psum.end(),
                                  small_flags_invert.begin(), //
                                  next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_large.begin(),
                                  is_zero{});

    thrust::exclusive_scan(next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_small.end(),
                           next_node_prim_count_psum_small.begin());
    thrust::exclusive_scan(next_node_prim_count_psum_large.begin(), next_node_prim_count_psum_large.end(),
                           next_node_prim_count_psum_large.begin());
    uint32_t next_num_prims_small = 0;
    thrust::copy_n(next_node_prim_count_psum_small.rbegin(), 1, &next_num_prims_small);
    uint32_t next_num_prims_large = 0;
    thrust::copy_n(next_node_prim_count_psum_large.rbegin(), 1, &next_num_prims_large);
    // Same. Need stable partition.
    thrust::device_vector<AABB3> next_node_loose_bound_small(num_small_nodes_next);
    thrust::device_vector<AABB3> next_node_loose_bound_large(num_large_nodes_next);
    thrust::stable_partition_copy(next_node_loose_bounds.begin(), next_node_loose_bounds.end(),
                                  small_flags_invert.begin(), //
                                  next_node_loose_bound_small.begin(), next_node_loose_bound_large.begin(), is_zero{});

    thrust::device_vector<uint32_t> next_prim_ids_small(next_num_prims_small);
    thrust::device_vector<uint32_t> next_prim_ids_large(next_num_prims_large);

    thrust::device_vector<uint32_t> assign_offsets_small = next_node_prim_count_psum_small;
    thrust::device_vector<uint32_t> assign_offsets_large = next_node_prim_count_psum_large;
    // Partition prim ids while accounting for large/small separation
    run_kernel_1d<CHUNK_SIZE>(partition_prims_assign, 0, (cudaStream_t)(0), num_prims, num_prims,
                              input.bounds.data().get(), large_nodes.prim_ids.data().get(), num_nodes,
                              large_nodes.node_chunk_count_psum.data().get(),
                              large_nodes.node_loose_bounds.data().get(), large_nodes.node_child_info.data().get(),
                              small_flags.data().get(), small_flags_invert.data().get(),          //
                              next_prim_ids_small.data().get(), next_prim_ids_large.data().get(), //
                              assign_offsets_small.data().get(), assign_offsets_large.data().get());

    LargeNodeArray next_large;
    next_large.prim_ids = std::move(next_prim_ids_large);
    next_large.node_prim_count_psum = std::move(next_node_prim_count_psum_large);
    next_large.prim_ids = std::move(next_prim_ids_large);

    // Append to a global small node array
    {
        uint32_t old_size = global_small_nodes.prim_ids.size();
        global_small_nodes.prim_ids.resize(old_size + next_prim_ids_small.size());
        thrust::copy(next_prim_ids_small.begin(), next_prim_ids_small.end(),
                     global_small_nodes.prim_ids.begin() + old_size);
    }
    {
        uint32_t old_size = global_small_nodes.node_loose_bounds.size();
        global_small_nodes.node_loose_bounds.resize(old_size + next_node_loose_bounds.size());
        thrust::copy(next_node_loose_bounds.begin(), next_node_loose_bounds.end(),
                     global_small_nodes.node_loose_bounds.begin() + old_size);
    }

    // Need to add a global offset and append
    uint32_t global_small_node_prim_count = 0;
    if (global_small_nodes.node_prim_count_psum.size() > 0) {
        thrust::copy_n(global_small_nodes.node_prim_count_psum.rbegin(), 1, &global_small_node_prim_count);
        thrust::transform(
            next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_small.end(),
            next_node_prim_count_psum_small.begin(),
            [global_small_node_prim_count] __device__(uint32_t x) { return x + global_small_node_prim_count; });
    }
    {
        uint32_t old_size = global_small_nodes.node_prim_count_psum.size();
        global_small_nodes.node_prim_count_psum.resize(std::max(old_size, 1u) - 1 +
                                                       next_node_prim_count_psum_small.size());
        thrust::copy(next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_small.end(),
                     global_small_nodes.node_prim_count_psum.begin() + std::max(old_size, 1u) - 1);
    }

    return next_large;
}

struct SAHSplitCandidate
{
    uint64_t left_mask;
    uint64_t right_mask;
    uint32_t split_axis;
    float split_pos;
};

// Per split candidate
__global__ void prepare_small_roots_kernel(uint32_t num_candidates, uint32_t num_nodes,
                                           const AABB3 *__restrict__ prim_bounds, const uint32_t *__restrict__ prim_ids,
                                           const uint32_t *__restrict__ node_prim_count_psum,
                                           SAHSplitCandidate *__restrict__ sah_split_candidates)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_candidates) {
        return;
    }
    // 6 candidates each prim: [xmin, xmax, ymin, ymax, zmin, zmax]
    // what is my split axis and pos?
    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return (gid / 6) >= node_prim_count_psum[i]; });
    uint32_t prim_offset = gid - 6 * (gid / 6);

    uint32_t axis = (gid % 6) / 3;
    uint32_t cand_prim_id = prim_ids[node_prim_count_psum[node_id] + prim_offset];
    AABB3 cand_bound = prim_bounds[cand_prim_id];
    float split_pos = cand_bound[gid % 2][axis];

    uint64_t left_mask = 0;
    uint64_t right_mask = 0;
    for (uint32_t i = node_prim_count_psum[node_id]; i < node_prim_count_psum[node_id + 1]; ++i) {
        // i should be < LARGE_NODE_THRESHOLD (64)
        uint32_t prim_id = prim_ids[i];
        AABB3 b = prim_bounds[prim_id];
        // We can have primitives on both sides
        if (b.min[axis] < split_pos) {
            left_mask |= (1 << i);
        }
        if (b.max[axis] > split_pos) {
            right_mask |= (1 << i);
        }
    }

    sah_split_candidates[gid].left_mask = left_mask;
    sah_split_candidates[gid].right_mask = right_mask;
    sah_split_candidates[gid].split_axis = axis;
    sah_split_candidates[gid].split_pos = split_pos;
}

void ParallelKdTree::prepare_small_roots(const ParallelKdTreeBuildInput &input, const SmallNodeArray &small_roots)
{
    uint32_t num_candidates = 0;
    thrust::copy_n(small_roots.node_prim_count_psum.rbegin(), 1, &num_candidates);
    num_candidates *= 6;
    thrust::device_vector<SAHSplitCandidate> sah_split_candidates(num_candidates);
    uint32_t num_nodes = small_roots.node_prim_count_psum.size() - 1;
    run_kernel_1d(prepare_small_roots_kernel, 0, (cudaStream_t)(0), num_candidates, num_candidates, num_nodes,
                  input.bounds.data().get(), small_roots.prim_ids.data().get(),
                  small_roots.node_prim_count_psum.data().get(), sah_split_candidates.data().get());
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaGetLastError());
    //
}

// One node per thread
__global__ void small_node_step_kernel(uint32_t num_nodes, const uint32_t *__restrict__ small_root_ids,
                                       const uint64_t *__restrict__ prim_masks,
                                       const AABB3 *__restrict__ node_loose_bounds,
                                       const uint32_t *__restrict__ small_root_node_prim_count_psum,
                                       const SAHSplitCandidate *__restrict__ sah_split_candidates)
{
    uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= num_nodes) {
        return;
    }
    uint32_t small_root = small_root_ids[node_id];
    uint64_t prim_mask = prim_masks[node_id];
    AABB3 bound = node_loose_bounds[node_id];
    float area = bound.surface_area();
    float min_sah = (float)__popcll(prim_mask);
    for (uint32_t i = 0; i < LARGE_NODE_THRESHOLD; ++i) {
        if ((prim_mask >> i) & 1) {
            uint32_t offset = (small_root_node_prim_count_psum[small_root] + i) * 6;
            for (uint32_t j = 0; j < 6; ++j) {
                const SAHSplitCandidate &s = sah_split_candidates[offset + j];
                uint64_t left = prim_mask & s.left_mask;
                uint64_t right = prim_mask & s.right_mask;
                // Count the number of bits that are set to 1 in a 64-bit integer.
                int n_left = __popcll(left);
                int n_right = __popcll(right);
                // Calculate split nodes area
                AABB3 b_left = bound;
                b_left.max[s.split_axis] = s.split_pos;
                float area_left = b_left.surface_area();
                AABB3 b_right = bound;
                b_right.min[s.split_axis] = s.split_pos;
                float area_right = b_right.surface_area();
                // Compute SAH
                // https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
                // https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/cpu/aggregates.cpp
                constexpr float cost_ts = 0.5f;
                float sah = (n_left * area_left + n_right + area_right) / area + cost_ts;
                // TODO
            }
        }
    }
}

void ParallelKdTree::small_node_step()
{
    //
}

} // namespace ksc