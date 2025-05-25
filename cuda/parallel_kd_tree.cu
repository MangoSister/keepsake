#include "api_error.h"
#include "device_util.cuh"
#include "parallel_kd_tree.h"

#include <thrust/device_free.h>
#include <thrust/device_new.h>
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

// #include "../md5.h"

namespace ksc
{

CONSTEXPR_VAL uint32_t CHUNK_SIZE = 256;
CONSTEXPR_VAL uint32_t LARGE_NODE_THRESHOLD = 64;
CONSTEXPR_VAL float EMPTY_SPACE_RATIO = 0.25f;
CONSTEXPR_VAL uint8_t BAD_SPLIT_TRYS = 3;
CONSTEXPR_VAL uint8_t MAX_DEPTH = 64;

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
__global__ void compute_chunk_bounds(uint32_t num_refs_chunked, const AABB3 *__restrict__ prim_bounds,
                                     const uint32_t *__restrict__ prim_ids, uint32_t num_nodes,
                                     const uint32_t *__restrict__ node_prim_count_psum,
                                     const uint32_t *__restrict__ node_chunk_count_psum,
                                     AABB3 *__restrict__ chunk_bounds, uint32_t *__restrict__ chunk_to_node_map)
{
    // CUDA_ASSERT(blockDim.x == CHUNK_SIZE);
    //
    // uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t chunk_id = blockIdx.x;

    // Which node am i in? prefix sum node count + binary search
    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    uint32_t num_valid = CHUNK_SIZE;
    // Am I in the last chunk of a node?
    if (chunk_id + 1 == node_chunk_count_psum[node_id + 1]) {
        uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
        if (node_prim_count % CHUNK_SIZE != 0) {
            num_valid = node_prim_count - (node_prim_count / CHUNK_SIZE) * CHUNK_SIZE;
        }
    }

    AABB3 b;
    if (threadIdx.x < num_valid) {
        uint32_t ref_id =
            node_prim_count_psum[node_id] + (chunk_id - node_chunk_count_psum[node_id]) * CHUNK_SIZE + threadIdx.x;
        uint32_t prim_id = prim_ids[ref_id];
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
        // printf("%u (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)\n",            //
        //        node_id, b_reduced.min.x, b_reduced.min.y, b_reduced.min.z, //
        //        b_reduced.max.x, b_reduced.max.y, b_reduced.max.z);
        chunk_bounds[chunk_id] = b_reduced;
        chunk_to_node_map[chunk_id] = node_id;
        // if (b_reduced.is_empty()) {
        //     uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
        //     printf("WTF %u, %u, %u, %u, %u\n", chunk_id, node_chunk_count_psum[node_id],
        //            node_chunk_count_psum[node_id + 1], node_prim_count, num_valid);
        //     CUDA_ASSERT(false);
        // }
        //  printf("%f\n", chunk_bounds[chunk_id].max.x);
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

    // vec3 tight_ext = tight_bound.extents();
    // AABB3 valid_bound = loose_bound;
    AABB3 isect_bound = intersect(loose_bound, tight_bound);
    AABB3 valid_bound;
    for (uint32_t d = 0; d < 3; ++d) {
        if ((isect_bound.min[d] - loose_bound.min[d]) > EMPTY_SPACE_RATIO * loose_ext[d]) {
            valid_bound.min[d] = isect_bound.min[d];
        } else {
            valid_bound.min[d] = loose_bound.min[d];
        }
        if ((loose_bound.max[d] - isect_bound.max[d]) > EMPTY_SPACE_RATIO * loose_ext[d]) {
            valid_bound.max[d] = isect_bound.max[d];
        } else {
            valid_bound.max[d] = loose_bound.max[d];
        }
    }
    uint32_t axis = valid_bound.largest_axis();

    float split_pos = 0.5f * (valid_bound.min[axis] + valid_bound.max[axis]);
    child_info[node_id].split.axis = axis;
    child_info[node_id].split.pos = split_pos;
    // if (isnan(split_pos)) {
    //     printf("(%f, %f, %f, %f, %f, %f), (%f, %f, %f, %f, %f, %f), (%f, %f, %f, %f, %f, %f)\n", //
    //            valid_bound.min.x, valid_bound.min.y, valid_bound.min.z,                          //
    //            valid_bound.max.x, valid_bound.max.y, valid_bound.max.z,                          //
    //            loose_bound.min.x, loose_bound.min.y, loose_bound.min.z,                          //
    //            loose_bound.max.x, loose_bound.max.y, loose_bound.max.z,                          //
    //            tight_bound.min.x, tight_bound.min.y, tight_bound.min.z,                          //
    //            tight_bound.max.x, tight_bound.max.y, tight_bound.max.z);
    //     CUDA_ASSERT(false);
    // }

    // printf(
    //     "L (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f), T (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f) axis %u, pos %.3f\n",
    //     // loose_bound.min.x, loose_bound.min.y, loose_bound.min.z, // loose_bound.max.x, loose_bound.max.y,
    //     loose_bound.max.z,                                                      // tight_bound.min.x,
    //     tight_bound.min.y, tight_bound.min.z,                                                      //
    //     tight_bound.max.x, tight_bound.max.y, tight_bound.max.z, // axis, split_pos);

    next_node_loose_bounds[node_id * 2] = loose_bound;
    next_node_loose_bounds[node_id * 2].max[axis] = split_pos;

    next_node_loose_bounds[node_id * 2 + 1] = loose_bound;
    next_node_loose_bounds[node_id * 2 + 1].min[axis] = split_pos;
}

struct join_aabb : public thrust::binary_function<AABB3, AABB3, AABB3>
{
    CUDA_HOST_DEVICE AABB3 operator()(AABB3 b0, AABB3 b1) { return join(b0, b1); }
};

__global__ void partition_prims_count(uint32_t num_refs_chunked, const AABB3 *__restrict__ prim_bounds,
                                      const uint32_t *__restrict__ prim_ids, uint32_t num_nodes,
                                      const uint32_t *__restrict__ node_prim_count_psum,
                                      const uint32_t *__restrict__ node_chunk_count_psum,
                                      const AABB3 *__restrict__ node_loose_bounds,
                                      const LargeNodeChildInfo *__restrict__ child_info,
                                      uint32_t *__restrict__ next_node_prim_count)
{
    // uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t chunk_id = blockIdx.x;

    // if (gid >= num_prims) {
    // return;
    //}

    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    uint32_t num_valid = CHUNK_SIZE;
    // Am I in the last chunk of a node?
    if (chunk_id + 1 == node_chunk_count_psum[node_id + 1]) {
        uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
        if (node_prim_count % CHUNK_SIZE != 0) {
            num_valid = node_prim_count - (node_prim_count / CHUNK_SIZE) * CHUNK_SIZE;
        }
    }
    if (threadIdx.x >= num_valid) {
        return;
    }

    uint32_t ref_id =
        node_prim_count_psum[node_id] + (chunk_id - node_chunk_count_psum[node_id]) * CHUNK_SIZE + threadIdx.x;
    uint32_t prim_id = prim_ids[ref_id];

    // uint32_t prim_id = prim_ids[gid];

    SplitPlane split = child_info[node_id].split;
    // AABB3 node_loose_bound = node_loose_bounds[node_id];
    float split_pos = split.pos;

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

__global__ void mark_small_nodes(float traversal_cost, uint32_t parent_num_nodes,
                                 //
                                 const uint32_t *__restrict__ parent_node_prim_count,
                                 const AABB3 *__restrict__ parent_loose_bounds,
                                 const uint8_t *__restrict__ parent_bad_flags,
                                 //
                                 const uint32_t *__restrict__ next_node_prim_count,
                                 const AABB3 *__restrict__ next_loose_bounds,
                                 //
                                 uint32_t *__restrict__ small_flags, uint32_t *__restrict__ small_flags_invert,
                                 uint8_t *__restrict__ bad_flags)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= parent_num_nodes) {
        return;
    }

    uint32_t left_node_id = parent_node_id * 2;
    uint32_t right_node_id = parent_node_id * 2 + 1;

    float sah0 = parent_node_prim_count[parent_node_id + 1] - parent_node_prim_count[parent_node_id];
    float area = parent_loose_bounds[parent_node_id].surface_area();
    uint32_t n_left = next_node_prim_count[left_node_id];
    float area_left = next_loose_bounds[left_node_id].surface_area();
    uint32_t n_right = next_node_prim_count[right_node_id];
    float area_right = next_loose_bounds[right_node_id].surface_area();
    // int isect_cost = 5, int traversal_cost = 1
    float sah = (n_left * area_left + n_right * area_right) / area + traversal_cost;
    if (sah < sah0) {
        bad_flags[left_node_id] = bad_flags[right_node_id] = 0;
    } else {
        bad_flags[left_node_id] = bad_flags[right_node_id] = parent_bad_flags[parent_node_id] + 1;
    }
    // printf("%u, %u, %u\n", left_node_id, (uint32_t)bad_flags[left_node_id], next_node_prim_count[left_node_id]);
    // printf("%u, %u, %u\n", right_node_id, (uint32_t)bad_flags[right_node_id], next_node_prim_count[right_node_id]);

    if (bad_flags[left_node_id] == BAD_SPLIT_TRYS || next_node_prim_count[left_node_id] <= LARGE_NODE_THRESHOLD) {
        small_flags[left_node_id] = 1;
        small_flags_invert[left_node_id] = 0;
    } else {
        small_flags[left_node_id] = 0;
        small_flags_invert[left_node_id] = 1;
    }
    if (bad_flags[right_node_id] == BAD_SPLIT_TRYS || next_node_prim_count[right_node_id] <= LARGE_NODE_THRESHOLD) {
        small_flags[right_node_id] = 1;
        small_flags_invert[right_node_id] = 0;
    } else {
        small_flags[right_node_id] = 0;
        small_flags_invert[right_node_id] = 1;
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
partition_prims_assign(uint32_t num_refs_chunked, const AABB3 *__restrict__ prim_bounds,
                       const uint32_t *__restrict__ prim_ids, uint32_t num_nodes,
                       const uint32_t *__restrict__ node_prim_count_psum,
                       const uint32_t *__restrict__ node_chunk_count_psum, const AABB3 *__restrict__ node_loose_bounds,
                       const LargeNodeChildInfo *__restrict__ child_info, const uint32_t *__restrict__ small_flags,
                       const uint32_t *__restrict__ small_flags_invert, uint32_t *__restrict__ next_prim_ids_small,
                       uint32_t *__restrict__ next_prim_ids_large, uint32_t *__restrict__ assign_offsets_small,
                       uint32_t *__restrict__ assign_offsets_large)
{
    // uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t chunk_id = blockIdx.x;
    // if (gid >= num_prims) {
    // return;
    //}

    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return chunk_id >= node_chunk_count_psum[i]; });
    uint32_t num_valid = CHUNK_SIZE;
    // Am I in the last chunk of a node?
    if (chunk_id + 1 == node_chunk_count_psum[node_id + 1]) {
        uint32_t node_prim_count = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
        if (node_prim_count % CHUNK_SIZE != 0) {
            num_valid = node_prim_count - (node_prim_count / CHUNK_SIZE) * CHUNK_SIZE;
        }
    }
    if (threadIdx.x >= num_valid) {
        return;
    }

    uint32_t ref_id =
        node_prim_count_psum[node_id] + (chunk_id - node_chunk_count_psum[node_id]) * CHUNK_SIZE + threadIdx.x;
    uint32_t prim_id = prim_ids[ref_id];

    SplitPlane split = child_info[node_id].split;
    // AABB3 node_loose_bound = node_loose_bounds[node_id];
    float split_pos = split.pos;

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
    total_bound = ksc::cuda_alloc_device_low_level(sizeof(ksc::AABB3));

    std::vector<LargeNodeArray> upper_tree{init_build(input)};
    SmallRootArray small_roots;

    // Large node stage.
    uint8_t upper_depth = 0;
    for (;; ++upper_depth) {
        ASSERT(upper_depth <= MAX_DEPTH);
        LargeNodeArray &curr = upper_tree.back();
        LargeNodeArray next = large_node_step(input, curr, upper_depth, small_roots);
        if (!next.node_loose_bounds.empty()) {
            upper_tree.push_back(std::move(next));
        } else {
            break;
        }
    }
    // Small node stage.
    prepare_small_roots(input, small_roots);
    std::vector<SmallNodeArray> lower_tree{SmallNodeArray{}};

    thrust::device_ptr<uint32_t> max_depth;
    if (input.stats) {
        max_depth = thrust::device_new<uint32_t>();
        thrust::fill_n(max_depth, 1, 0);
    }
    uint8_t lower_depth = 0;
    for (;; ++lower_depth) {
        SmallNodeArray &curr = lower_tree.back();
        SmallNodeArray next = small_node_step(input, curr, small_roots, lower_depth, max_depth);
        if (!next.node_loose_bounds.empty()) {
            lower_tree.push_back(std::move(next));
        } else {
            break;
        }
    }
    thrust::device_ptr<uint32_t> n_leaves;
    thrust::device_ptr<uint32_t> prim_ref_storage;
    if (input.stats) {
        n_leaves = thrust::device_new<uint32_t>();
        thrust::fill_n(n_leaves, 1, 0);
        prim_ref_storage = thrust::device_new<uint32_t>();
        thrust::fill_n(prim_ref_storage, 1, 0);
    }
    // Final compact stage.
    compact(input, upper_tree, small_roots, lower_tree, n_leaves, prim_ref_storage);

    if (input.stats) {
        input.stats->compact_strorage_bytes = nodes_storage.requested_size;
        thrust::copy_n(max_depth, 1, &input.stats->max_depth);
        input.stats->upper_max_depth = upper_depth + 1;
        input.stats->lower_max_depth = lower_depth;
        uint32_t prim_ref_storage_cpu;
        thrust::copy_n(prim_ref_storage, 1, &prim_ref_storage_cpu);
        thrust::copy_n(n_leaves, 1, &input.stats->n_leaves);
        input.stats->n_nodes =
            (input.stats->compact_strorage_bytes - prim_ref_storage_cpu * sizeof(uint32_t)) / sizeof(CompactKdTreeNode);
        input.stats->n_small_roots = (uint32_t)small_roots.node_loose_bounds.size();
        input.stats->n_prim_refs = input.stats->n_leaves + prim_ref_storage_cpu;
        thrust::device_free(max_depth);
        thrust::device_free(prim_ref_storage);
    }

    return;
}

LargeNodeArray ParallelKdTree::init_build(const ParallelKdTreeBuildInput &input)
{
    uint32_t num_prims = input.num_prims;
    LargeNodeArray large_nodes;
    large_nodes.prim_ids.resize(num_prims);
    thrust::sequence(large_nodes.prim_ids.begin(), large_nodes.prim_ids.end());

    // Allocate one more to automatically get the total count from prefix sum.
    large_nodes.node_prim_count_psum.resize(2);
    thrust::fill_n(large_nodes.node_prim_count_psum.begin(), 1, num_prims);
    thrust::exclusive_scan(large_nodes.node_prim_count_psum.begin(), large_nodes.node_prim_count_psum.end(),
                           large_nodes.node_prim_count_psum.begin()); // support in-place prefix sum.

    large_nodes.bad_flags.resize(1, (uint8_t)0);

    return large_nodes;
}

LargeNodeArray ParallelKdTree::large_node_step(const ParallelKdTreeBuildInput &input, LargeNodeArray &large_nodes,
                                               uint8_t depth, SmallRootArray &small_roots)
{
    bool record_depth = (input.stats != nullptr);
    uint32_t num_nodes = large_nodes.node_prim_count_psum.size() - 1;
    uint32_t num_prims = input.num_prims;
    const AABB3 *prim_bounds_ptr = reinterpret_cast<const AABB3 *>(input.prim_bounds_storage.dptr);

    thrust::copy_n(large_nodes.node_prim_count_psum.rbegin(), 1, &num_prims);

    large_nodes.node_chunk_count_psum.resize(num_nodes + 1);
    run_kernel_1d(count_node_chunks, 0, (cudaStream_t)(0), num_nodes, num_nodes,
                  large_nodes.node_prim_count_psum.data().get(), large_nodes.node_chunk_count_psum.data().get());
    thrust::exclusive_scan(large_nodes.node_chunk_count_psum.begin(), large_nodes.node_chunk_count_psum.end(),
                           large_nodes.node_chunk_count_psum.begin()); // support in-place prefix sum.

    uint32_t num_chunks = 0;
    thrust::copy_n(large_nodes.node_chunk_count_psum.begin() + num_nodes, 1, &num_chunks);
    large_nodes.chunk_bounds.resize(num_chunks);
    large_nodes.chunk_to_node_map.resize(num_chunks);
    uint32_t num_refs_chunked = num_chunks * CHUNK_SIZE;
    run_kernel_1d<CHUNK_SIZE>(compute_chunk_bounds, 0, (cudaStream_t)(0), num_refs_chunked, num_refs_chunked,
                              prim_bounds_ptr, large_nodes.prim_ids.data().get(), num_nodes,
                              large_nodes.node_prim_count_psum.data().get(),
                              large_nodes.node_chunk_count_psum.data().get(), large_nodes.chunk_bounds.data().get(),
                              large_nodes.chunk_to_node_map.data().get());

    // Perform segmented reduction on per-chunk reduction result to compute per-node tight bounding box
    large_nodes.node_tight_bounds.resize(num_nodes);
    thrust::device_vector<uint32_t> segmented_reduce_output_keys;
    segmented_reduce_output_keys.resize(large_nodes.node_tight_bounds.size());
    // https://nvidia.github.io/cccl/thrust/api/function_group__reductions_1ga12501ca04b9b21b101d6a299d899d7af.html
    thrust::reduce_by_key(large_nodes.chunk_to_node_map.begin(), large_nodes.chunk_to_node_map.end(),
                          large_nodes.chunk_bounds.begin(), segmented_reduce_output_keys.begin(),
                          large_nodes.node_tight_bounds.begin(), cuda::std::equal_to<uint32_t>{}, join_aabb{});
    if (large_nodes.node_loose_bounds.empty()) {
        // Initialize: same as tight bound.
        large_nodes.node_loose_bounds = large_nodes.node_tight_bounds;
        cuda_check(cudaMemcpy((void *)total_bound.dptr, large_nodes.node_tight_bounds.data().get(), sizeof(ksc::AABB3),
                              cudaMemcpyKind::cudaMemcpyDefault));
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

    run_kernel_1d<CHUNK_SIZE>(
        partition_prims_count, 0, (cudaStream_t)(0), num_refs_chunked, num_refs_chunked, prim_bounds_ptr,
        large_nodes.prim_ids.data().get(), num_nodes, large_nodes.node_prim_count_psum.data().get(),
        large_nodes.node_chunk_count_psum.data().get(), large_nodes.node_loose_bounds.data().get(),
        large_nodes.node_child_info.data().get(), next_node_prim_count_psum.data().get());
    // if (depth == 7) {
    //     cuda_check(cudaDeviceSynchronize());
    //     cuda_check(cudaGetLastError());
    //     printf("num_refs_chunked: %u\n", num_refs_chunked);
    //     //{
    //     //    thrust::host_vector<uint32_t> h = large_nodes.prim_ids;
    //     //    std::vector<uint32_t> v(h.begin(), h.end());
    //     //    for (int j = 0; j < 100; ++j) {
    //     //        printf("%u ", v[j]);
    //     //        if ((j + 1) % 10 == 0) {
    //     //            printf("\n");
    //     //        }
    //     //    }
    //     //    printf("\n");
    //     //}
    //     {
    //         thrust::host_vector<uint32_t> h = large_nodes.node_prim_count_psum;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("%u ", v[j]);
    //             if ((j + 1) % 10 == 0) {
    //                 printf("\n");
    //             }
    //         }
    //         printf("\n");
    //     }
    //     {
    //         thrust::host_vector<AABB3> h = large_nodes.chunk_bounds;
    //         std::vector<AABB3> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("(%f, %f, %f, %f, %f, %f)\n", v[j].min.x, v[j].min.y, v[j].min.z, v[j].max.x, v[j].max.y,
    //                    v[j].max.z);
    //         }
    //         printf("\n");
    //     }
    //     {
    //         thrust::host_vector<uint32_t> h = large_nodes.node_chunk_count_psum;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("%u ", v[j]);
    //             if ((j + 1) % 10 == 0) {
    //                 printf("\n");
    //             }
    //         }
    //         printf("\n");
    //     }
    //     {
    //         thrust::host_vector<AABB3> h = large_nodes.node_loose_bounds;
    //         std::vector<AABB3> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("(%f, %f, %f, %f, %f, %f)\n", v[j].min.x, v[j].min.y, v[j].min.z, v[j].max.x, v[j].max.y,
    //                    v[j].max.z);
    //         }
    //         printf("\n");
    //     }
    //     {
    //         thrust::host_vector<LargeNodeChildInfo> h = large_nodes.node_child_info;
    //         std::vector<LargeNodeChildInfo> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("%u, %f, %u, %u, %u, %u\n", v[j].split.axis, v[j].split.pos, v[j].children[0].index,
    //                    (uint32_t)v[j].children[0].type, v[j].children[1].index, (uint32_t)v[j].children[1].type);
    //         }
    //         printf("\n");
    //     }
    //     {
    //         thrust::host_vector<uint32_t> h = next_node_prim_count_psum;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         for (int j = 0; j < v.size(); ++j) {
    //             printf("%u ", v[j]);
    //             if ((j + 1) % 10 == 0) {
    //                 printf("\n");
    //             }
    //         }
    //         printf("\n");
    //     }
    // }
    //  cuda_check(cudaDeviceSynchronize());
    //  cuda_check(cudaGetLastError());

    // TODO: DEBUG next level large/small split logic below

    thrust::device_vector<uint32_t> small_flags(num_nodes_next);
    thrust::device_vector<uint32_t> small_flags_invert(num_nodes_next);
    thrust::device_vector<uint8_t> next_node_bad_flags(num_nodes_next);
    run_kernel_1d(mark_small_nodes, 0, (cudaStream_t)(0), num_nodes, input.traversal_cost, num_nodes,
                  large_nodes.node_prim_count_psum.data().get(), large_nodes.node_loose_bounds.data().get(),
                  large_nodes.bad_flags.data().get(),
                  //
                  next_node_prim_count_psum.data().get(), next_node_loose_bounds.data().get(),
                  //
                  small_flags.data().get(), small_flags_invert.data().get(), next_node_bad_flags.data().get());

    thrust::device_vector<uint32_t> small_flags_copy = small_flags;
    thrust::device_vector<uint32_t> small_flags_invert_copy = small_flags_invert;

    thrust::inclusive_scan(small_flags.begin(), small_flags.end(), small_flags.begin());
    thrust::inclusive_scan(small_flags_invert.begin(), small_flags_invert.end(), small_flags_invert.begin());

    uint32_t num_small_nodes_next = 0;
    thrust::copy_n(small_flags.rbegin(), 1, &num_small_nodes_next);
    uint32_t num_large_nodes_next = num_nodes_next - num_small_nodes_next;
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());

    printf("[%u] %u = %u (small) + %u (large)\n", (uint32_t)(depth), num_nodes_next, num_small_nodes_next,
           num_large_nodes_next);

    thrust::transform(small_flags.begin(), small_flags.end(), small_flags_copy.begin(), small_flags.begin(),
                      fix_small_flag{});
    thrust::transform(small_flags_invert.begin(), small_flags_invert.end(), small_flags_invert_copy.begin(),
                      small_flags_invert.begin(), fix_small_flag{});
    // After transform, small_flag should either be 0 for large nodes, or the 1-based node indices for small nodes.
    // vice-verse for small_flags_invert

    // Record parent/child references
    uint32_t curr_small_node_count = small_roots.node_loose_bounds.size();
    run_kernel_1d(store_child_refs, 0, (cudaStream_t)(0), num_nodes, num_nodes, small_flags.data().get(),
                  small_flags_invert.data().get(), large_nodes.node_child_info.data().get(), curr_small_node_count);

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
    // Same. Need stable partition.
    thrust::device_vector<uint8_t> next_node_bad_flags_small(num_small_nodes_next);
    thrust::device_vector<uint8_t> next_node_bad_flags_large(num_large_nodes_next);
    thrust::stable_partition_copy(next_node_bad_flags.begin(), next_node_bad_flags.end(),
                                  small_flags_invert.begin(), //
                                  next_node_bad_flags_small.begin(), next_node_bad_flags_large.begin(), is_zero{});

    thrust::device_vector<uint32_t> next_prim_ids_small(next_num_prims_small);
    thrust::device_vector<uint32_t> next_prim_ids_large(next_num_prims_large);

    thrust::device_vector<uint32_t> assign_offsets_small = next_node_prim_count_psum_small;
    thrust::device_vector<uint32_t> assign_offsets_large = next_node_prim_count_psum_large;
    // Partition prim ids while accounting for large/small separation
    run_kernel_1d<CHUNK_SIZE>(
        partition_prims_assign, 0, (cudaStream_t)(0), num_refs_chunked, num_refs_chunked, prim_bounds_ptr,
        large_nodes.prim_ids.data().get(), num_nodes, large_nodes.node_prim_count_psum.data().get(),
        large_nodes.node_chunk_count_psum.data().get(), large_nodes.node_loose_bounds.data().get(),
        large_nodes.node_child_info.data().get(), small_flags.data().get(), small_flags_invert.data().get(), //
        next_prim_ids_small.data().get(), next_prim_ids_large.data().get(),                                  //
        assign_offsets_small.data().get(), assign_offsets_large.data().get());

    LargeNodeArray next_large;
    next_large.prim_ids = std::move(next_prim_ids_large);
    next_large.node_prim_count_psum = std::move(next_node_prim_count_psum_large);
    next_large.node_loose_bounds = std::move(next_node_loose_bound_large);
    next_large.bad_flags = std::move(next_node_bad_flags_large);

    // Append to a global small node array
    {
        uint32_t old_size = small_roots.prim_ids.size();
        small_roots.prim_ids.resize(old_size + next_prim_ids_small.size());
        thrust::copy(next_prim_ids_small.begin(), next_prim_ids_small.end(), small_roots.prim_ids.begin() + old_size);
    }
    {
        uint32_t old_size = small_roots.node_loose_bounds.size();
        small_roots.node_loose_bounds.resize(old_size + next_node_loose_bound_small.size());
        thrust::copy(next_node_loose_bound_small.begin(), next_node_loose_bound_small.end(),
                     small_roots.node_loose_bounds.begin() + old_size);
    }
    {
        uint32_t old_size = small_roots.bad_flags.size();
        small_roots.bad_flags.resize(old_size + next_node_bad_flags_small.size());
        thrust::copy(next_node_bad_flags_small.begin(), next_node_bad_flags_small.end(),
                     small_roots.bad_flags.begin() + old_size);
    }
    if (record_depth) {
        uint32_t old_size = small_roots.depths.size();
        // Same depth for all new small nodes this batch.
        small_roots.depths.resize(old_size + next_node_bad_flags_small.size());
        thrust::fill_n(small_roots.depths.begin() + old_size, next_node_bad_flags_small.size(), depth + 1);
    }

    // Need to add a global offset and append
    uint32_t global_small_node_prim_count = 0;
    if (small_roots.node_prim_count_psum.size() > 0) {
        thrust::copy_n(small_roots.node_prim_count_psum.rbegin(), 1, &global_small_node_prim_count);
        thrust::transform(
            next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_small.end(),
            next_node_prim_count_psum_small.begin(),
            [global_small_node_prim_count] __device__(uint32_t x) { return x + global_small_node_prim_count; });
    }
    {
        uint32_t old_size = small_roots.node_prim_count_psum.size();
        small_roots.node_prim_count_psum.resize(std::max(old_size, 1u) - 1 + next_node_prim_count_psum_small.size());
        thrust::copy(next_node_prim_count_psum_small.begin(), next_node_prim_count_psum_small.end(),
                     small_roots.node_prim_count_psum.begin() + std::max(old_size, 1u) - 1);
    }

    return next_large;
}

// Per split candidate
__global__ void prepare_small_roots_kernel(uint32_t num_candidates, uint32_t num_nodes,
                                           const AABB3 *__restrict__ prim_bounds, const uint32_t *__restrict__ prim_ids,
                                           const uint32_t *__restrict__ node_prim_count_psum,
                                           const uint8_t *__restrict__ node_bad_flags,
                                           SAHSplitCandidate *__restrict__ sah_split_candidates)
{
    uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= num_candidates) {
        return;
    }
    // 6 candidates each prim: [xmin, xmax, ymin, ymax, zmin, zmax]
    // what is my split axis and pos?
    uint32_t node_id = find_interval(num_nodes + 1, [&](uint32_t i) { return (gid / 6) >= node_prim_count_psum[i]; });
    if (node_bad_flags[node_id] == BAD_SPLIT_TRYS) {
        // Skip bad nodes
        // DEBUG
        sah_split_candidates[gid].left_mask = 0;
        sah_split_candidates[gid].right_mask = 0;
        sah_split_candidates[gid].split_axis = 0;
        sah_split_candidates[gid].split_pos = nanf("");
        return;
    }

    uint32_t cand_prim_id = prim_ids[gid / 6];
    AABB3 cand_bound = prim_bounds[cand_prim_id];
    uint32_t axis = (gid % 6) / 2;
    float split_pos = cand_bound[gid % 2][axis];

    uint64_t left_mask = 0;
    uint64_t right_mask = 0;
    for (uint32_t i = node_prim_count_psum[node_id], bit = 0; i < node_prim_count_psum[node_id + 1]; ++i, ++bit) {
        // There should be <= LARGE_NODE_THRESHOLD (64) prims
        CUDA_ASSERT(bit < LARGE_NODE_THRESHOLD);
        uint32_t prim_id = prim_ids[i];
        AABB3 b = prim_bounds[prim_id];
        // We can have primitives on both sides,
        // but make sure each is not crossing both sides when using one of its own side as the splitting plane.
        if (b.min[axis] < split_pos) {
            left_mask |= (uint64_t)(1llu << (uint64_t)bit);
        }
        if (b.max[axis] > split_pos) {
            right_mask |= (uint64_t)(1llu << (uint64_t)bit);
        }
    }

    // DEBUG
    //{
    //    uint64_t n_prims = node_prim_count_psum[node_id + 1] - node_prim_count_psum[node_id];
    //    uint64_t prim_mask;
    //    if (n_prims == LARGE_NODE_THRESHOLD) {
    //        prim_mask = (uint64_t)(~0llu);s
    //    } else {
    //        prim_mask = ((uint64_t)1llu << (uint64_t)n_prims) - 1llu;
    //    }
    //    CUDA_ASSERT(prim_mask == left_mask | right_mask);
    //}

    sah_split_candidates[gid].left_mask = left_mask;
    sah_split_candidates[gid].right_mask = right_mask;
    sah_split_candidates[gid].split_axis = axis;
    sah_split_candidates[gid].split_pos = split_pos;
}

void ParallelKdTree::prepare_small_roots(const ParallelKdTreeBuildInput &input, SmallRootArray &small_roots)
{
    const AABB3 *prim_bounds_ptr = reinterpret_cast<const AABB3 *>(input.prim_bounds_storage.dptr);

    uint32_t num_candidates = 0;
    thrust::copy_n(small_roots.node_prim_count_psum.rbegin(), 1, &num_candidates);
    num_candidates *= 6;
    small_roots.sah_split_candidates.resize(num_candidates);
    uint32_t num_nodes = small_roots.node_prim_count_psum.size() - 1;
    run_kernel_1d(prepare_small_roots_kernel, 0, (cudaStream_t)(0), num_candidates, num_candidates, num_nodes,
                  prim_bounds_ptr, small_roots.prim_ids.data().get(), small_roots.node_prim_count_psum.data().get(),
                  small_roots.bad_flags.data().get(), small_roots.sah_split_candidates.data().get());
    // DEBUG
    // cuda_check(cudaDeviceSynchronize());
    // cuda_check(cudaGetLastError());
    //{
    //    thrust::host_vector<uint32_t> h = small_roots.prim_ids;
    //    std::vector<uint32_t> v(h.begin(), h.end());
    //    std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //    printf("prim_ids: %s\n", md5.c_str());
    //}
    //{
    //    thrust::host_vector<uint32_t> h = small_roots.node_prim_count_psum;
    //    std::vector<uint32_t> v(h.begin(), h.end());
    //    for (int j = 0; j < v.size() - 1; ++j) {
    //        if (v[j + 1] - v[j] > 64) {
    //            int wtf = 0;
    //        }
    //    }
    //    std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //    printf("node_prim_count_psum: %s\n", md5.c_str());
    //}
    //{
    //    thrust::host_vector<uint8_t> h = small_roots.bad_flags;
    //    std::vector<uint8_t> v(h.begin(), h.end());
    //    std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint8_t) * v.size()).finalize().toString();
    //    printf("bad_flags: %s\n", md5.c_str());
    //}
    //{
    //    thrust::host_vector<SAHSplitCandidate> h = small_roots.sah_split_candidates;
    //    std::vector<SAHSplitCandidate> v(h.begin(), h.end());
    //    std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(SAHSplitCandidate) *
    //    v.size()).finalize().toString(); printf("sah_split_candidates: %s\n", md5.c_str());

    //    int debug = 0;
    //}
}

// One node per thread
__global__ void compute_sah_split_small_nodes(uint32_t max_leaf_prims, float traversal_cost, uint32_t num_nodes,
                                              const uint32_t *__restrict__ small_root_ids,
                                              const uint64_t *__restrict__ prim_masks,
                                              const AABB3 *__restrict__ node_loose_bounds,
                                              const uint8_t *__restrict__ small_root_bad_flags,
                                              const uint8_t *__restrict__ small_root_depths,
                                              const uint32_t *__restrict__ small_root_node_prim_count_psum,
                                              const SAHSplitCandidate *__restrict__ sah_split_candidates, uint8_t depth,
                                              uint32_t *__restrict__ sah_splits, uint32_t *__restrict__ split_tags,
                                              //
                                              uint32_t *__restrict__ max_depth)
{
    uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= num_nodes) {
        return;
    }
    uint32_t small_root = small_root_ids ? small_root_ids[node_id] : node_id;
    uint32_t curr_depth = small_root_depths[small_root] + depth;
    // Check depth
    // Skip duplicate bookkeeping in the first small stage iteration.
    //     if (node_bad_flags[node_id] == BAD_SPLIT_TRYS) {
    bool bad_flag = small_root_bad_flags ? (small_root_bad_flags[node_id] == BAD_SPLIT_TRYS) : false;
    if (bad_flag || curr_depth == MAX_DEPTH) {
        sah_splits[node_id] = (uint32_t)(~0);
        split_tags[node_id] = 0;
        if (max_depth) {
            atomicMax(max_depth, curr_depth);
        }
        return;
    }

    uint64_t prim_mask;
    uint32_t n_prims;
    if (prim_masks) {
        prim_mask = prim_masks[node_id];
        n_prims = (uint32_t)__popcll(prim_mask);
    } else {
        // For first iteration (small roots), just fill mask with (least significant) n_prims bits.
        n_prims = small_root_node_prim_count_psum[node_id + 1] - small_root_node_prim_count_psum[node_id];
        if (n_prims == LARGE_NODE_THRESHOLD) {
            prim_mask = (uint64_t)(~0llu);
        } else {
            prim_mask = ((uint64_t)1llu << (uint64_t)n_prims) - 1llu;
        }
    }
    CUDA_ASSERT(n_prims <= LARGE_NODE_THRESHOLD);

    if (n_prims <= max_leaf_prims) {
        // DEBUG
        // printf("Leq than leaf threshold %d\n", n_prims);
        sah_splits[node_id] = (uint32_t)(~0);
        split_tags[node_id] = 0;
        if (max_depth) {
            atomicMax(max_depth, curr_depth);
        }
        return;
    }

    AABB3 bound = node_loose_bounds[node_id];
    CUDA_ASSERT(!bound.is_empty());
    float area = bound.surface_area();
    float min_sah = (float)n_prims;
    uint32_t best_split = (uint32_t)(~0);
    // printf("%f, %f\n", area, min_sah);

    // DEBUG
    // int best_n_left = 0;
    // int best_n_right = 0;
    for (uint32_t bit = 0; bit < LARGE_NODE_THRESHOLD; ++bit) {
        if ((prim_mask >> bit) & 1) {
            uint32_t offset = (small_root_node_prim_count_psum[small_root] + bit) * 6;
            for (uint32_t j = 0; j < 6; ++j) {
                uint32_t split_idx = offset + j;
                const SAHSplitCandidate &s = sah_split_candidates[split_idx];

                CUDA_ASSERT((prim_mask & (s.left_mask | s.right_mask)) == prim_mask);

                // CUDA_ASSERT(s.split_pos > bound.min[s.split_axis] && s.split_pos < bound.max[s.split_axis]);
                //  TODO: IMPROVE ME!!!
                if (s.split_pos > bound.min[s.split_axis] && s.split_pos < bound.max[s.split_axis]) {
                    uint64_t left = prim_mask & s.left_mask;
                    uint64_t right = prim_mask & s.right_mask;
                    // Count the number of bits that are set to 1 in a 64-bit integer.
                    int n_left = __popcll(left);
                    int n_right = __popcll(right);
                    // Calculate split nodes area
                    AABB3 b_left = bound;
                    b_left.max[s.split_axis] = s.split_pos;
                    CUDA_ASSERT(!b_left.is_empty());
                    float area_left = b_left.surface_area();
                    AABB3 b_right = bound;
                    b_right.min[s.split_axis] = s.split_pos;
                    CUDA_ASSERT(!b_right.is_empty());
                    float area_right = b_right.surface_area();
                    // Compute SAH
                    // int isect_cost = 5, int traversal_cost = 1
                    float sah = (n_left * area_left + n_right * area_right) / area + traversal_cost;
                    CUDA_ASSERT(!isnan(sah) && !isnan(min_sah) && sah > 0.0f && min_sah > 0.0f);
                    if (sah < min_sah) {
                        min_sah = sah;
                        best_split = split_idx;
                    } // It is possible to get multiple equally best splits (which can be a source of indeterministic
                      // build results when paired with parallel execution).
                }
            }
        }
    }

    sah_splits[node_id] = best_split;
    split_tags[node_id] = (best_split == (uint32_t)(~0)) ? 0 : 1;
    if (max_depth) {
        atomicMax(max_depth, curr_depth);
    }

    // DEBUG
    // if (split_tags[node_id]) {
    //    printf("SAH Split P: %d L: %d, R: %d\n", n_prims, best_n_left, best_n_right);
    //} else {
    //    printf("SAH no split %d\n", n_prims);
    //}
}

// One node per thread
__global__ void assign_children_small_nodes(uint32_t num_nodes, const uint32_t *__restrict__ parent_small_root_ids,
                                            const uint64_t *__restrict__ parent_prim_masks,
                                            const AABB3 *__restrict__ parent_node_loose_bounds,
                                            const uint32_t *__restrict__ child_offsets, //
                                            const uint32_t *__restrict__ small_root_node_prim_count_psum,
                                            const SAHSplitCandidate *__restrict__ sah_split_candidates,
                                            const uint32_t *__restrict__ sah_splits,
                                            uint32_t *__restrict__ child_small_root_ids,
                                            uint64_t *__restrict__ child_prim_masks,
                                            AABB3 *__restrict__ child_node_loose_bounds)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= num_nodes) {
        return;
    }
    uint32_t ch = child_offsets[parent_node_id];
    if (ch == (uint32_t)(~0)) {
        return;
    }

    uint64_t parent_prim_mask;
    if (parent_prim_masks) {
        parent_prim_mask = parent_prim_masks[parent_node_id];
    } else {
        // For first iteration (small roots), just fill mask with (least significant) n_prims bits.
        uint32_t n_prims =
            small_root_node_prim_count_psum[parent_node_id + 1] - small_root_node_prim_count_psum[parent_node_id];
        CUDA_ASSERT(n_prims <= LARGE_NODE_THRESHOLD);
        if (n_prims == LARGE_NODE_THRESHOLD) {
            parent_prim_mask = (uint64_t)(~0llu);
        } else {
            parent_prim_mask = ((uint64_t)1llu << (uint64_t)n_prims) - 1llu;
        }
    }
    uint32_t split_idx = sah_splits[parent_node_id];
    const SAHSplitCandidate &s = sah_split_candidates[split_idx];

    AABB3 parent_bound = parent_node_loose_bounds[parent_node_id];

    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t child_node_id = 2 * ch + i;
        bool left = (i % 2 == 0);
        uint64_t child_prim_mask = parent_prim_mask & (left ? s.left_mask : s.right_mask);
        child_prim_masks[child_node_id] = child_prim_mask;
        child_small_root_ids[child_node_id] =
            parent_small_root_ids ? parent_small_root_ids[parent_node_id] : parent_node_id;

        CUDA_ASSERT(s.split_pos > parent_bound.min[s.split_axis] && s.split_pos < parent_bound.max[s.split_axis]);

        AABB3 child_bound = parent_bound;
        if (left) {
            child_bound.max[s.split_axis] = s.split_pos;
        } else {
            child_bound.min[s.split_axis] = s.split_pos;
        }
        child_node_loose_bounds[child_node_id] = child_bound;
    }
}

SmallNodeArray ParallelKdTree::small_node_step(const ParallelKdTreeBuildInput &input, SmallNodeArray &small_nodes,
                                               const SmallRootArray &small_roots, uint8_t depth,
                                               thrust::device_ptr<uint32_t> max_depth)
{
    uint32_t num_nodes;
    const uint32_t *curr_small_root_ids_ptr;
    const uint64_t *curr_prim_masks_ptr;
    const AABB3 *curr_node_loose_bounds_ptr;
    const uint8_t *curr_bad_flags_ptr;
    // Skip duplicate bookkeeping in the first small stage iteration.
    if (small_nodes.small_root_ids.empty()) {
        num_nodes = small_roots.node_prim_count_psum.size() - 1;
        curr_small_root_ids_ptr = nullptr;
        curr_node_loose_bounds_ptr = small_roots.node_loose_bounds.data().get();
        curr_bad_flags_ptr = small_roots.bad_flags.data().get();
        curr_prim_masks_ptr = nullptr;

    } else {
        num_nodes = small_nodes.node_loose_bounds.size();
        curr_small_root_ids_ptr = small_nodes.small_root_ids.data().get();
        curr_node_loose_bounds_ptr = small_nodes.node_loose_bounds.data().get();
        curr_bad_flags_ptr = nullptr;
        curr_prim_masks_ptr = small_nodes.prim_masks.data().get();
    }

    thrust::device_vector<uint32_t> split_tags(num_nodes);
    small_nodes.sah_splits.resize(num_nodes);
    run_kernel_1d(compute_sah_split_small_nodes, 0, (cudaStream_t)(0), num_nodes, input.max_leaf_prims,
                  input.traversal_cost, num_nodes, curr_small_root_ids_ptr, curr_prim_masks_ptr,
                  curr_node_loose_bounds_ptr,
                  curr_bad_flags_ptr, //
                  small_roots.depths.data().get(), small_roots.node_prim_count_psum.data().get(),
                  small_roots.sah_split_candidates.data().get(), depth, //
                  small_nodes.sah_splits.data().get(), split_tags.data().get(), max_depth.get());

    small_nodes.child_offsets = split_tags;
    thrust::inclusive_scan(small_nodes.child_offsets.begin(), small_nodes.child_offsets.end(),
                           small_nodes.child_offsets.begin());

    // if (depth == 4) {
    //     cuda_check(cudaDeviceSynchronize());
    //     cuda_check(cudaGetLastError());
    //     {
    //         thrust::host_vector<uint32_t> h = small_nodes.small_root_ids;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //         printf("small_root_ids: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<uint64_t> h = small_nodes.prim_masks;
    //         std::vector<uint64_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint64_t) * v.size()).finalize().toString();
    //         printf("prim_masks: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<AABB3> h = small_nodes.node_loose_bounds;
    //         std::vector<AABB3> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(AABB3) * v.size()).finalize().toString();
    //         printf("node_loose_bounds: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<uint8_t> h = small_roots.depths;
    //         std::vector<uint8_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint8_t) * v.size()).finalize().toString();
    //         printf("small_root_depths: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<uint32_t> h = small_roots.node_prim_count_psum;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //         printf("small_root_node_prim_count_psum: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<SAHSplitCandidate> h = small_roots.sah_split_candidates;
    //         std::vector<SAHSplitCandidate> v(h.begin(), h.end());
    //         std::string md5 =
    //             Chocobo1::MD5().addData(v.data(), sizeof(SAHSplitCandidate) * v.size()).finalize().toString();
    //         printf("sah_split_candidates: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<uint32_t> h = small_nodes.sah_splits;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //         printf("sah_splits: %s\n", md5.c_str());
    //     }
    //     {
    //         thrust::host_vector<uint32_t> h = small_nodes.child_offsets;
    //         std::vector<uint32_t> v(h.begin(), h.end());
    //         std::string md5 = Chocobo1::MD5().addData(v.data(), sizeof(uint32_t) * v.size()).finalize().toString();
    //         printf("child_offsets: %s\n", md5.c_str());
    //     }
    //     int debug = 0;
    // }

    // Copy this before the transform below.
    uint32_t num_nodes_next = 0;
    thrust::copy_n(small_nodes.child_offsets.rbegin(), 1, &num_nodes_next);
    num_nodes_next *= 2;

    printf("[%u] %u\n", (uint32_t)depth, num_nodes_next);

    thrust::transform(small_nodes.child_offsets.begin(), small_nodes.child_offsets.end(), split_tags.begin(),
                      small_nodes.child_offsets.begin(), [] __device__(uint32_t co, uint32_t st) -> uint32_t {
                          if (!st) {
                              return (uint32_t)(~0);
                          } else {
                              return co - 1;
                          }
                      });

    if (num_nodes_next == 0) {
        // All done.
        return SmallNodeArray{};
    }

    SmallNodeArray small_nodes_next;
    small_nodes_next.node_loose_bounds.resize(num_nodes_next);
    small_nodes_next.prim_masks.resize(num_nodes_next);
    small_nodes_next.small_root_ids.resize(num_nodes_next);
    run_kernel_1d(assign_children_small_nodes, 0, (cudaStream_t)(0), num_nodes, num_nodes, curr_small_root_ids_ptr,
                  curr_prim_masks_ptr, curr_node_loose_bounds_ptr, small_nodes.child_offsets.data().get(),
                  small_roots.node_prim_count_psum.data().get(), //
                  small_roots.sah_split_candidates.data().get(), small_nodes.sah_splits.data().get(),
                  small_nodes_next.small_root_ids.data().get(), small_nodes_next.prim_masks.data().get(),
                  small_nodes_next.node_loose_bounds.data().get());

    return small_nodes_next;
}

__global__ void count_lower_subtree_sizes(uint32_t parent_num_nodes,
                                          const uint32_t *__restrict__ small_root_node_prim_count_psum,
                                          const uint64_t *__restrict__ parent_prim_masks,
                                          const uint32_t *__restrict__ child_offsets,
                                          const uint32_t *__restrict__ child_sizes, uint32_t *__restrict__ parent_sizes,
                                          //
                                          uint32_t *__restrict__ n_leaves, uint32_t *__restrict__ prim_ref_storage)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= parent_num_nodes) {
        return;
    }
    uint32_t ch = child_offsets[parent_node_id];
    if (ch == (uint32_t)(~0)) {
        uint32_t node_prim_size_u32;
        if (parent_prim_masks) {
            uint64_t prim_mask = parent_prim_masks[parent_node_id];
            // First primitive is stored inside the node.
            node_prim_size_u32 = (uint32_t)cuda::std::max((int)__popcll(prim_mask) - 1, 0);
            // node_prim_size_u32 = LARGE_NODE_THRESHOLD;
        } else {
            // Small root, may have bad nodes.
            uint32_t n_prims =
                small_root_node_prim_count_psum[parent_node_id + 1] - small_root_node_prim_count_psum[parent_node_id];
            node_prim_size_u32 = (uint32_t)cuda::std::max((int)n_prims - 1, 0);
        }

        parent_sizes[parent_node_id] = node_header_size_u32 + node_prim_size_u32;
        if (n_leaves) {
            atomicAdd(n_leaves, 1);
        }
        if (prim_ref_storage) {
            atomicAdd(prim_ref_storage, node_prim_size_u32);
        }

    } else {
        // printf("wtf\n");
        parent_sizes[parent_node_id] = node_header_size_u32 + child_sizes[2 * ch] + child_sizes[2 * ch + 1];
    }
}

__global__ void count_upper_subtree_sizes(uint32_t parent_num_nodes, const LargeNodeChildInfo *__restrict__ child_info,
                                          const uint32_t *__restrict__ large_child_sizes,
                                          const uint32_t *__restrict__ small_root_sizes,
                                          uint32_t *__restrict__ parent_sizes)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= parent_num_nodes) {
        return;
    }
    LargeNodeChildInfo info = child_info[parent_node_id];
    uint32_t size = node_header_size_u32;
    for (int i = 0; i < 2; ++i) {
        // In the bot level of upper tree, all children must be small roots.
        if (info.children[i].type == LargeNodeChildType::Large) {
            size += large_child_sizes[info.children[i].index];
        } else {
            size += small_root_sizes[info.children[i].index];
        }
    }
    parent_sizes[parent_node_id] = size;
}

__global__ void compact_upper_tree(uint32_t parent_num_nodes, const LargeNodeChildInfo *__restrict__ child_info,
                                   const uint32_t *__restrict__ parent_addrs, uint32_t *__restrict__ child_addrs,
                                   uint32_t *__restrict__ small_root_addrs, uint32_t *compact_nodes_storage)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= parent_num_nodes) {
        return;
    }
    uint32_t parent_addr = parent_addrs[parent_node_id];
    // CUDA_ASSERT(parent_addr + node_header_size_u32 <= 1459);
    LargeNodeChildInfo info = child_info[parent_node_id];
    uint32_t left_size = 0;

    LargeNodeChildRef left = info.children[0];
    if (left.type == LargeNodeChildType::Large) {
        left_size = child_addrs[left.index];
        child_addrs[left.index] = parent_addr + node_header_size_u32;
    } else {
        left_size = small_root_addrs[left.index];
        small_root_addrs[left.index] = parent_addr + node_header_size_u32;
    }

    uint32_t right_addr = parent_addr + node_header_size_u32 + left_size;
    LargeNodeChildRef right = info.children[1];
    if (right.type == LargeNodeChildType::Large) {
        child_addrs[right.index] = right_addr;
    } else {
        small_root_addrs[right.index] = right_addr;
    }

    CompactKdTreeNode &compact_parent = *reinterpret_cast<CompactKdTreeNode *>(&compact_nodes_storage[parent_addr]);

    // uint32_t prim_count = parent_node_prim_count_psum[parent_node_id + 1] -
    // parent_node_prim_count_psum[parent_node_id];
    //  Large nodes are all interior (?)
    //  TODO: actually we need a way to prevent infinite split in degenerated cases.
    compact_parent.init_interior(info.split.axis, right_addr, info.split.pos);
    CUDA_ASSERT(compact_parent.above_child() > parent_addr);
}

__global__ void compact_lower_tree(uint32_t parent_num_nodes, const uint32_t *__restrict__ parent_small_root_ids,
                                   const uint64_t *__restrict__ parent_prim_masks,
                                   const uint32_t *__restrict__ parent_sah_splits,
                                   const uint32_t *__restrict__ child_offsets,
                                   const uint32_t *__restrict__ parent_addrs,
                                   const uint32_t *__restrict__ small_root_prim_ids,
                                   const uint32_t *__restrict__ small_root_node_prim_count_psum,
                                   const SAHSplitCandidate *__restrict__ sah_split_candidates,
                                   uint32_t *__restrict__ child_addrs, uint32_t *compact_nodes_storage)
{
    uint32_t parent_node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (parent_node_id >= parent_num_nodes) {
        return;
    }
    uint32_t parent_addr = parent_addrs[parent_node_id];
    // CUDA_ASSERT(parent_addr + node_header_size_u32 <= 1459);

    CompactKdTreeNode &compact_parent = *reinterpret_cast<CompactKdTreeNode *>(&compact_nodes_storage[parent_addr]);

    uint32_t ch = child_offsets[parent_node_id];
    if (ch == (uint32_t)(~0)) {
        if (parent_prim_masks) {
            uint32_t small_root = parent_small_root_ids[parent_node_id];
            uint64_t prim_mask = parent_prim_masks[parent_node_id];
            // prim_mask = 0xFFFFFFFFFFFFFFFF;

            uint32_t n_prims = __popcll(prim_mask);
            compact_parent.init_leaf(n_prims);
            uint32_t count = 0; // NOTE: need a separate count to track the number of set bits!
            for (uint32_t bit = 0; bit < LARGE_NODE_THRESHOLD; ++bit) {
                if ((prim_mask >> bit) & 1) { // WAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIT
                    uint32_t offset = (small_root_node_prim_count_psum[small_root] + bit);
                    uint32_t prim = small_root_prim_ids[offset];
                    if (count == 0) {
                        compact_parent.first_prim = prim;
                        ++count;
                    } else {
                        *(compact_nodes_storage + parent_addr + node_header_size_u32 + (count - 1)) = prim;
                        ++count;
                    }
                }
            }
            CUDA_ASSERT(count == n_prims);
        } else {
            // For first iteration (small roots), there can be more than LARGE_NODE_THRESHOLD per node (bad nodes).
            uint32_t prim_ref_start = small_root_node_prim_count_psum[parent_node_id];
            uint32_t n_prims = small_root_node_prim_count_psum[parent_node_id + 1] - prim_ref_start;
            // CUDA_ASSERT(n_prims <= LARGE_NODE_THRESHOLD);
            compact_parent.init_leaf(n_prims);
            for (uint32_t i = 0; i < n_prims; ++i) {
                uint32_t prim = small_root_prim_ids[prim_ref_start + i];
                if (i == 0) {
                    compact_parent.first_prim = prim;
                } else {
                    *(compact_nodes_storage + parent_addr + node_header_size_u32 + (i - 1)) = prim;
                }
            }
        }
    } else {
        uint32_t left_size = child_addrs[2 * ch];
        child_addrs[2 * ch] = parent_addr + node_header_size_u32;
        uint32_t right_addr = parent_addr + node_header_size_u32 + left_size;
        child_addrs[2 * ch + 1] = right_addr;
        uint32_t split_idx = parent_sah_splits[parent_node_id];
        SAHSplitCandidate split = sah_split_candidates[split_idx];
        compact_parent.init_interior(split.split_axis, right_addr, split.split_pos);
        CUDA_ASSERT(compact_parent.above_child() > parent_addr);
    }
}

void ParallelKdTree::compact(const ParallelKdTreeBuildInput &input, std::vector<LargeNodeArray> &upper_tree,
                             const SmallRootArray &small_roots, std::vector<SmallNodeArray> &lower_tree,
                             thrust::device_ptr<uint32_t> n_leaves, thrust::device_ptr<uint32_t> prim_ref_storage)
{
    for (int l = (int)lower_tree.size() - 1; l >= 0; --l) {
        uint32_t parent_num_nodes;
        const uint64_t *curr_prim_masks_ptr;
        if (l > 0) {
            parent_num_nodes = (uint32_t)lower_tree[l].node_loose_bounds.size();
            curr_prim_masks_ptr = lower_tree[l].prim_masks.data().get();
        } else {
            parent_num_nodes = small_roots.node_prim_count_psum.size() - 1;
            curr_prim_masks_ptr = nullptr;
        }

        lower_tree[l].subtree_sizes.resize(parent_num_nodes);
        run_kernel_1d(count_lower_subtree_sizes, 0, (cudaStream_t)(0), parent_num_nodes, parent_num_nodes,
                      small_roots.node_prim_count_psum.data().get(), curr_prim_masks_ptr,
                      lower_tree[l].child_offsets.data().get(),
                      (l < (int)lower_tree.size() - 1) ? lower_tree[l + 1].subtree_sizes.data().get() : nullptr,
                      lower_tree[l].subtree_sizes.data().get(), n_leaves.get(), prim_ref_storage.get());
    }
    for (int l = (int)upper_tree.size() - 1; l >= 0; --l) {
        uint32_t parent_num_nodes = (uint32_t)upper_tree[l].node_loose_bounds.size();
        upper_tree[l].subtree_sizes.resize(parent_num_nodes);
        run_kernel_1d(count_upper_subtree_sizes, 0, (cudaStream_t)(0), parent_num_nodes, parent_num_nodes,
                      upper_tree[l].node_child_info.data().get(),
                      (l < (int)upper_tree.size() - 1) ? upper_tree[l + 1].subtree_sizes.data().get() : nullptr,
                      lower_tree[0].subtree_sizes.data().get(), upper_tree[l].subtree_sizes.data().get());
        // cuda_check(cudaDeviceSynchronize());
        // cuda_check(cudaGetLastError());
    }

    uint32_t total_storage_u32 = 0;
    thrust::copy_n(upper_tree[0].subtree_sizes.begin(), 1, &total_storage_u32);
    constexpr uint32_t root_address = 0;
    thrust::fill_n(upper_tree[0].subtree_sizes.begin(), 1, root_address);

    nodes_storage = cuda_alloc_device_low_level(total_storage_u32 * sizeof(uint32_t));
    uint32_t *node_storage_ptr_u32 = reinterpret_cast<uint32_t *>(nodes_storage.dptr);

    for (int l = 0; l < (int)upper_tree.size(); ++l) {
        uint32_t parent_num_nodes = (uint32_t)upper_tree[l].node_loose_bounds.size();
        run_kernel_1d(compact_upper_tree, 0, (cudaStream_t)(0), parent_num_nodes, parent_num_nodes,
                      upper_tree[l].node_child_info.data().get(), upper_tree[l].subtree_sizes.data().get(),
                      (l < (int)upper_tree.size() - 1) ? upper_tree[l + 1].subtree_sizes.data().get() : nullptr,
                      lower_tree[0].subtree_sizes.data().get(), node_storage_ptr_u32);
        // cuda_check(cudaDeviceSynchronize());
        // cuda_check(cudaGetLastError());
    }
    // DEBUG
    {
        thrust::host_vector<uint32_t> h2 = lower_tree[0].subtree_sizes;
        std::vector<uint32_t> v2(h2.begin(), h2.end());
        // std::string md5 = Chocobo1::MD5().addData(v2.data(), sizeof(uint32_t) * v2.size()).finalize().toString();
        // printf("small root addr: %s\n", md5.c_str());
        int debug = 0;
    }
    for (int l = 0; l < (int)lower_tree.size(); ++l) {
        uint32_t parent_num_nodes;
        if (l > 0) {
            parent_num_nodes = (uint32_t)lower_tree[l].node_loose_bounds.size();
        } else {
            parent_num_nodes = small_roots.node_prim_count_psum.size() - 1;
        }
        run_kernel_1d(compact_lower_tree, 0, (cudaStream_t)(0), parent_num_nodes, parent_num_nodes,
                      lower_tree[l].small_root_ids.data().get(), lower_tree[l].prim_masks.data().get(),
                      lower_tree[l].sah_splits.data().get(), lower_tree[l].child_offsets.data().get(),
                      lower_tree[l].subtree_sizes.data().get(), small_roots.prim_ids.data().get(),
                      small_roots.node_prim_count_psum.data().get(), small_roots.sah_split_candidates.data().get(),
                      (l < (int)lower_tree.size() - 1) ? lower_tree[l + 1].subtree_sizes.data().get() : nullptr,
                      node_storage_ptr_u32);
        // cuda_check(cudaDeviceSynchronize());
        // cuda_check(cudaGetLastError());
    }
}

AABB3 ParallelKdTree::get_total_bound() const
{
    AABB3 b;
    cuda_check(
        cudaMemcpy((void *)&b, (const void *)total_bound.dptr, sizeof(AABB3), cudaMemcpyKind::cudaMemcpyDefault));
    return b;
}

} // namespace ksc