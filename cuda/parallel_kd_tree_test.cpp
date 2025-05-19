
#include "../config.h"
#include "../file_util.h"
#include "../log_util.h"
#include "parallel_kd_tree.h"
#include "rng.cuh"
#include <thrust/host_vector.h>

namespace ks
{

// std::vector<uint32_t> arr = {0, 100, 200, 300, 400};
// uint32_t gid = 50;
// uint32_t ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });
// gid = 0;
// ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });
// gid = 300;
// ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });
// gid = 350;
// ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });
// gid = 400;
// ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });
// gid = 999;
// ret = find_interval((uint32_t)arr.size(), [&](uint32_t i) { return gid >= arr[i]; });

void parallel_kd_tree_test(const ConfigArgs &args, const fs::path &task_dir, int task_id)
{
    // compare cpu serial to cuda parallel results
    constexpr uint32_t num_prims = 85;
    thrust::host_vector<ksc::AABB3> bounds(num_prims);
    ksc::RNG rng;
    for (uint32_t i = 0; i < num_prims; ++i) {
        bounds[i].min = ksc::_lerp(ksc::vec3(-10.0f), ksc::vec3(10.0f), rng.Uniform<ksc::vec3>());
        bounds[i].max = ksc::_lerp(ksc::vec3(-10.0f), ksc::vec3(10.0f), rng.Uniform<ksc::vec3>());
        for (int j = 0; j < 3; ++j) {
            if (bounds[i].min[j] > bounds[i].max[j]) {
                std::swap(bounds[i].min[j], bounds[i].max[j]);
            }
        }
    }
    for (uint32_t i = 0; i < num_prims; ++i) {
        bounds[i] = bounds[0];
    }
    printf("(%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f)\n", bounds[0].min.x, bounds[0].min.y, bounds[0].min.z,
           bounds[0].max.x, bounds[0].max.y, bounds[0].max.z);

    constexpr uint32_t CHUNK_SIZE = 256;
    uint32_t num_chunks = (bounds.size() + (CHUNK_SIZE - 1)) / CHUNK_SIZE;
    constexpr uint32_t num_nodes = 1;
    thrust::host_vector<ksc::AABB3> chunk_bounds(num_chunks);
    thrust::host_vector<ksc::AABB3> node_tight_bounds(num_nodes);
    for (uint32_t i = 0; i < num_prims; ++i) {
        uint32_t chunk_idx = i / CHUNK_SIZE;
        chunk_bounds[chunk_idx].expand(bounds[i]);
        constexpr uint32_t node_idx = 0;
        node_tight_bounds[node_idx].expand(bounds[i]);
    }

    thrust::device_vector<ksc::AABB3> device_bounds;
    device_bounds = bounds;

    // ksc::ParallelKdTreeBuildInput build_input{.bounds = device_bounds};
    // ksc::ParallelKdTree tree;
    // ksc::ParallelKdTreeBuildStats stats;
    // build_input.stats = &stats;
    // tree.build(build_input);
    // get_default_logger().info("Parallel kd-tree build stats:\n{}", stats.to_string());

    // thrust::host_vector<ksc::AABB3> parallel_chunk_bounds = out.large_nodes.chunk_bounds;
    // thrust::host_vector<ksc::AABB3> parallel_node_tight_bounds = out.large_nodes.node_tight_bounds;
    // for (uint32_t i = 0; i < num_chunks; ++i) {
    //     ASSERT(chunk_bounds[i].min == parallel_chunk_bounds[i].min);
    //     ASSERT(chunk_bounds[i].max == parallel_chunk_bounds[i].max);
    // }
    // for (uint32_t i = 0; i < num_nodes; ++i) {
    //     ASSERT(node_tight_bounds[i].min == parallel_node_tight_bounds[i].min);
    //     ASSERT(node_tight_bounds[i].max == parallel_node_tight_bounds[i].max);
    // }

    return;
}

} // namespace ks