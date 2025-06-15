#include "cuda_parallel_kd_tree.h"
#include "../file_util.h"

namespace ksc
{

std::string CudaParallelKdTreeBuildStats::to_string() const
{
    // Full binary tree.
    uint32_t n_interior = n_leaves - 1;
    float avg_prim_per_leaf = (float)(n_prim_refs) / (float)(n_leaves);
    return ks::string_format("{\n"
                             "    [compact storage]: %llu (bytes),\n"
                             "    [max depth]: %u <= %u (upper) + %u (lower),\n"
                             "    [nodes]: %u = %u (interior) + %u (leaves),\n"
                             "    [small roots]: %u,\n"
                             "    [primitive refs]: %u,\n"
                             "    [avg prim/leaf]: %.1f\n"
                             "}",
                             compact_strorage_bytes, max_depth, upper_max_depth, lower_max_depth, n_nodes, n_interior,
                             n_leaves, n_small_roots, n_prim_refs, avg_prim_per_leaf);
}

} // namespace ksc