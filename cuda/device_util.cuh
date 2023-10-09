#pragma once
#include "vecmath.cuh"

namespace ksc
{

constexpr uint32_t n_threads_1d_default = 256;
constexpr uint32_t n_threads_2d_default = 16;

template <typename T>
constexpr CUDA_HOST_DEVICE uint32_t n_blocks_1d(T n_elements, uint32_t n_threads = n_threads_1d_default)
{
    return (uint32_t)ceil((float)n_elements / (float)n_threads);
}

template <uint32_t N_THREADS = n_threads_1d_default, typename K, typename... Types>
inline void run_kernel_1d(K kernel, uint32_t shmem_size, cudaStream_t stream, uint32_t n_elements, Types &&...args)
{
    KSC_ASSERT(n_elements > 0);
    kernel<<<n_blocks_1d(n_elements, N_THREADS), N_THREADS, shmem_size, stream>>>(std::forward<Types>(args)...);
}

template <uint32_t N_THREADS = n_threads_2d_default, typename K, typename... Types>
inline void run_kernel_2d(K kernel, uint32_t shmem_size, cudaStream_t stream, vec2i n_elements, Types &&...args)
{
    KSC_ASSERT(n_elements.x > 0 && n_elements.y > 0);
    dim3 threads_per_block(N_THREADS, N_THREADS, 1);
    dim3 num_blocks(n_blocks_1d(n_elements.x, N_THREADS), n_blocks_1d(n_elements.y, N_THREADS), 1);
    kernel<<<num_blocks, threads_per_block, shmem_size, stream>>>(std::forward<Types>(args)...);
}

} // namespace ksc