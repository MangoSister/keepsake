#pragma once

#include "basic.cuh"
#include <cstdint>

namespace ksc
{

// Helmer's stateless iterative stochastic Owen-scrambled Sobol:
// Support up to 64 dimensions

// Query for an arbitrary coordinate from a scrambled Sobol (0,2)-sequence. Seed
// can be the same across all dimensions, or it can be different for each
// dimension.
//
// nd is the number of dimensions shared by the same seed value, to get separate
// hashed values for each sample index. If a different seed is provided for each
// dimension, this can be used with nd=1.

CUDA_HOST_DEVICE
float sobol_owen(int idx, int dim, uint32_t seed, int nd);

} // namespace ksc