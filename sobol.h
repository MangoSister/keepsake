#pragma once

// Helmer, Andrew, Per H.Christensen, and Andrew Kensler."Stochastic Generation of (t, s) Sample Sequences."
// EGSR(DL).2021.
// https://github.com/Andrew-Helmer/stochastic-generation

#include <cstdint>

namespace ks
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
float sobol_owen(int idx, int dim, uint32_t seed, int nd);

} // namespace ks