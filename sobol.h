#pragma once

// Sobol sequence with Owen scrambling without precomputed tables.
// Based on Brent Burley, Practical Hash-based Owen Scrambling, Journal of Computer Graphics Techniques (JCGT), vol. 9,
// no. 4, 1-20, 2020
// http://jcgt.org/published/0009/04/01/

#include "namespace.h"
#include <cstdint>

KS_NAMESPACE_BEGIN

// TODO: can be vectorized, etc.
float sobol_owen(uint32_t index, uint32_t dim, uint32_t seed);

KS_NAMESPACE_END