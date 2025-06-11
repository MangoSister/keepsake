#pragma once
#include "aabb.h"
#include "maths.h"
#include <array>

namespace ks
{

int tri_box_overlap(const std::array<ks::vec3, 3> &tri, const ks::AABB3 &box);

}