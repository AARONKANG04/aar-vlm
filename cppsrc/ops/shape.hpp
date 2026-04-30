#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"

namespace vlm {
    Tensor reshape(const Tensor& x, std::vector<int64_t> new_shape);
    Tensor transpose(const Tensor& x, int64_t dim_a, int64_t dim_b);
}
