#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
}
