#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor matmul_a_bt(const Tensor& a, const Tensor& b);
    Tensor matmul_at_b(const Tensor& a, const Tensor& b);
}
