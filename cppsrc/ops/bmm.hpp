#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor bmm(const Tensor& a, const Tensor& b);
    Tensor bmm_a_bt(const Tensor& a, const Tensor& b);
    Tensor bmm_at_b(const Tensor& a, const Tensor& b);
}
