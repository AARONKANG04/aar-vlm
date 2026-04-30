#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor flash_attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal);
}
