#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor apply_causal_mask(const Tensor& x);
}