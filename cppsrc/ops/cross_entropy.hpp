#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace vlm {
    Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                         int64_t ignore_index = -100);
}
