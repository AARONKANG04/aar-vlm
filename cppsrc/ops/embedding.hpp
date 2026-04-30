#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor embedding(const Tensor& weight, const Tensor& ids);
}
