#pragma once

#include "core/tensor.hpp"

namespace vlm {
    void run_backward(const Tensor& root, const Tensor& grad_output);
}
