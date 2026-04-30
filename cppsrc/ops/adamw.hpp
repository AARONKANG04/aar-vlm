#pragma once

#include "core/tensor.hpp"

namespace vlm {
    void adamw_step(Tensor& param, const Tensor& grad,
                    Tensor& m, Tensor& v,
                    float lr, float beta1, float beta2,
                    float eps, float weight_decay,
                    float bias_correction1, float bias_correction2);
}
