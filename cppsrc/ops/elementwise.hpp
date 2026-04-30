#pragma once

#include "core/tensor.hpp"

namespace vlm {
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor relu(const Tensor& a);
    Tensor sum_all(const Tensor& a);

    void scaled_add_inplace(Tensor& dst, const Tensor& src, float alpha);
}
