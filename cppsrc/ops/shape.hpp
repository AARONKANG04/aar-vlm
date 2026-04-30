#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.hpp"

namespace vlm {
    Tensor reshape(const Tensor& x, std::vector<int64_t> new_shape);
    Tensor transpose(const Tensor& x, int64_t dim_a, int64_t dim_b);
    Tensor slice(const Tensor& x, int64_t dim, int64_t start, int64_t end);
    Tensor squeeze(const Tensor& x, int64_t dim);
    Tensor unsqueeze(const Tensor& x, int64_t dim);
    Tensor contiguous(const Tensor& x);

    Tensor make_view(const Tensor& src,
                     std::vector<int64_t> shape,
                     std::vector<int64_t> strides,
                     int64_t storage_offset);

    void copy_strided_to_contiguous(const Tensor& src, Tensor& dst);
}
