#include "ops/shape.hpp"

#include <stdexcept>

namespace vlm {
    void contiguous_cuda(const Tensor&, Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    void copy_contiguous_into_strided_cuda(const Tensor&, Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
