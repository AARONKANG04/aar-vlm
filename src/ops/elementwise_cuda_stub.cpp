#include "ops/elementwise.hpp"

#include <stdexcept>

namespace vlm {
    Tensor add_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor mul_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor relu_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor sum_all_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor relu_backward_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    void fill_cuda(Tensor&, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
