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

    void scaled_add_inplace_cuda(Tensor&, const Tensor&, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor sub_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor neg_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor gelu_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor gelu_backward_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor add_bias_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor bias_grad_cuda(const Tensor&, int64_t) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
