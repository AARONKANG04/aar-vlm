#include "ops/softmax.hpp"

#include <stdexcept>

namespace vlm {
    Tensor softmax_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    Tensor softmax_backward_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
