#include "ops/attention.hpp"

#include <stdexcept>

namespace vlm {
    Tensor apply_causal_mask_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    Tensor causal_mask_backward_cuda(const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
