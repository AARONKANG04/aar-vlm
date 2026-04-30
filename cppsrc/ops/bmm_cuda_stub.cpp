#include "ops/bmm.hpp"

#include <stdexcept>

namespace vlm {
    Tensor bmm_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    Tensor bmm_a_bt_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    Tensor bmm_at_b_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
