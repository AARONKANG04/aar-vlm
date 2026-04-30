#include "ops/cross_entropy.hpp"

#include <stdexcept>

namespace vlm {
    Tensor cross_entropy_cuda(const Tensor&, const Tensor&, int64_t) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    Tensor cross_entropy_backward_cuda(const Tensor&, const Tensor&,
                                       const Tensor&, int64_t) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
