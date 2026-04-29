#include "ops/layernorm.hpp"

#include <stdexcept>

namespace vlm {
    Tensor layernorm_cuda(const Tensor&, const Tensor&, const Tensor&, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    void layernorm_with_stats_cuda(const Tensor&, const Tensor&, const Tensor&, float,
                                    Tensor&, Tensor&, Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    void layernorm_backward_cuda(const Tensor&, const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&,
                                  Tensor&, Tensor&, Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
