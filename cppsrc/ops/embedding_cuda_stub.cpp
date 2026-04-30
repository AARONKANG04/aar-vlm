#include "ops/embedding.hpp"

#include <stdexcept>

namespace vlm {
    Tensor embedding_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
    Tensor embedding_backward_cuda(const Tensor&, const Tensor&, int64_t) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
