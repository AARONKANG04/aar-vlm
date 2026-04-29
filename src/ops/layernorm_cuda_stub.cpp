#include "ops/layernorm.hpp"

#include <stdexcept>

namespace vlm {
    Tensor layernorm_cuda(const Tensor&, const Tensor&, const Tensor&, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
