#include "ops/shape.hpp"

#include <stdexcept>

namespace vlm {
    Tensor transpose_cuda(const Tensor&, int64_t, int64_t) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
