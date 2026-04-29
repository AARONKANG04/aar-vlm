#include "ops/matmul.hpp"

#include <stdexcept>

namespace vlm {
    Tensor matmul_cuda(const Tensor&, const Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
