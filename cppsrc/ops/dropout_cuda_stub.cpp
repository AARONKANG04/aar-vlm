#include "ops/dropout.hpp"

#include <stdexcept>

namespace vlm {
    void dropout_cuda(const Tensor&, float, uint64_t, Tensor&, Tensor&) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
