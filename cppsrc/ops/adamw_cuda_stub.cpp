#include "ops/adamw.hpp"

#include <stdexcept>

namespace vlm {
    void adamw_step_cuda(Tensor&, const Tensor&, Tensor&, Tensor&,
                         float, float, float, float, float, float, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
