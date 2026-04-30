#include "core/tensor.hpp"

#include <stdexcept>

namespace vlm {
    void flash_attention_forward_cuda(const Tensor&, const Tensor&, const Tensor&,
                                      Tensor&, Tensor&, bool, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }

    void flash_attention_backward_cuda(const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, const Tensor&, const Tensor&,
                                       Tensor&, Tensor&, Tensor&, bool, float) {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
