#include "core/cuda_allocator.hpp"

#include <stdexcept>

namespace vlm {
    Allocator* cuda_allocator() {
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
