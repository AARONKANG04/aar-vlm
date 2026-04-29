#include "core/cuda_copy.hpp"

#include <cstring>
#include <stdexcept>

namespace vlm {
    void copy_bytes(void* dst, Device dst_device,
                    const void* src, Device src_device,
                    size_t nbytes) {
        if (nbytes == 0) return;
        if (src_device == Device::CPU && dst_device == Device::CPU) {
            std::memcpy(dst, src, nbytes);
            return;
        }
        throw std::runtime_error("CUDA support not built; rebuild with CUDA");
    }
}
