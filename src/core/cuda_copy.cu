#include "core/cuda_copy.hpp"

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    void copy_bytes(void* dst, Device dst_device,
                    const void* src, Device src_device,
                    size_t nbytes) {
        if (nbytes == 0) return;

        cudaMemcpyKind kind;
        if (src_device == Device::CPU && dst_device == Device::CPU) {
            std::memcpy(dst, src, nbytes);
            return;
        } else if (src_device == Device::CPU && dst_device == Device::CUDA) {
            kind = cudaMemcpyHostToDevice;
        } else if (src_device == Device::CUDA && dst_device == Device::CPU) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyDeviceToDevice;
        }
        CUDA_CHECK(cudaMemcpy(dst, src, nbytes, kind));
    }
}
