#include "core/cuda_allocator.hpp"

#include <cuda_runtime.h>

#include <new>

namespace vlm {
    void* CUDAAllocator::alloc(size_t nbytes, size_t /*alignment*/) {
        if (nbytes == 0) return nullptr;
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, nbytes) != cudaSuccess) throw std::bad_alloc();
        return ptr;
    }

    void CUDAAllocator::free(void* ptr) noexcept {
        if (ptr) cudaFree(ptr);
    }

    Allocator* cuda_allocator() {
        static CUDAAllocator instance;
        return &instance;
    }
}
