#pragma once

#include "core/allocator.hpp"

namespace vlm {
    class CUDAAllocator final : public Allocator {
    public:
        [[nodiscard]] void* alloc(size_t nbytes, size_t alignment = 64) override;
        void free(void* ptr) noexcept override;
        Device device() const noexcept override { return Device::CUDA; }
    };

    Allocator* cuda_allocator();
}
