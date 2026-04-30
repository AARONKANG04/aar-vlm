#pragma once

#include "core/allocator.hpp"

namespace vlm {
    class CPUAllocator : public Allocator {
        public:
            [[nodiscard]] void* alloc(size_t bytes, size_t alignment = 64) override;
            void free(void* ptr) noexcept override;
            Device device() const noexcept override { return Device::CPU; }
    };
}