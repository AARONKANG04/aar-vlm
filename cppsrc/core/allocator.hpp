#pragma once

#include <cstddef>
#include <cstdint>
#include "core/types.hpp"

namespace vlm {
    class Allocator {
        public:
            virtual ~Allocator() = default;

            [[nodiscard]] virtual void* alloc(size_t bytes, size_t alignment = 64) = 0;
            virtual void free(void* ptr) noexcept = 0;
            virtual Device device() const noexcept = 0;

            Allocator(const Allocator&) = delete;
            Allocator& operator=(const Allocator&) = delete;
            Allocator(Allocator&&) = delete;
            Allocator& operator=(Allocator&&) = delete;

        protected:
            Allocator() = default;
    };

    // Returns the process-wide CPU allocator instance.
    Allocator* cpu_allocator();
}