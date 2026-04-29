#pragma once

#include <cstddef>
#include "core/types.hpp"

namespace vlm {
    class Allocator;

    class Storage {
        public:
            Storage(size_t nbytes, Allocator* allocator);
            ~Storage();

            Storage(Storage&& other) noexcept;
            Storage& operator=(Storage&& other) noexcept;
            Storage(const Storage&) = delete;
            Storage& operator=(const Storage&) = delete;

            void* data() noexcept { return data_; }
            const void* data() const noexcept { return data_; }
            size_t nbytes() const noexcept { return nbytes_; }
            Device device() const noexcept;
            Allocator* allocator() const noexcept { return allocator_; }

        private:
            void* data_ = nullptr;
            size_t nbytes_ = 0;
            Allocator* allocator_ = nullptr;
    };
}