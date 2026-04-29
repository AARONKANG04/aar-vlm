#include "core/storage.hpp"

#include <cassert>

#include "core/allocator.hpp"

namespace vlm {
    Storage::Storage(size_t nbytes, Allocator* allocator)
        : nbytes_(nbytes), allocator_(allocator) {
        assert(allocator_ != nullptr);
        data_ = allocator_->alloc(nbytes_);
    }

    Storage::~Storage() {
        if (data_) allocator_->free(data_);
    }

    Storage::Storage(Storage&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_), allocator_(other.allocator_) {
        other.data_ = nullptr;
        other.nbytes_ = 0;
        // allocator_ left set so device() still works on a moved-from Storage.
    }

    Storage& Storage::operator=(Storage&& other) noexcept {
        if (this == &other) return *this;
        if (data_) allocator_->free(data_);
        data_ = other.data_;
        nbytes_ = other.nbytes_;
        allocator_ = other.allocator_;
        other.data_ = nullptr;
        other.nbytes_ = 0;
        return *this;
    }

    Device Storage::device() const noexcept {
        return allocator_->device();
    }
}
