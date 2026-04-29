#include "core/tensor.hpp"

#include <cstring>
#include <stdexcept>

#include "core/allocator.hpp"
#include "core/cuda_allocator.hpp"
#include "core/cuda_copy.hpp"

namespace vlm {
    namespace {
        Allocator* allocator_for(Device device) {
            switch (device) {
                case Device::CPU: return cpu_allocator();
                case Device::CUDA: return cuda_allocator();
            }
            throw std::runtime_error("Unknown device");
        }
    }

    size_t Tensor::numel() const {
        size_t n = 1;
        for (int64_t dim : shape) {
            n *= static_cast<size_t>(dim);
        }
        return n;
    }

    size_t Tensor::nbytes() const {
        return numel() * dtype_size(dtype);
    }

    Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype, Device device) {
        Tensor t(std::move(shape), dtype, device);
        if (t.numel() > 0) {
            t.storage = std::make_shared<Storage>(t.nbytes(), allocator_for(device));
        }
        return t;
    }

    Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, Device device) {
        Tensor t = empty(std::move(shape), dtype, device);
        if (t.storage) {
            if (device == Device::CPU) {
                std::memset(t.storage->data(), 0, t.storage->nbytes());
            } else {
                throw std::runtime_error("zeros() on non-CPU device not yet implemented");
            }
        }
        return t;
    }

    Tensor Tensor::to(Device target) const {
        if (target == device) return *this;
        Tensor out = Tensor::empty(shape, dtype, target);
        copy_bytes(out.storage->data(), target,
                   storage ? storage->data() : nullptr, device,
                   nbytes());
        return out;
    }

    void* Tensor::data() {
        return storage ? storage->data() : nullptr;
    }

    const void* Tensor::data() const {
        return storage ? storage->data() : nullptr;
    }
}
