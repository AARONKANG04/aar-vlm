#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/storage.hpp"
#include "core/types.hpp"

namespace vlm {
    struct Tensor {
        std::vector<int64_t> shape;
        DType dtype;
        Device device;
        std::shared_ptr<Storage> storage;

        Tensor(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU)
            : shape(std::move(shape)), dtype(dtype), device(device) {}

        static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);
        static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);

        size_t numel() const;
        size_t nbytes() const;

        void* data();
        const void* data() const;

        Tensor to(Device target) const;
    };
}
