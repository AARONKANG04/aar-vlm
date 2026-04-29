#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "core/storage.hpp"
#include "core/types.hpp"

namespace vlm {
    class Function;

    struct Tensor {
        std::vector<int64_t> shape;
        DType dtype = DType::Fp32;
        Device device = Device::CPU;
        std::shared_ptr<Storage> storage;

        bool requires_grad = false;
        std::shared_ptr<Function> grad_fn;
        std::shared_ptr<std::shared_ptr<Tensor>> grad_slot;

        Tensor() = default;
        Tensor(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU)
            : shape(std::move(shape)), dtype(dtype), device(device) {}

        static Tensor empty(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);
        static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);
        static Tensor ones(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);

        size_t numel() const;
        size_t nbytes() const;

        void* data();
        const void* data() const;

        Tensor to(Device target) const;

        Tensor& set_requires_grad(bool req);
        bool is_leaf() const { return grad_fn == nullptr; }
        std::shared_ptr<Tensor> grad() const;
        void zero_grad();
        void backward();
        void backward(const Tensor& grad_output);
    };
}
