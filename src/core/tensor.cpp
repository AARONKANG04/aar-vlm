#include "core/tensor.hpp"

#include <cstring>
#include <stdexcept>

#include "autograd/backward.hpp"
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
            zero_bytes(t.storage->data(), device, t.storage->nbytes());
        }
        return t;
    }

    Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype, Device device) {
        Tensor cpu = empty(shape, dtype, Device::CPU);
        if (cpu.storage) {
            float* p = static_cast<float*>(cpu.storage->data());
            const size_t n = cpu.numel();
            for (size_t i = 0; i < n; ++i) p[i] = 1.0f;
        }
        if (device == Device::CPU) return cpu;
        return cpu.to(device);
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

    Tensor& Tensor::set_requires_grad(bool req) {
        if (req && !is_leaf()) {
            throw std::runtime_error("set_requires_grad: cannot set on non-leaf tensor");
        }
        requires_grad = req;
        if (req && !grad_slot) {
            grad_slot = std::make_shared<std::shared_ptr<Tensor>>();
        }
        return *this;
    }

    std::shared_ptr<Tensor> Tensor::grad() const {
        return grad_slot ? *grad_slot : nullptr;
    }

    void Tensor::zero_grad() {
        if (grad_slot) *grad_slot = nullptr;
    }

    void Tensor::backward() {
        if (numel() != 1) {
            throw std::runtime_error("backward(): grad implicitly created only for scalar outputs");
        }
        Tensor g = Tensor::ones(shape, dtype, device);
        run_backward(*this, g);
    }

    void Tensor::backward(const Tensor& grad_output) {
        if (grad_output.shape != shape) {
            throw std::runtime_error("backward(grad_output): shape mismatch");
        }
        run_backward(*this, grad_output);
    }
}
