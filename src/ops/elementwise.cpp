#include "ops/elementwise.hpp"

#include <algorithm>
#include <stdexcept>

namespace vlm {
    Tensor add_cuda(const Tensor& a, const Tensor& b);
    Tensor mul_cuda(const Tensor& a, const Tensor& b);
    Tensor relu_cuda(const Tensor& a);

    namespace {
        void check_binary(const Tensor& a, const Tensor& b, const char* op) {
            if (a.shape != b.shape) {
                throw std::invalid_argument(std::string(op) + ": shape mismatch");
            }
            if (a.dtype != b.dtype) {
                throw std::invalid_argument(std::string(op) + ": dtype mismatch");
            }
            if (a.device != b.device) {
                throw std::invalid_argument(std::string(op) + ": device mismatch");
            }
            if (a.dtype != DType::Fp32) {
                throw std::invalid_argument(std::string(op) + ": only Fp32 supported");
            }
        }

        Tensor add_cpu(const Tensor& a, const Tensor& b) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) C[i] = A[i] + B[i];
            return out;
        }

        Tensor mul_cpu(const Tensor& a, const Tensor& b) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) C[i] = A[i] * B[i];
            return out;
        }

        Tensor relu_cpu(const Tensor& a) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) C[i] = std::max(0.0f, A[i]);
            return out;
        }
    }

    Tensor add(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "add");
        switch (a.device) {
            case Device::CPU: return add_cpu(a, b);
            case Device::CUDA: return add_cuda(a, b);
        }
        throw std::runtime_error("add: unsupported device");
    }

    Tensor mul(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "mul");
        switch (a.device) {
            case Device::CPU: return mul_cpu(a, b);
            case Device::CUDA: return mul_cuda(a, b);
        }
        throw std::runtime_error("mul: unsupported device");
    }

    Tensor relu(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("relu: only Fp32 supported");
        }
        switch (a.device) {
            case Device::CPU: return relu_cpu(a);
            case Device::CUDA: return relu_cuda(a);
        }
        throw std::runtime_error("relu: unsupported device");
    }
}
