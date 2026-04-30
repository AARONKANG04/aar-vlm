#include "ops/elementwise.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "core/cuda_copy.hpp"

namespace vlm {
    Tensor add_cuda(const Tensor& a, const Tensor& b);
    Tensor mul_cuda(const Tensor& a, const Tensor& b);
    Tensor relu_cuda(const Tensor& a);
    Tensor sum_all_cuda(const Tensor& a);
    Tensor relu_backward_cuda(const Tensor& grad_out, const Tensor& x);
    void fill_cuda(Tensor& out, float v);

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

        Tensor sum_all_cpu(const Tensor& a) {
            Tensor out = Tensor::empty({}, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            float acc = 0.0f;
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) acc += A[i];
            *static_cast<float*>(out.data()) = acc;
            return out;
        }

        Tensor add_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? add_cpu(a, b) : add_cuda(a, b);
        }

        Tensor mul_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? mul_cpu(a, b) : mul_cuda(a, b);
        }

        Tensor relu_backward_cpu(const Tensor& grad_out, const Tensor& x) {
            Tensor g = Tensor::empty(grad_out.shape, grad_out.dtype, grad_out.device);
            const float* go = static_cast<const float*>(grad_out.data());
            const float* xp = static_cast<const float*>(x.data());
            float* gp = static_cast<float*>(g.data());
            const size_t n = grad_out.numel();
            for (size_t i = 0; i < n; ++i) gp[i] = xp[i] > 0.0f ? go[i] : 0.0f;
            return g;
        }

        Tensor relu_backward_no_grad(const Tensor& grad_out, const Tensor& x) {
            return grad_out.device == Device::CPU ? relu_backward_cpu(grad_out, x)
                                                  : relu_backward_cuda(grad_out, x);
        }

        class AddFunction : public Function {
        public:
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {grad_output, grad_output};
            }
            const char* name() const override { return "AddFunction"; }
        };

        class MulFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {mul_no_grad(grad_output, saved_b), mul_no_grad(grad_output, saved_a)};
            }
            const char* name() const override { return "MulFunction"; }
        };

        class ReluFunction : public Function {
        public:
            Tensor saved_x;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {relu_backward_no_grad(grad_output, saved_x)};
            }
            const char* name() const override { return "ReluFunction"; }
        };

        class SumAllFunction : public Function {
        public:
            std::vector<int64_t> input_shape;
            DType input_dtype = DType::Fp32;
            Device input_device = Device::CPU;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor g = Tensor::empty(input_shape, input_dtype, input_device);
                float v;
                copy_bytes(&v, Device::CPU, grad_output.data(), grad_output.device, sizeof(float));
                if (input_device == Device::CPU) {
                    float* p = static_cast<float*>(g.data());
                    const size_t n = g.numel();
                    for (size_t i = 0; i < n; ++i) p[i] = v;
                } else {
                    fill_cuda(g, v);
                }
                return {g};
            }
            const char* name() const override { return "SumAllFunction"; }
        };
    }

    Tensor add(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "add");
        if (!a.requires_grad && !b.requires_grad) {
            return add_no_grad(a, b);
        }
        auto fn = std::make_shared<AddFunction>();
        fn->record_input(a);
        fn->record_input(b);
        Tensor out = add_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor mul(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "mul");
        if (!a.requires_grad && !b.requires_grad) {
            return mul_no_grad(a, b);
        }
        auto fn = std::make_shared<MulFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = mul_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor relu(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("relu: only Fp32 supported");
        }
        if (!a.requires_grad) {
            return a.device == Device::CPU ? relu_cpu(a) : relu_cuda(a);
        }
        auto fn = std::make_shared<ReluFunction>();
        fn->record_input(a);
        fn->saved_x = a;
        Tensor out = a.device == Device::CPU ? relu_cpu(a) : relu_cuda(a);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor sum_all(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("sum_all: only Fp32 supported");
        }
        if (!a.requires_grad) {
            return a.device == Device::CPU ? sum_all_cpu(a) : sum_all_cuda(a);
        }
        auto fn = std::make_shared<SumAllFunction>();
        fn->record_input(a);
        fn->input_shape = a.shape;
        fn->input_dtype = a.dtype;
        fn->input_device = a.device;
        Tensor out = a.device == Device::CPU ? sum_all_cpu(a) : sum_all_cuda(a);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
