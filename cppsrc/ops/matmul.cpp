#include "ops/matmul.hpp"

#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"

namespace vlm {
    Tensor matmul_cuda(const Tensor& a, const Tensor& b);
    Tensor matmul_a_bt_cuda(const Tensor& a, const Tensor& b);
    Tensor matmul_at_b_cuda(const Tensor& a, const Tensor& b);

    namespace {
        Tensor matmul_cpu(const Tensor& a, const Tensor& b) {
            const int64_t M = a.shape[0];
            const int64_t K = a.shape[1];
            const int64_t N = b.shape[1];

            Tensor out = Tensor::empty({M, N}, a.dtype, a.device);

            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = acc;
                }
            }
            return out;
        }

        Tensor matmul_a_bt_cpu(const Tensor& a, const Tensor& b) {
            const int64_t M = a.shape[0];
            const int64_t K = a.shape[1];
            const int64_t N = b.shape[0];

            Tensor out = Tensor::empty({M, N}, a.dtype, a.device);

            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += A[i * K + k] * B[j * K + k];
                    }
                    C[i * N + j] = acc;
                }
            }
            return out;
        }

        Tensor matmul_at_b_cpu(const Tensor& a, const Tensor& b) {
            const int64_t K = a.shape[0];
            const int64_t M = a.shape[1];
            const int64_t N = b.shape[1];

            Tensor out = Tensor::empty({M, N}, a.dtype, a.device);

            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += A[k * M + i] * B[k * N + j];
                    }
                    C[i * N + j] = acc;
                }
            }
            return out;
        }

        Tensor matmul_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? matmul_cpu(a, b) : matmul_cuda(a, b);
        }

        Tensor matmul_a_bt_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? matmul_a_bt_cpu(a, b) : matmul_a_bt_cuda(a, b);
        }

        Tensor matmul_at_b_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? matmul_at_b_cpu(a, b) : matmul_at_b_cuda(a, b);
        }

        class MatmulFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = matmul_a_bt_no_grad(grad_output, saved_b);
                Tensor dB = matmul_at_b_no_grad(saved_a, grad_output);
                return {dA, dB};
            }
            const char* name() const override { return "MatmulFunction"; }
        };

        class MatmulABTFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = matmul_no_grad(grad_output, saved_b);
                Tensor dB = matmul_at_b_no_grad(grad_output, saved_a);
                return {dA, dB};
            }
            const char* name() const override { return "MatmulABTFunction"; }
        };

        class MatmulATBFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = matmul_a_bt_no_grad(saved_b, grad_output);
                Tensor dB = matmul_no_grad(saved_a, grad_output);
                return {dA, dB};
            }
            const char* name() const override { return "MatmulATBFunction"; }
        };
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        if (a.shape.size() != 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul: inputs must be 2D");
        }
        if (a.dtype != b.dtype) {
            throw std::invalid_argument("matmul: dtype mismatch");
        }
        if (a.device != b.device) {
            throw std::invalid_argument("matmul: device mismatch");
        }
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("matmul: only Fp32 supported");
        }
        if (b.shape[0] != a.shape[1]) {
            throw std::invalid_argument("matmul: inner dimensions mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return matmul_no_grad(a, b);
        }
        auto fn = std::make_shared<MatmulFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = matmul_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor matmul_a_bt(const Tensor& a, const Tensor& b) {
        if (a.shape.size() != 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul_a_bt: inputs must be 2D");
        }
        if (a.dtype != b.dtype || a.device != b.device) {
            throw std::invalid_argument("matmul_a_bt: dtype/device mismatch");
        }
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("matmul_a_bt: only Fp32 supported");
        }
        if (a.shape[1] != b.shape[1]) {
            throw std::invalid_argument("matmul_a_bt: contraction dim mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return matmul_a_bt_no_grad(a, b);
        }
        auto fn = std::make_shared<MatmulABTFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = matmul_a_bt_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor matmul_at_b(const Tensor& a, const Tensor& b) {
        if (a.shape.size() != 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul_at_b: inputs must be 2D");
        }
        if (a.dtype != b.dtype || a.device != b.device) {
            throw std::invalid_argument("matmul_at_b: dtype/device mismatch");
        }
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("matmul_at_b: only Fp32 supported");
        }
        if (a.shape[0] != b.shape[0]) {
            throw std::invalid_argument("matmul_at_b: contraction dim mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return matmul_at_b_no_grad(a, b);
        }
        auto fn = std::make_shared<MatmulATBFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = matmul_at_b_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
