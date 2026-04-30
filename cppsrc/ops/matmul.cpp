#include "ops/matmul.hpp"

#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor matmul_cuda(const Tensor& a, const Tensor& b);
    Tensor matmul_a_bt_cuda(const Tensor& a, const Tensor& b);
    Tensor matmul_at_b_cuda(const Tensor& a, const Tensor& b);

    namespace {
        Tensor as_view(const Tensor& src, std::vector<int64_t> new_shape) {
            auto strides = compute_contiguous_strides(new_shape);
            return make_view(src, std::move(new_shape), std::move(strides), src.storage_offset);
        }

        Tensor as_contig(const Tensor& t) {
            return t.is_contiguous() ? t : contiguous(t);
        }

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
            if (a.shape.size() == 2) {
                return a.device == Device::CPU ? matmul_cpu(a, b) : matmul_cuda(a, b);
            }
            const int64_t K = a.shape.back();
            const int64_t N = b.shape[1];
            const int64_t M_flat = static_cast<int64_t>(a.numel()) / K;
            Tensor a2d = as_view(a, {M_flat, K});
            Tensor out2d = a.device == Device::CPU ? matmul_cpu(a2d, b) : matmul_cuda(a2d, b);
            std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
            out_shape.push_back(N);
            return as_view(out2d, std::move(out_shape));
        }

        Tensor matmul_a_bt_no_grad(const Tensor& a, const Tensor& b) {
            if (a.shape.size() == 2) {
                return a.device == Device::CPU ? matmul_a_bt_cpu(a, b) : matmul_a_bt_cuda(a, b);
            }
            const int64_t K = a.shape.back();
            const int64_t N = b.shape[0];
            const int64_t M_flat = static_cast<int64_t>(a.numel()) / K;
            Tensor a2d = as_view(a, {M_flat, K});
            Tensor out2d = a.device == Device::CPU ? matmul_a_bt_cpu(a2d, b)
                                                   : matmul_a_bt_cuda(a2d, b);
            std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
            out_shape.push_back(N);
            return as_view(out2d, std::move(out_shape));
        }

        Tensor matmul_at_b_no_grad(const Tensor& a, const Tensor& b) {
            if (a.shape.size() == 2 && b.shape.size() == 2) {
                return a.device == Device::CPU ? matmul_at_b_cpu(a, b) : matmul_at_b_cuda(a, b);
            }
            const int64_t M = a.shape.back();
            const int64_t N = b.shape.back();
            const int64_t K_flat = static_cast<int64_t>(a.numel()) / M;
            Tensor a2d = as_view(a, {K_flat, M});
            Tensor b2d = as_view(b, {K_flat, N});
            return a.device == Device::CPU ? matmul_at_b_cpu(a2d, b2d)
                                           : matmul_at_b_cuda(a2d, b2d);
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
        if (a.shape.size() < 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul: LHS rank must be >= 2 and RHS must be 2D");
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
        if (b.shape[0] != a.shape.back()) {
            throw std::invalid_argument("matmul: inner dimensions mismatch");
        }
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return matmul_no_grad(ac, bc);
        }
        auto fn = std::make_shared<MatmulFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        fn->saved_a = ac;
        fn->saved_b = bc;
        Tensor out = matmul_no_grad(ac, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor matmul_a_bt(const Tensor& a, const Tensor& b) {
        if (a.shape.size() < 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul_a_bt: LHS rank must be >= 2 and RHS must be 2D");
        }
        if (a.dtype != b.dtype || a.device != b.device) {
            throw std::invalid_argument("matmul_a_bt: dtype/device mismatch");
        }
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("matmul_a_bt: only Fp32 supported");
        }
        if (a.shape.back() != b.shape[1]) {
            throw std::invalid_argument("matmul_a_bt: contraction dim mismatch");
        }
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return matmul_a_bt_no_grad(ac, bc);
        }
        auto fn = std::make_shared<MatmulABTFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        fn->saved_a = ac;
        fn->saved_b = bc;
        Tensor out = matmul_a_bt_no_grad(ac, bc);
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
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return matmul_at_b_no_grad(ac, bc);
        }
        auto fn = std::make_shared<MatmulATBFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        fn->saved_a = ac;
        fn->saved_b = bc;
        Tensor out = matmul_at_b_no_grad(ac, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
