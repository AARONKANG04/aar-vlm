#include <limits>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "ops/attention.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor apply_causal_mask_cuda(const Tensor& x);
    Tensor causal_mask_backward_cuda(const Tensor& grad_out);

    namespace {
        void check_causal(const Tensor& x) {
            if (x.dtype != DType::Fp32) {
                throw std::invalid_argument("apply_causal_mask: only Fp32 supported");
            }
            if (x.shape.size() < 2 || 
                x.shape[x.shape.size() - 1] != x.shape[x.shape.size() - 2]) {
                    throw std::invalid_argument("apply_causal_mask: input must be a square matrix");
            }
        }

        Tensor apply_causal_mask_cpu(const Tensor& x) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            float* Y = static_cast<float*>(out.data());
            const int64_t T = x.shape.back();
            const size_t M = x.numel() / static_cast<size_t>(T * T);
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (size_t m = 0; m < M; ++m) {
                for (int64_t i = 0; i < T; ++i) {
                    for (int64_t j = 0; j < T; ++j) {
                        const size_t idx = m * T * T + i * T + j;
                        Y[idx] = (j <= i) ? X[idx] : neg_inf;
                    }
                }
            }
            return out;
        }

        Tensor causal_mask_backward_cpu(const Tensor& grad_out) {
            Tensor g = Tensor::empty(grad_out.shape, grad_out.dtype, grad_out.device);
            const float* GO = static_cast<const float*>(grad_out.data());
            float* G = static_cast<float*>(g.data());
            const int64_t T = grad_out.shape.back();
            const size_t M = grad_out.numel() / static_cast<size_t>(T * T);
            for (size_t m = 0; m < M; ++m) {
                for (int64_t i = 0; i < T; ++i) {
                    for (int64_t j = 0; j < T; ++j) {
                        const size_t idx = m * T * T + i * T + j;
                        G[idx] = (j <= i) ? GO[idx] : 0.0f;
                    }
                }
            }
            return g;
        }

        Tensor apply_causal_mask_no_grad(const Tensor& x) {
            return x.device == Device::CPU ? apply_causal_mask_cpu(x) : apply_causal_mask_cuda(x);
        }

        Tensor causal_mask_backward_no_grad(const Tensor& grad_out) {
            return grad_out.device == Device::CPU ? causal_mask_backward_cpu(grad_out) : causal_mask_backward_cuda(grad_out);
        }

        class CausalMaskFunction : public Function {
        public:
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {causal_mask_backward_no_grad(grad_output)};
            }
            const char* name() const override { return "CausalMaskFunction"; }
        };
    }

    Tensor apply_causal_mask(const Tensor& x) {
        check_causal(x);
        Tensor xc = x.is_contiguous() ? x : contiguous(x);
        if (!xc.requires_grad) {
            return apply_causal_mask_no_grad(xc);
        }
        auto fn = std::make_shared<CausalMaskFunction>();
        fn->record_input(xc);
        Tensor out = apply_causal_mask_no_grad(xc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}