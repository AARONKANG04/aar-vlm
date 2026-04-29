#include "ops/softmax.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"

namespace vlm {
    Tensor softmax_cuda(const Tensor& x);
    Tensor softmax_backward_cuda(const Tensor& g, const Tensor& y);

    namespace {
        Tensor softmax_cpu(const Tensor& x) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            float* Y = static_cast<float*>(out.data());
            const int64_t last = x.shape.back();
            const int64_t outer = x.numel() / last;
            for (int64_t i = 0; i < outer; ++i) {
                const float* row = X + i * last;
                float* out_row = Y + i * last;
                float max_val = row[0];
                for (int64_t j = 1; j < last; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) {
                    out_row[j] = std::exp(row[j] - max_val);
                    sum += out_row[j];
                }
                const float inv_sum = 1.0f / sum;
                for (int64_t j = 0; j < last; ++j) out_row[j] *= inv_sum;
            }
            return out;
        }

        Tensor softmax_backward_cpu(const Tensor& g, const Tensor& y) {
            Tensor dx = Tensor::empty(g.shape, g.dtype, g.device);
            const float* G = static_cast<const float*>(g.data());
            const float* Y = static_cast<const float*>(y.data());
            float* DX = static_cast<float*>(dx.data());
            const int64_t last = g.shape.back();
            const int64_t outer = g.numel() / last;
            for (int64_t i = 0; i < outer; ++i) {
                const float* g_row = G + i * last;
                const float* y_row = Y + i * last;
                float* dx_row = DX + i * last;
                float dot = 0.0f;
                for (int64_t j = 0; j < last; ++j) dot += g_row[j] * y_row[j];
                for (int64_t j = 0; j < last; ++j) dx_row[j] = y_row[j] * (g_row[j] - dot);
            }
            return dx;
        }

        Tensor softmax_no_grad(const Tensor& x) {
            return x.device == Device::CPU ? softmax_cpu(x) : softmax_cuda(x);
        }

        Tensor softmax_backward_no_grad(const Tensor& g, const Tensor& y) {
            return g.device == Device::CPU ? softmax_backward_cpu(g, y) : softmax_backward_cuda(g, y);
        }

        class SoftmaxFunction : public Function {
        public:
            Tensor saved_y;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {softmax_backward_no_grad(grad_output, saved_y)};
            }
            const char* name() const override { return "SoftmaxFunction"; }
        };
    }

    Tensor softmax(const Tensor& x) {
        if (x.shape.empty()) {
            throw std::invalid_argument("softmax: input must have at least 1 dim");
        }
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("softmax: only Fp32 supported");
        }
        if (x.shape.back() == 0) {
            throw std::invalid_argument("softmax: last dim must be > 0");
        }
        if (!x.requires_grad) {
            return softmax_no_grad(x);
        }
        auto fn = std::make_shared<SoftmaxFunction>();
        fn->record_input(x);
        Tensor out = softmax_no_grad(x);
        fn->saved_y = out;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
