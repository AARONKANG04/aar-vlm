#include "ops/cross_entropy.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "core/cuda_copy.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor cross_entropy_cuda(const Tensor& logits, const Tensor& targets,
                              int64_t ignore_index);
    Tensor cross_entropy_backward_cuda(const Tensor& grad_output,
                                       const Tensor& logits,
                                       const Tensor& targets,
                                       int64_t ignore_index);

    namespace {
        Tensor cross_entropy_cpu(const Tensor& logits, const Tensor& targets,
                                  int64_t ignore_index) {
            const int64_t N = logits.shape[0];
            const int64_t V = logits.shape[1];
            const float* L = static_cast<const float*>(logits.data());
            const int64_t* T = static_cast<const int64_t*>(targets.data());
            double total = 0.0;
            int64_t active = 0;
            for (int64_t i = 0; i < N; ++i) {
                const int64_t t = T[i];
                if (t == ignore_index) continue;
                if (t < 0 || t >= V) {
                    throw std::out_of_range("cross_entropy: target out of range [0, V)");
                }
                const float* row = L + i * V;
                float m = row[0];
                for (int64_t v = 1; v < V; ++v) if (row[v] > m) m = row[v];
                float s = 0.0f;
                for (int64_t v = 0; v < V; ++v) s += std::exp(row[v] - m);
                const float lse = m + std::log(s);
                total += static_cast<double>(lse - row[t]);
                ++active;
            }
            Tensor out = Tensor::empty({}, logits.dtype, logits.device);
            const float mean = active > 0
                ? static_cast<float>(total / static_cast<double>(active))
                : 0.0f;
            *static_cast<float*>(out.data()) = mean;
            return out;
        }

        Tensor cross_entropy_backward_cpu(const Tensor& grad_output,
                                           const Tensor& logits,
                                           const Tensor& targets,
                                           int64_t ignore_index) {
            const int64_t N = logits.shape[0];
            const int64_t V = logits.shape[1];
            Tensor dL = Tensor::zeros({N, V}, logits.dtype, logits.device);
            const float* L = static_cast<const float*>(logits.data());
            const int64_t* TG = static_cast<const int64_t*>(targets.data());
            float* DL = static_cast<float*>(dL.data());
            const float go = *static_cast<const float*>(grad_output.data());

            int64_t active = 0;
            for (int64_t i = 0; i < N; ++i) {
                if (TG[i] != ignore_index) ++active;
            }
            if (active == 0) return dL;
            const float scale = go / static_cast<float>(active);

            for (int64_t i = 0; i < N; ++i) {
                const int64_t t = TG[i];
                if (t == ignore_index) continue;
                const float* row = L + i * V;
                float* drow = DL + i * V;
                float m = row[0];
                for (int64_t v = 1; v < V; ++v) if (row[v] > m) m = row[v];
                float s = 0.0f;
                for (int64_t v = 0; v < V; ++v) {
                    drow[v] = std::exp(row[v] - m);
                    s += drow[v];
                }
                const float inv_s = 1.0f / s;
                for (int64_t v = 0; v < V; ++v) {
                    drow[v] = (drow[v] * inv_s) * scale;
                }
                drow[t] -= scale;
            }
            return dL;
        }

        Tensor cross_entropy_no_grad(const Tensor& logits, const Tensor& targets,
                                      int64_t ignore_index) {
            return logits.device == Device::CPU
                ? cross_entropy_cpu(logits, targets, ignore_index)
                : cross_entropy_cuda(logits, targets, ignore_index);
        }

        Tensor cross_entropy_backward_no_grad(const Tensor& grad_output,
                                               const Tensor& logits,
                                               const Tensor& targets,
                                               int64_t ignore_index) {
            return logits.device == Device::CPU
                ? cross_entropy_backward_cpu(grad_output, logits, targets, ignore_index)
                : cross_entropy_backward_cuda(grad_output, logits, targets, ignore_index);
        }

        class CrossEntropyFunction : public Function {
        public:
            Tensor saved_logits;
            Tensor saved_targets;
            int64_t ignore_index = -100;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dL = cross_entropy_backward_no_grad(grad_output, saved_logits,
                                                            saved_targets, ignore_index);
                return {dL};
            }
            const char* name() const override { return "CrossEntropyFunction"; }
        };
    }

    Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                          int64_t ignore_index) {
        if (logits.dtype != DType::Fp32) {
            throw std::invalid_argument("cross_entropy: logits must be Fp32");
        }
        if (targets.dtype != DType::Int64) {
            throw std::invalid_argument("cross_entropy: targets must be Int64");
        }
        if (logits.shape.size() != 2) {
            throw std::invalid_argument("cross_entropy: logits must be 2D (N, V)");
        }
        if (targets.shape.size() != 1 || targets.shape[0] != logits.shape[0]) {
            throw std::invalid_argument("cross_entropy: targets must be 1D matching N");
        }
        if (logits.device != targets.device) {
            throw std::invalid_argument("cross_entropy: device mismatch");
        }
        Tensor lc = logits.is_contiguous() ? logits : contiguous(logits);
        Tensor tc = targets.is_contiguous() ? targets : contiguous(targets);
        if (!lc.requires_grad) {
            return cross_entropy_no_grad(lc, tc, ignore_index);
        }
        auto fn = std::make_shared<CrossEntropyFunction>();
        fn->record_input(lc);
        fn->saved_logits = lc;
        fn->saved_targets = tc;
        fn->ignore_index = ignore_index;
        Tensor out = cross_entropy_no_grad(lc, tc, ignore_index);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
