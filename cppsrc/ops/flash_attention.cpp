#include "ops/flash_attention.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "ops/attention.hpp"
#include "ops/bmm.hpp"
#include "ops/elementwise.hpp"
#include "ops/shape.hpp"
#include "ops/softmax.hpp"

namespace vlm {
    void flash_attention_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v,
                                      Tensor& out, Tensor& lse, bool causal, float scale);

    void flash_attention_backward_cuda(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& o, const Tensor& lse,
                                       const Tensor& grad_out,
                                       Tensor& dq, Tensor& dk, Tensor& dv,
                                       bool causal, float scale);

    namespace {
        void check(const Tensor& q, const Tensor& k, const Tensor& v) {
            if (q.dtype != DType::Fp32 || k.dtype != DType::Fp32 || v.dtype != DType::Fp32) {
                throw std::invalid_argument("flash_attention: only Fp32 supported");
            }
            if (q.device != k.device || q.device != v.device) {
                throw std::invalid_argument("flash_attention: device mismatch");
            }
            if (q.shape.size() < 3 || k.shape.size() < 3 || v.shape.size() < 3) {
                throw std::invalid_argument("flash_attention: need at least 3 dims [..., T, d]");
            }
            if (q.shape != k.shape || q.shape != v.shape) {
                throw std::invalid_argument("flash_attention: q, k, v must share shape");
            }
        }

        Tensor flash_attention_cpu(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
            const float s = 1.0f / std::sqrt(static_cast<float>(q.shape.back()));
            Tensor scores = scale(bmm_a_bt(q, k), s);
            if (causal) scores = apply_causal_mask(scores);
            return bmm(softmax(scores), v);
        }

        class FlashAttentionFunction : public Function {
        public:
            Tensor saved_q, saved_k, saved_v, saved_o, saved_lse;
            bool causal = false;
            float scale_factor = 1.0f;

            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dq = Tensor::zeros(saved_q.shape, saved_q.dtype, saved_q.device);
                Tensor dk = Tensor::zeros(saved_k.shape, saved_k.dtype, saved_k.device);
                Tensor dv = Tensor::zeros(saved_v.shape, saved_v.dtype, saved_v.device);
                flash_attention_backward_cuda(saved_q, saved_k, saved_v, saved_o, saved_lse,
                                              grad_output, dq, dk, dv, causal, scale_factor);
                return {dq, dk, dv};
            }
            const char* name() const override { return "FlashAttentionFunction"; }
        };
    }

    Tensor flash_attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal) {
        check(q, k, v);
        Tensor qc = q.is_contiguous() ? q : contiguous(q);
        Tensor kc = k.is_contiguous() ? k : contiguous(k);
        Tensor vc = v.is_contiguous() ? v : contiguous(v);

        if (qc.device == Device::CPU) {
            return flash_attention_cpu(qc, kc, vc, causal);
        }

        const float s = 1.0f / std::sqrt(static_cast<float>(qc.shape.back()));
        std::vector<int64_t> lse_shape(qc.shape.begin(), qc.shape.end() - 1);
        Tensor out = Tensor::empty(qc.shape, qc.dtype, qc.device);
        Tensor lse = Tensor::empty(lse_shape, qc.dtype, qc.device);
        flash_attention_forward_cuda(qc, kc, vc, out, lse, causal, s);

        if (!qc.requires_grad && !kc.requires_grad && !vc.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<FlashAttentionFunction>();
        fn->record_input(qc);
        fn->record_input(kc);
        fn->record_input(vc);
        fn->saved_q = qc;
        fn->saved_k = kc;
        fn->saved_v = vc;
        fn->saved_o = out;
        fn->saved_lse = lse;
        fn->causal = causal;
        fn->scale_factor = s;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
