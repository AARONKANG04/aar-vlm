#include "ops/embedding.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor embedding_cuda(const Tensor& weight, const Tensor& ids);
    Tensor embedding_backward_cuda(const Tensor& grad_out, const Tensor& ids,
                                    int64_t vocab_size);

    namespace {
        Tensor embedding_cpu(const Tensor& weight, const Tensor& ids) {
            const int64_t D = weight.shape[1];
            std::vector<int64_t> out_shape = ids.shape;
            out_shape.push_back(D);
            Tensor out = Tensor::empty(out_shape, weight.dtype, weight.device);
            const float* W = static_cast<const float*>(weight.data());
            const int64_t* I = static_cast<const int64_t*>(ids.data());
            float* O = static_cast<float*>(out.data());
            const int64_t V = weight.shape[0];
            const size_t N = ids.numel();
            for (size_t i = 0; i < N; ++i) {
                const int64_t idx = I[i];
                if (idx < 0 || idx >= V) {
                    throw std::out_of_range("embedding: id out of range [0, vocab_size)");
                }
                std::memcpy(O + i * D, W + idx * D, D * sizeof(float));
            }
            return out;
        }

        Tensor embedding_backward_cpu(const Tensor& grad_out, const Tensor& ids,
                                       int64_t vocab_size) {
            const int64_t D = grad_out.shape.back();
            Tensor dW = Tensor::zeros({vocab_size, D}, grad_out.dtype, grad_out.device);
            const float* G = static_cast<const float*>(grad_out.data());
            const int64_t* I = static_cast<const int64_t*>(ids.data());
            float* DW = static_cast<float*>(dW.data());
            const size_t N = ids.numel();
            for (size_t i = 0; i < N; ++i) {
                const int64_t idx = I[i];
                const float* grow = G + i * D;
                float* drow = DW + idx * D;
                for (int64_t k = 0; k < D; ++k) drow[k] += grow[k];
            }
            return dW;
        }

        Tensor embedding_no_grad(const Tensor& weight, const Tensor& ids) {
            return weight.device == Device::CPU
                ? embedding_cpu(weight, ids)
                : embedding_cuda(weight, ids);
        }

        Tensor embedding_backward_no_grad(const Tensor& grad_out, const Tensor& ids,
                                           int64_t vocab_size) {
            return grad_out.device == Device::CPU
                ? embedding_backward_cpu(grad_out, ids, vocab_size)
                : embedding_backward_cuda(grad_out, ids, vocab_size);
        }

        class EmbeddingFunction : public Function {
        public:
            Tensor saved_ids;
            int64_t vocab_size = 0;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dW = embedding_backward_no_grad(grad_output, saved_ids, vocab_size);
                return {dW};
            }
            const char* name() const override { return "EmbeddingFunction"; }
        };
    }

    Tensor embedding(const Tensor& weight, const Tensor& ids) {
        if (weight.dtype != DType::Fp32) {
            throw std::invalid_argument("embedding: weight must be Fp32");
        }
        if (ids.dtype != DType::Int64) {
            throw std::invalid_argument("embedding: ids must be Int64");
        }
        if (weight.shape.size() != 2) {
            throw std::invalid_argument("embedding: weight must be 2D (vocab_size, embed_dim)");
        }
        if (weight.device != ids.device) {
            throw std::invalid_argument("embedding: device mismatch");
        }
        Tensor wc = weight.is_contiguous() ? weight : contiguous(weight);
        Tensor ic = ids.is_contiguous() ? ids : contiguous(ids);
        if (!wc.requires_grad) {
            return embedding_no_grad(wc, ic);
        }
        auto fn = std::make_shared<EmbeddingFunction>();
        fn->record_input(wc);
        fn->saved_ids = ic;
        fn->vocab_size = wc.shape[0];
        Tensor out = embedding_no_grad(wc, ic);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
