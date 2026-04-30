#include "ops/dropout.hpp"

#include <atomic>
#include <memory>
#include <random>
#include <stdexcept>

#include "autograd/function.hpp"
#include "core/rng.hpp"
#include "ops/elementwise.hpp"
#include "ops/shape.hpp"

namespace vlm {
    void dropout_cuda(const Tensor& x, float p, uint64_t seed,
                      Tensor& out, Tensor& mask);

    namespace {
        std::atomic<uint64_t>& seed_counter() {
            static std::atomic<uint64_t> counter{
                static_cast<uint64_t>(std::random_device{}())
            };
            return counter;
        }

        uint64_t next_seed() {
            return seed_counter().fetch_add(1, std::memory_order_relaxed);
        }

        void dropout_cpu(const Tensor& x, float p, uint64_t seed,
                         Tensor& out, Tensor& mask) {
            const float* X = static_cast<const float*>(x.data());
            float* O = static_cast<float*>(out.data());
            float* M = static_cast<float*>(mask.data());
            const float scale = 1.0f / (1.0f - p);
            const size_t n = x.numel();
            for (size_t i = 0; i < n; ++i) {
                const float r = philox_uniform(seed, static_cast<uint64_t>(i));
                const float keep_scaled = (r >= p) ? scale : 0.0f;
                M[i] = keep_scaled;
                O[i] = X[i] * keep_scaled;
            }
        }

        void dropout_no_grad(const Tensor& x, float p, uint64_t seed,
                              Tensor& out, Tensor& mask) {
            if (x.device == Device::CPU) {
                dropout_cpu(x, p, seed, out, mask);
            } else {
                dropout_cuda(x, p, seed, out, mask);
            }
        }

        class DropoutFunction : public Function {
        public:
            Tensor saved_mask;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {mul(grad_output, saved_mask)};
            }
            const char* name() const override { return "DropoutFunction"; }
        };
    }

    void manual_seed(uint64_t seed) {
        seed_counter().store(seed, std::memory_order_relaxed);
    }

    Tensor dropout(const Tensor& x, float p) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("dropout: only Fp32 supported");
        }
        if (!(p >= 0.0f && p < 1.0f)) {
            throw std::invalid_argument("dropout: p must be in [0, 1)");
        }
        Tensor xc = x.is_contiguous() ? x : contiguous(x);
        if (p == 0.0f) {
            return xc;
        }
        const uint64_t seed = next_seed();
        Tensor out = Tensor::empty(xc.shape, xc.dtype, xc.device);
        Tensor mask = Tensor::empty(xc.shape, xc.dtype, xc.device);
        dropout_no_grad(xc, p, seed, out, mask);
        if (!xc.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<DropoutFunction>();
        fn->record_input(xc);
        fn->saved_mask = mask;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
