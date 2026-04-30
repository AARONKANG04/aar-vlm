#include "ops/bmm.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

#include "autograd/function.hpp"

namespace vlm {
    Tensor bmm_cuda(const Tensor& a, const Tensor& b);
    Tensor bmm_a_bt_cuda(const Tensor& a, const Tensor& b);
    Tensor bmm_at_b_cuda(const Tensor& a, const Tensor& b);

    namespace {
        void check_bmm_common(const Tensor& a, const Tensor& b, const char* op) {
            if (a.shape.size() < 2 || b.shape.size() < 2) {
                throw std::invalid_argument(std::string(op) + ": rank must be >= 2");
            }
            if (a.shape.size() != b.shape.size()) {
                throw std::invalid_argument(std::string(op) + ": rank mismatch");
            }
            if (a.dtype != b.dtype || a.dtype != DType::Fp32) {
                throw std::invalid_argument(std::string(op) + ": only Fp32 supported");
            }
            if (a.device != b.device) {
                throw std::invalid_argument(std::string(op) + ": device mismatch");
            }
            for (size_t i = 0; i + 2 < a.shape.size(); ++i) {
                if (a.shape[i] != b.shape[i]) {
                    throw std::invalid_argument(std::string(op) + ": leading dims must match");
                }
            }
        }

        int64_t leading_product(const Tensor& a) {
            int64_t p = 1;
            for (size_t i = 0; i + 2 < a.shape.size(); ++i) p *= a.shape[i];
            return p;
        }

        std::vector<int64_t> with_last_two(const Tensor& a, int64_t last_minus_1, int64_t last) {
            std::vector<int64_t> out(a.shape.begin(), a.shape.end() - 2);
            out.push_back(last_minus_1);
            out.push_back(last);
            return out;
        }

        // Walk the leading-dim shape of `t` and produce per-flat-batch storage offsets
        // (relative to t.data() which already includes t.storage_offset).
        std::vector<int64_t> batch_offsets(const Tensor& t, int64_t B) {
            std::vector<int64_t> offs(static_cast<size_t>(B), 0);
            const size_t nd = t.shape.size();
            if (nd <= 2 || B == 0) return offs;
            const size_t bdims = nd - 2;
            std::vector<int64_t> idx(bdims, 0);
            for (int64_t bi = 0; bi < B; ++bi) {
                int64_t off = 0;
                for (size_t d = 0; d < bdims; ++d) off += idx[d] * t.strides[d];
                offs[bi] = off;
                for (int d = static_cast<int>(bdims) - 1; d >= 0; --d) {
                    if (++idx[d] < t.shape[d]) break;
                    idx[d] = 0;
                }
            }
            return offs;
        }

        Tensor bmm_cpu(const Tensor& a, const Tensor& b) {
            const size_t nd = a.shape.size();
            const int64_t M = a.shape[nd - 2];
            const int64_t K = a.shape[nd - 1];
            const int64_t N = b.shape[nd - 1];
            const int64_t B = leading_product(a);
            Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, a.device);
            const int64_t a_rs = a.strides[nd - 2];
            const int64_t a_cs = a.strides[nd - 1];
            const int64_t b_rs = b.strides[nd - 2];
            const int64_t b_cs = b.strides[nd - 1];
            const auto a_off = batch_offsets(a, B);
            const auto b_off = batch_offsets(b, B);
            const float* A = static_cast<const float*>(a.data());
            const float* Bp = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            for (int64_t bi = 0; bi < B; ++bi) {
                const float* Ab = A + a_off[bi];
                const float* Bb = Bp + b_off[bi];
                float* Cb = C + bi * M * N;
                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        float acc = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            acc += Ab[i * a_rs + k * a_cs] * Bb[k * b_rs + j * b_cs];
                        }
                        Cb[i * N + j] = acc;
                    }
                }
            }
            return out;
        }

        Tensor bmm_a_bt_cpu(const Tensor& a, const Tensor& b) {
            const size_t nd = a.shape.size();
            const int64_t M = a.shape[nd - 2];
            const int64_t K = a.shape[nd - 1];
            const int64_t N = b.shape[nd - 2];
            const int64_t B = leading_product(a);
            Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, a.device);
            const int64_t a_rs = a.strides[nd - 2];
            const int64_t a_cs = a.strides[nd - 1];
            const int64_t b_rs = b.strides[nd - 2];
            const int64_t b_cs = b.strides[nd - 1];
            const auto a_off = batch_offsets(a, B);
            const auto b_off = batch_offsets(b, B);
            const float* A = static_cast<const float*>(a.data());
            const float* Bp = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            for (int64_t bi = 0; bi < B; ++bi) {
                const float* Ab = A + a_off[bi];
                const float* Bb = Bp + b_off[bi];
                float* Cb = C + bi * M * N;
                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        float acc = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            acc += Ab[i * a_rs + k * a_cs] * Bb[j * b_rs + k * b_cs];
                        }
                        Cb[i * N + j] = acc;
                    }
                }
            }
            return out;
        }

        Tensor bmm_at_b_cpu(const Tensor& a, const Tensor& b) {
            const size_t nd = a.shape.size();
            const int64_t K = a.shape[nd - 2];
            const int64_t M = a.shape[nd - 1];
            const int64_t N = b.shape[nd - 1];
            const int64_t B = leading_product(a);
            Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, a.device);
            const int64_t a_rs = a.strides[nd - 2];
            const int64_t a_cs = a.strides[nd - 1];
            const int64_t b_rs = b.strides[nd - 2];
            const int64_t b_cs = b.strides[nd - 1];
            const auto a_off = batch_offsets(a, B);
            const auto b_off = batch_offsets(b, B);
            const float* A = static_cast<const float*>(a.data());
            const float* Bp = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            for (int64_t bi = 0; bi < B; ++bi) {
                const float* Ab = A + a_off[bi];
                const float* Bb = Bp + b_off[bi];
                float* Cb = C + bi * M * N;
                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        float acc = 0.0f;
                        for (int64_t k = 0; k < K; ++k) {
                            acc += Ab[k * a_rs + i * a_cs] * Bb[k * b_rs + j * b_cs];
                        }
                        Cb[i * N + j] = acc;
                    }
                }
            }
            return out;
        }

        Tensor bmm_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? bmm_cpu(a, b) : bmm_cuda(a, b);
        }
        Tensor bmm_a_bt_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? bmm_a_bt_cpu(a, b) : bmm_a_bt_cuda(a, b);
        }
        Tensor bmm_at_b_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? bmm_at_b_cpu(a, b) : bmm_at_b_cuda(a, b);
        }

        class BmmFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = bmm_a_bt_no_grad(grad_output, saved_b);
                Tensor dB = bmm_at_b_no_grad(saved_a, grad_output);
                return {dA, dB};
            }
            const char* name() const override { return "BmmFunction"; }
        };

        class BmmABTFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = bmm_no_grad(grad_output, saved_b);
                Tensor dB = bmm_at_b_no_grad(grad_output, saved_a);
                return {dA, dB};
            }
            const char* name() const override { return "BmmABTFunction"; }
        };

        class BmmATBFunction : public Function {
        public:
            Tensor saved_a, saved_b;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor dA = bmm_a_bt_no_grad(saved_b, grad_output);
                Tensor dB = bmm_no_grad(saved_a, grad_output);
                return {dA, dB};
            }
            const char* name() const override { return "BmmATBFunction"; }
        };
    }

    Tensor bmm(const Tensor& a, const Tensor& b) {
        check_bmm_common(a, b, "bmm");
        const size_t nd = a.shape.size();
        if (a.shape[nd - 1] != b.shape[nd - 2]) {
            throw std::invalid_argument("bmm: contraction dim mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return bmm_no_grad(a, b);
        }
        auto fn = std::make_shared<BmmFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = bmm_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor bmm_a_bt(const Tensor& a, const Tensor& b) {
        check_bmm_common(a, b, "bmm_a_bt");
        const size_t nd = a.shape.size();
        if (a.shape[nd - 1] != b.shape[nd - 1]) {
            throw std::invalid_argument("bmm_a_bt: contraction dim mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return bmm_a_bt_no_grad(a, b);
        }
        auto fn = std::make_shared<BmmABTFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = bmm_a_bt_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor bmm_at_b(const Tensor& a, const Tensor& b) {
        check_bmm_common(a, b, "bmm_at_b");
        const size_t nd = a.shape.size();
        if (a.shape[nd - 2] != b.shape[nd - 2]) {
            throw std::invalid_argument("bmm_at_b: contraction dim mismatch");
        }
        if (!a.requires_grad && !b.requires_grad) {
            return bmm_at_b_no_grad(a, b);
        }
        auto fn = std::make_shared<BmmATBFunction>();
        fn->record_input(a);
        fn->record_input(b);
        fn->saved_a = a;
        fn->saved_b = b;
        Tensor out = bmm_at_b_no_grad(a, b);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
