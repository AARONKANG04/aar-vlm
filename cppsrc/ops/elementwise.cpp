#include "ops/elementwise.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"
#include "core/cuda_copy.hpp"
#include "ops/shape.hpp"

namespace vlm {
    Tensor add_cuda(const Tensor& a, const Tensor& b);
    Tensor mul_cuda(const Tensor& a, const Tensor& b);
    Tensor relu_cuda(const Tensor& a);
    Tensor sum_all_cuda(const Tensor& a);
    Tensor relu_backward_cuda(const Tensor& grad_out, const Tensor& x);
    void fill_cuda(Tensor& out, float v);
    void scaled_add_inplace_cuda(Tensor& dst, const Tensor& src, float alpha);
    Tensor sub_cuda(const Tensor& a, const Tensor& b);
    Tensor neg_cuda(const Tensor& a);
    Tensor gelu_cuda(const Tensor& a);
    Tensor gelu_backward_cuda(const Tensor& grad_out, const Tensor& x);
    Tensor add_bias_cuda(const Tensor& x, const Tensor& bias);
    Tensor bias_grad_cuda(const Tensor& grad_out, int64_t D);
    Tensor scale_cuda(const Tensor& x, float alpha);

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

        Tensor scale_cpu(const Tensor& x, float alpha) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            float* Y = static_cast<float*>(out.data());
            const size_t n = x.numel();
            for (size_t i = 0; i < n; ++i) Y[i] = alpha * X[i];
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

        Tensor scale_no_grad(const Tensor& x, float alpha) {
            return x.device == Device::CPU ? scale_cpu(x, alpha) : scale_cuda(x, alpha);
        }

        Tensor sub_cpu(const Tensor& a, const Tensor& b) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) C[i] = A[i] - B[i];
            return out;
        }

        Tensor neg_cpu(const Tensor& a) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            for (size_t i = 0; i < n; ++i) C[i] = -A[i];
            return out;
        }

        Tensor sub_no_grad(const Tensor& a, const Tensor& b) {
            return a.device == Device::CPU ? sub_cpu(a, b) : sub_cuda(a, b);
        }

        Tensor neg_no_grad(const Tensor& a) {
            return a.device == Device::CPU ? neg_cpu(a) : neg_cuda(a);
        }

        Tensor gelu_cpu(const Tensor& a) {
            Tensor out = Tensor::empty(a.shape, a.dtype, a.device);
            const float* A = static_cast<const float*>(a.data());
            float* C = static_cast<float*>(out.data());
            const size_t n = a.numel();
            const float inv_sqrt2 = 0.70710678118654752440f;
            for (size_t i = 0; i < n; ++i) {
                C[i] = 0.5f * A[i] * (1.0f + std::erf(A[i] * inv_sqrt2));
            }
            return out;
        }

        Tensor gelu_backward_cpu(const Tensor& grad_out, const Tensor& x) {
            Tensor g = Tensor::empty(x.shape, x.dtype, x.device);
            const float* go = static_cast<const float*>(grad_out.data());
            const float* xp = static_cast<const float*>(x.data());
            float* gp = static_cast<float*>(g.data());
            const size_t n = x.numel();
            const float inv_sqrt2 = 0.70710678118654752440f;
            const float inv_sqrt_2pi = 0.39894228040143267794f;
            for (size_t i = 0; i < n; ++i) {
                const float xi = xp[i];
                const float cdf = 0.5f * (1.0f + std::erf(xi * inv_sqrt2));
                const float pdf = inv_sqrt_2pi * std::exp(-0.5f * xi * xi);
                gp[i] = go[i] * (cdf + xi * pdf);
            }
            return g;
        }

        Tensor gelu_backward_no_grad(const Tensor& grad_out, const Tensor& x) {
            return grad_out.device == Device::CPU ? gelu_backward_cpu(grad_out, x)
                                                  : gelu_backward_cuda(grad_out, x);
        }

        void check_bias(const Tensor& x, const Tensor& bias) {
            if (bias.shape.size() != 1) {
                throw std::invalid_argument("add_bias: bias must be 1-D");
            }
            if (x.shape.empty() || x.shape.back() != bias.shape[0]) {
                throw std::invalid_argument("add_bias: bias size must match x's last dim");
            }
            if (x.dtype != bias.dtype || x.dtype != DType::Fp32) {
                throw std::invalid_argument("add_bias: only Fp32 supported");
            }
            if (x.device != bias.device) {
                throw std::invalid_argument("add_bias: device mismatch");
            }
        }

        Tensor add_bias_cpu(const Tensor& x, const Tensor& bias) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            const float* B = static_cast<const float*>(bias.data());
            float* Y = static_cast<float*>(out.data());
            const size_t N = x.numel();
            const size_t D = bias.numel();
            for (size_t i = 0; i < N; ++i) Y[i] = X[i] + B[i % D];
            return out;
        }

        Tensor bias_grad_cpu(const Tensor& grad_out, int64_t D) {
            Tensor db = Tensor::zeros({D}, grad_out.dtype, grad_out.device);
            const float* G = static_cast<const float*>(grad_out.data());
            float* B = static_cast<float*>(db.data());
            const size_t N = grad_out.numel();
            const size_t Du = static_cast<size_t>(D);
            for (size_t i = 0; i < N; ++i) B[i % Du] += G[i];
            return db;
        }

        Tensor add_bias_no_grad(const Tensor& x, const Tensor& bias) {
            return x.device == Device::CPU ? add_bias_cpu(x, bias) : add_bias_cuda(x, bias);
        }

        Tensor bias_grad_no_grad(const Tensor& grad_out, int64_t D) {
            return grad_out.device == Device::CPU ? bias_grad_cpu(grad_out, D)
                                                  : bias_grad_cuda(grad_out, D);
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

        class ScaleFunction : public Function {
        public:
            float alpha = 1.0f;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {scale_no_grad(grad_output, alpha)};
            }
            const char* name() const override { return "ScaleFunction"; }
        };

        class ReluFunction : public Function {
        public:
            Tensor saved_x;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {relu_backward_no_grad(grad_output, saved_x)};
            }
            const char* name() const override { return "ReluFunction"; }
        };

        class SubFunction : public Function {
        public:
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {grad_output, neg_no_grad(grad_output)};
            }
            const char* name() const override { return "SubFunction"; }
        };

        class GeluFunction : public Function {
        public:
            Tensor saved_x;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {gelu_backward_no_grad(grad_output, saved_x)};
            }
            const char* name() const override { return "GeluFunction"; }
        };

        class AddBiasFunction : public Function {
        public:
            int64_t bias_size = 0;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor db = bias_grad_no_grad(grad_output, bias_size);
                return {grad_output, db};
            }
            const char* name() const override { return "AddBiasFunction"; }
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

    namespace {
        Tensor as_contig(const Tensor& t) {
            return t.is_contiguous() ? t : contiguous(t);
        }
    }

    Tensor add(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "add");
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return add_no_grad(ac, bc);
        }
        auto fn = std::make_shared<AddFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        Tensor out = add_no_grad(ac, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor mul(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "mul");
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return mul_no_grad(ac, bc);
        }
        auto fn = std::make_shared<MulFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        fn->saved_a = ac;
        fn->saved_b = bc;
        Tensor out = mul_no_grad(ac, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor scale(const Tensor& x, float alpha) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("scale: only Fp32 supported");
        }
        Tensor xc = as_contig(x);
        if (!xc.requires_grad) {
            return scale_no_grad(xc, alpha);
        }
        auto fn = std::make_shared<ScaleFunction>();
        fn->record_input(xc);
        fn->alpha = alpha;
        Tensor out = scale_no_grad(xc, alpha);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor relu(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("relu: only Fp32 supported");
        }
        Tensor ac = as_contig(a);
        if (!ac.requires_grad) {
            return ac.device == Device::CPU ? relu_cpu(ac) : relu_cuda(ac);
        }
        auto fn = std::make_shared<ReluFunction>();
        fn->record_input(ac);
        fn->saved_x = ac;
        Tensor out = ac.device == Device::CPU ? relu_cpu(ac) : relu_cuda(ac);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor sum_all(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("sum_all: only Fp32 supported");
        }
        Tensor ac = as_contig(a);
        if (!ac.requires_grad) {
            return ac.device == Device::CPU ? sum_all_cpu(ac) : sum_all_cuda(ac);
        }
        auto fn = std::make_shared<SumAllFunction>();
        fn->record_input(ac);
        fn->input_shape = ac.shape;
        fn->input_dtype = ac.dtype;
        fn->input_device = ac.device;
        Tensor out = ac.device == Device::CPU ? sum_all_cpu(ac) : sum_all_cuda(ac);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor sub(const Tensor& a, const Tensor& b) {
        check_binary(a, b, "sub");
        Tensor ac = as_contig(a);
        Tensor bc = as_contig(b);
        if (!ac.requires_grad && !bc.requires_grad) {
            return sub_no_grad(ac, bc);
        }
        auto fn = std::make_shared<SubFunction>();
        fn->record_input(ac);
        fn->record_input(bc);
        Tensor out = sub_no_grad(ac, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor gelu(const Tensor& a) {
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("gelu: only Fp32 supported");
        }
        Tensor ac = as_contig(a);
        if (!ac.requires_grad) {
            return ac.device == Device::CPU ? gelu_cpu(ac) : gelu_cuda(ac);
        }
        auto fn = std::make_shared<GeluFunction>();
        fn->record_input(ac);
        fn->saved_x = ac;
        Tensor out = ac.device == Device::CPU ? gelu_cpu(ac) : gelu_cuda(ac);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor add_bias(const Tensor& x, const Tensor& bias) {
        check_bias(x, bias);
        Tensor xc = as_contig(x);
        Tensor bc = as_contig(bias);
        if (!xc.requires_grad && !bc.requires_grad) {
            return add_bias_no_grad(xc, bc);
        }
        auto fn = std::make_shared<AddBiasFunction>();
        fn->record_input(xc);
        fn->record_input(bc);
        fn->bias_size = bc.shape[0];
        Tensor out = add_bias_no_grad(xc, bc);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    void scaled_add_inplace(Tensor& dst, const Tensor& src, float alpha) {
        if (dst.shape != src.shape) {
            throw std::invalid_argument("scaled_add_inplace: shape mismatch");
        }
        if (dst.dtype != src.dtype || dst.dtype != DType::Fp32) {
            throw std::invalid_argument("scaled_add_inplace: only Fp32 supported");
        }
        if (dst.device != src.device) {
            throw std::invalid_argument("scaled_add_inplace: device mismatch");
        }
        if (!dst.is_contiguous()) {
            throw std::invalid_argument("scaled_add_inplace: dst must be contiguous");
        }
        Tensor sc = src.is_contiguous() ? src : contiguous(src);
        if (dst.device == Device::CUDA) {
            scaled_add_inplace_cuda(dst, sc, alpha);
            return;
        }
        float* d = static_cast<float*>(dst.data());
        const float* s = static_cast<const float*>(sc.data());
        const size_t n = dst.numel();
        for (size_t i = 0; i < n; ++i) d[i] += alpha * s[i];
    }
}
