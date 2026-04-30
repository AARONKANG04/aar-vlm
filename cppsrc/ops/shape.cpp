#include "ops/shape.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "autograd/function.hpp"

namespace vlm {
    Tensor transpose_cuda(const Tensor& x, int64_t dim_a, int64_t dim_b);

    namespace {
        std::vector<int64_t> resolve_new_shape(const std::vector<int64_t>& shape, size_t numel) {
            std::vector<int64_t> out = shape;
            int neg_pos = -1;
            int64_t known = 1;
            for (size_t i = 0; i < out.size(); ++i) {
                if (out[i] == -1) {
                    if (neg_pos != -1) {
                        throw std::invalid_argument("reshape: at most one -1 dim allowed");
                    }
                    neg_pos = static_cast<int>(i);
                } else if (out[i] < 0) {
                    throw std::invalid_argument("reshape: negative dim other than -1");
                } else {
                    known *= out[i];
                }
            }
            if (neg_pos == -1) {
                int64_t prod = 1;
                for (auto d : out) prod *= d;
                if (static_cast<size_t>(prod) != numel) {
                    throw std::invalid_argument("reshape: numel mismatch");
                }
            } else {
                if (known == 0 || numel % static_cast<size_t>(known) != 0) {
                    throw std::invalid_argument("reshape: cannot infer -1 dim");
                }
                out[neg_pos] = static_cast<int64_t>(numel) / known;
            }
            return out;
        }

        Tensor reshape_view(const Tensor& x, std::vector<int64_t> new_shape) {
            Tensor v;
            v.shape = std::move(new_shape);
            v.dtype = x.dtype;
            v.device = x.device;
            v.storage = x.storage;
            return v;
        }

        Tensor transpose_cpu(const Tensor& x, int64_t dim_a, int64_t dim_b) {
            const size_t ndim = x.shape.size();
            std::vector<int64_t> out_shape = x.shape;
            std::swap(out_shape[dim_a], out_shape[dim_b]);
            Tensor out = Tensor::empty(out_shape, x.dtype, x.device);

            std::vector<int64_t> in_strides(ndim);
            int64_t s = 1;
            for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
                in_strides[i] = s;
                s *= x.shape[i];
            }
            std::vector<int64_t> out_strides(ndim);
            s = 1;
            for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
                out_strides[i] = s;
                s *= out_shape[i];
            }

            const float* X = static_cast<const float*>(x.data());
            float* Y = static_cast<float*>(out.data());
            const size_t total = out.numel();
            std::vector<int64_t> idx(ndim, 0);
            for (size_t flat = 0; flat < total; ++flat) {
                size_t rem = flat;
                for (size_t d = 0; d < ndim; ++d) {
                    idx[d] = static_cast<int64_t>(rem / static_cast<size_t>(out_strides[d]));
                    rem -= static_cast<size_t>(idx[d] * out_strides[d]);
                }
                std::swap(idx[dim_a], idx[dim_b]);
                size_t in_flat = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    in_flat += static_cast<size_t>(idx[d] * in_strides[d]);
                }
                Y[flat] = X[in_flat];
            }
            return out;
        }

        Tensor transpose_no_grad(const Tensor& x, int64_t dim_a, int64_t dim_b) {
            return x.device == Device::CPU ? transpose_cpu(x, dim_a, dim_b)
                                            : transpose_cuda(x, dim_a, dim_b);
        }

        class ReshapeFunction : public Function {
        public:
            std::vector<int64_t> original_shape;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {reshape_view(grad_output, original_shape)};
            }
            const char* name() const override { return "ReshapeFunction"; }
        };

        class TransposeFunction : public Function {
        public:
            int64_t dim_a = 0;
            int64_t dim_b = 0;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {transpose_no_grad(grad_output, dim_a, dim_b)};
            }
            const char* name() const override { return "TransposeFunction"; }
        };
    }

    Tensor reshape(const Tensor& x, std::vector<int64_t> new_shape) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("reshape: only Fp32 supported");
        }
        auto resolved = resolve_new_shape(new_shape, x.numel());
        Tensor out = reshape_view(x, resolved);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<ReshapeFunction>();
        fn->record_input(x);
        fn->original_shape = x.shape;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor transpose(const Tensor& x, int64_t dim_a, int64_t dim_b) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("transpose: only Fp32 supported");
        }
        const int64_t ndim = static_cast<int64_t>(x.shape.size());
        if (dim_a < 0 || dim_a >= ndim || dim_b < 0 || dim_b >= ndim) {
            throw std::invalid_argument("transpose: dim out of range");
        }
        Tensor out = transpose_no_grad(x, dim_a, dim_b);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<TransposeFunction>();
        fn->record_input(x);
        fn->dim_a = dim_a;
        fn->dim_b = dim_b;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
