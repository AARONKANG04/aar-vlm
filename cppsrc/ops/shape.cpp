#include "ops/shape.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "autograd/function.hpp"

namespace vlm {
    void contiguous_cuda(const Tensor& src, Tensor& dst);
    void copy_contiguous_into_strided_cuda(const Tensor& src, Tensor& dst);

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

        void copy_strided_to_contiguous_cpu(const Tensor& src, Tensor& dst) {
            const size_t ndim = src.shape.size();
            const size_t total = src.numel();
            if (total == 0) return;
            const float* X = static_cast<const float*>(src.data());
            float* Y = static_cast<float*>(dst.data());
            if (ndim == 0) {
                Y[0] = X[0];
                return;
            }
            const auto& src_strides = src.strides;
            const auto& dst_strides = dst.strides;
            std::vector<int64_t> idx(ndim, 0);
            for (size_t flat = 0; flat < total; ++flat) {
                size_t rem = flat;
                int64_t src_off = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    idx[d] = static_cast<int64_t>(rem / static_cast<size_t>(dst_strides[d]));
                    rem -= static_cast<size_t>(idx[d] * dst_strides[d]);
                    src_off += idx[d] * src_strides[d];
                }
                Y[flat] = X[src_off];
            }
        }

        void copy_contiguous_into_strided_cpu(const Tensor& src, Tensor& dst) {
            const size_t ndim = src.shape.size();
            const size_t total = src.numel();
            if (total == 0) return;
            const float* X = static_cast<const float*>(src.data());
            float* Y = static_cast<float*>(dst.data());
            if (ndim == 0) {
                Y[0] = X[0];
                return;
            }
            const auto src_contig = compute_contiguous_strides(src.shape);
            const auto& dst_strides = dst.strides;
            std::vector<int64_t> idx(ndim, 0);
            for (size_t flat = 0; flat < total; ++flat) {
                size_t rem = flat;
                int64_t dst_off = 0;
                for (size_t d = 0; d < ndim; ++d) {
                    idx[d] = static_cast<int64_t>(rem / static_cast<size_t>(src_contig[d]));
                    rem -= static_cast<size_t>(idx[d] * src_contig[d]);
                    dst_off += idx[d] * dst_strides[d];
                }
                Y[dst_off] = X[flat];
            }
        }
    }

    Tensor make_view(const Tensor& src,
                     std::vector<int64_t> shape,
                     std::vector<int64_t> strides,
                     int64_t storage_offset) {
        Tensor v;
        v.shape = std::move(shape);
        v.strides = std::move(strides);
        v.storage_offset = storage_offset;
        v.dtype = src.dtype;
        v.device = src.device;
        v.storage = src.storage;
        return v;
    }

    void copy_strided_to_contiguous(const Tensor& src, Tensor& dst) {
        if (src.shape != dst.shape) {
            throw std::runtime_error("copy_strided_to_contiguous: shape mismatch");
        }
        if (src.dtype != dst.dtype || src.device != dst.device) {
            throw std::runtime_error("copy_strided_to_contiguous: dtype/device mismatch");
        }
        if (src.dtype != DType::Fp32) {
            throw std::runtime_error("copy_strided_to_contiguous: only Fp32 supported");
        }
        if (!dst.is_contiguous()) {
            throw std::runtime_error("copy_strided_to_contiguous: dst must be contiguous");
        }
        if (src.device == Device::CUDA) {
            contiguous_cuda(src, dst);
            return;
        }
        copy_strided_to_contiguous_cpu(src, dst);
    }

    namespace {
        void copy_contiguous_into_strided(const Tensor& src_contig, Tensor& dst_strided) {
            if (src_contig.shape != dst_strided.shape) {
                throw std::runtime_error("copy_contiguous_into_strided: shape mismatch");
            }
            if (src_contig.dtype != dst_strided.dtype
                || src_contig.device != dst_strided.device) {
                throw std::runtime_error("copy_contiguous_into_strided: dtype/device mismatch");
            }
            if (!src_contig.is_contiguous()) {
                throw std::runtime_error("copy_contiguous_into_strided: src must be contiguous");
            }
            if (src_contig.device == Device::CUDA) {
                copy_contiguous_into_strided_cuda(src_contig, dst_strided);
                return;
            }
            copy_contiguous_into_strided_cpu(src_contig, dst_strided);
        }
    }

    namespace {
        class ReshapeFunction : public Function {
        public:
            std::vector<int64_t> original_shape;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor g = grad_output.is_contiguous() ? grad_output : contiguous(grad_output);
                Tensor out = make_view(g, original_shape,
                                       compute_contiguous_strides(original_shape),
                                       g.storage_offset);
                return {out};
            }
            const char* name() const override { return "ReshapeFunction"; }
        };

        class TransposeFunction : public Function {
        public:
            int64_t dim_a = 0;
            int64_t dim_b = 0;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                std::vector<int64_t> sh = grad_output.shape;
                std::vector<int64_t> st = grad_output.strides;
                std::swap(sh[dim_a], sh[dim_b]);
                std::swap(st[dim_a], st[dim_b]);
                Tensor view = make_view(grad_output, sh, st, grad_output.storage_offset);
                return {view.is_contiguous() ? view : contiguous(view)};
            }
            const char* name() const override { return "TransposeFunction"; }
        };

        class SliceFunction : public Function {
        public:
            std::vector<int64_t> original_shape;
            int64_t dim = 0;
            int64_t start = 0;
            int64_t end = 0;
            DType dtype = DType::Fp32;
            Device device = Device::CPU;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                Tensor full = Tensor::zeros(original_shape, dtype, device);
                Tensor slot = make_view(full, grad_output.shape, full.strides,
                                        full.storage_offset + start * full.strides[dim]);
                Tensor src = grad_output.is_contiguous() ? grad_output : contiguous(grad_output);
                copy_contiguous_into_strided(src, slot);
                return {full};
            }
            const char* name() const override { return "SliceFunction"; }
        };

        class SqueezeFunction : public Function {
        public:
            int64_t dim = 0;
            int64_t inserted_size_stride = 1;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                std::vector<int64_t> sh = grad_output.shape;
                std::vector<int64_t> st = grad_output.strides;
                sh.insert(sh.begin() + dim, 1);
                st.insert(st.begin() + dim, inserted_size_stride);
                Tensor view = make_view(grad_output, sh, st, grad_output.storage_offset);
                return {view.is_contiguous() ? view : contiguous(view)};
            }
            const char* name() const override { return "SqueezeFunction"; }
        };

        class UnsqueezeFunction : public Function {
        public:
            int64_t dim = 0;
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                std::vector<int64_t> sh = grad_output.shape;
                std::vector<int64_t> st = grad_output.strides;
                sh.erase(sh.begin() + dim);
                st.erase(st.begin() + dim);
                Tensor view = make_view(grad_output, sh, st, grad_output.storage_offset);
                return {view.is_contiguous() ? view : contiguous(view)};
            }
            const char* name() const override { return "UnsqueezeFunction"; }
        };

        class ContiguousFunction : public Function {
        public:
            std::vector<Tensor> backward(const Tensor& grad_output) override {
                return {grad_output.is_contiguous() ? grad_output : contiguous(grad_output)};
            }
            const char* name() const override { return "ContiguousFunction"; }
        };
    }

    Tensor contiguous(const Tensor& x) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("contiguous: only Fp32 supported");
        }
        if (x.is_contiguous()) {
            return x;
        }
        Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
        copy_strided_to_contiguous(x, out);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<ContiguousFunction>();
        fn->record_input(x);
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor reshape(const Tensor& x, std::vector<int64_t> new_shape) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("reshape: only Fp32 supported");
        }
        auto resolved = resolve_new_shape(new_shape, x.numel());
        Tensor src = x.is_contiguous() ? x : contiguous(x);
        Tensor out = make_view(src, resolved, compute_contiguous_strides(resolved),
                               src.storage_offset);
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
        std::vector<int64_t> sh = x.shape;
        std::vector<int64_t> st = x.strides;
        std::swap(sh[dim_a], sh[dim_b]);
        std::swap(st[dim_a], st[dim_b]);
        Tensor out = make_view(x, sh, st, x.storage_offset);
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

    Tensor slice(const Tensor& x, int64_t dim, int64_t start, int64_t end) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("slice: only Fp32 supported");
        }
        const int64_t ndim = static_cast<int64_t>(x.shape.size());
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("slice: dim out of range");
        }
        if (start < 0 || end > x.shape[dim] || start >= end) {
            throw std::invalid_argument("slice: invalid range");
        }
        std::vector<int64_t> sh = x.shape;
        sh[dim] = end - start;
        const int64_t off = x.storage_offset + start * x.strides[dim];
        Tensor out = make_view(x, sh, x.strides, off);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<SliceFunction>();
        fn->record_input(x);
        fn->original_shape = x.shape;
        fn->dim = dim;
        fn->start = start;
        fn->end = end;
        fn->dtype = x.dtype;
        fn->device = x.device;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor squeeze(const Tensor& x, int64_t dim) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("squeeze: only Fp32 supported");
        }
        const int64_t ndim = static_cast<int64_t>(x.shape.size());
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("squeeze: dim out of range");
        }
        if (x.shape[dim] != 1) {
            throw std::invalid_argument("squeeze: dim size must be 1");
        }
        std::vector<int64_t> sh = x.shape;
        std::vector<int64_t> st = x.strides;
        const int64_t inserted_stride = st[dim];
        sh.erase(sh.begin() + dim);
        st.erase(st.begin() + dim);
        Tensor out = make_view(x, sh, st, x.storage_offset);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<SqueezeFunction>();
        fn->record_input(x);
        fn->dim = dim;
        fn->inserted_size_stride = inserted_stride;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }

    Tensor unsqueeze(const Tensor& x, int64_t dim) {
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("unsqueeze: only Fp32 supported");
        }
        const int64_t ndim = static_cast<int64_t>(x.shape.size());
        if (dim < 0 || dim > ndim) {
            throw std::invalid_argument("unsqueeze: dim out of range");
        }
        std::vector<int64_t> sh = x.shape;
        std::vector<int64_t> st = x.strides;
        const int64_t new_stride = (dim < ndim)
            ? x.strides[dim] * x.shape[dim]
            : 1;
        sh.insert(sh.begin() + dim, 1);
        st.insert(st.begin() + dim, new_stride);
        Tensor out = make_view(x, sh, st, x.storage_offset);
        if (!x.requires_grad) {
            return out;
        }
        auto fn = std::make_shared<UnsqueezeFunction>();
        fn->record_input(x);
        fn->dim = dim;
        out.requires_grad = true;
        out.grad_fn = fn;
        return out;
    }
}
