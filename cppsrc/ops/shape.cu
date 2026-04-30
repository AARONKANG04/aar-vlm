#include "ops/shape.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;
        constexpr int MAX_DIMS = 8;

        __global__ void transpose_kernel(const float* X, float* Y,
                                         int ndim,
                                         const int64_t* in_strides,
                                         const int64_t* out_strides,
                                         const int64_t* out_shape,
                                         int dim_a, int dim_b,
                                         size_t total) {
            size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (flat >= total) return;
            int64_t idx[MAX_DIMS];
            size_t rem = flat;
            for (int d = 0; d < ndim; ++d) {
                idx[d] = rem / static_cast<size_t>(out_strides[d]);
                rem -= static_cast<size_t>(idx[d] * out_strides[d]);
            }
            int64_t tmp = idx[dim_a];
            idx[dim_a] = idx[dim_b];
            idx[dim_b] = tmp;
            size_t in_flat = 0;
            for (int d = 0; d < ndim; ++d) {
                in_flat += static_cast<size_t>(idx[d] * in_strides[d]);
            }
            Y[flat] = X[in_flat];
        }

        unsigned int grid_for(size_t n) {
            return static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        }
    }

    Tensor transpose_cuda(const Tensor& x, int64_t dim_a, int64_t dim_b) {
        const int ndim = static_cast<int>(x.shape.size());
        if (ndim > MAX_DIMS) {
            throw std::runtime_error("transpose_cuda: rank exceeds MAX_DIMS");
        }
        std::vector<int64_t> out_shape = x.shape;
        std::swap(out_shape[dim_a], out_shape[dim_b]);
        Tensor out = Tensor::empty(out_shape, x.dtype, Device::CUDA);

        int64_t in_strides_h[MAX_DIMS] = {0};
        int64_t out_strides_h[MAX_DIMS] = {0};
        int64_t out_shape_h[MAX_DIMS] = {0};
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            in_strides_h[i] = s;
            s *= x.shape[i];
        }
        s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            out_strides_h[i] = s;
            s *= out_shape[i];
        }
        for (int i = 0; i < ndim; ++i) out_shape_h[i] = out_shape[i];

        int64_t *in_strides_d, *out_strides_d, *out_shape_d;
        const size_t bytes = sizeof(int64_t) * MAX_DIMS;
        CUDA_CHECK(cudaMalloc(&in_strides_d, bytes));
        CUDA_CHECK(cudaMalloc(&out_strides_d, bytes));
        CUDA_CHECK(cudaMalloc(&out_shape_d, bytes));
        CUDA_CHECK(cudaMemcpy(in_strides_d, in_strides_h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out_strides_d, out_strides_h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(out_shape_d, out_shape_h, bytes, cudaMemcpyHostToDevice));

        const size_t total = out.numel();
        transpose_kernel<<<grid_for(total), BLOCK>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            ndim, in_strides_d, out_strides_d, out_shape_d,
            static_cast<int>(dim_a), static_cast<int>(dim_b),
            total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(in_strides_d);
        cudaFree(out_strides_d);
        cudaFree(out_shape_d);
        return out;
    }
}
