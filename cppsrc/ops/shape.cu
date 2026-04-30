#include "ops/shape.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;
        constexpr int MAX_DIMS = 8;

        __global__ void strided_to_contiguous_kernel(const float* X, float* Y,
                                                     int ndim,
                                                     const int64_t* src_strides,
                                                     const int64_t* dst_strides,
                                                     size_t total) {
            size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (flat >= total) return;
            size_t rem = flat;
            int64_t src_off = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t i = rem / static_cast<size_t>(dst_strides[d]);
                rem -= static_cast<size_t>(i * dst_strides[d]);
                src_off += i * src_strides[d];
            }
            Y[flat] = X[src_off];
        }

        __global__ void contiguous_to_strided_kernel(const float* X, float* Y,
                                                     int ndim,
                                                     const int64_t* src_strides,
                                                     const int64_t* dst_strides,
                                                     size_t total) {
            size_t flat = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (flat >= total) return;
            size_t rem = flat;
            int64_t dst_off = 0;
            for (int d = 0; d < ndim; ++d) {
                int64_t i = rem / static_cast<size_t>(src_strides[d]);
                rem -= static_cast<size_t>(i * src_strides[d]);
                dst_off += i * dst_strides[d];
            }
            Y[dst_off] = X[flat];
        }

        unsigned int grid_for(size_t n) {
            return static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        }
    }

    void contiguous_cuda(const Tensor& src, Tensor& dst) {
        const int ndim = static_cast<int>(src.shape.size());
        if (ndim > MAX_DIMS) {
            throw std::runtime_error("contiguous_cuda: rank exceeds MAX_DIMS");
        }
        const size_t total = src.numel();
        if (total == 0) return;
        if (ndim == 0) {
            CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), sizeof(float),
                                   cudaMemcpyDeviceToDevice));
            return;
        }

        int64_t src_strides_h[MAX_DIMS] = {0};
        int64_t dst_strides_h[MAX_DIMS] = {0};
        for (int i = 0; i < ndim; ++i) src_strides_h[i] = src.strides[i];
        for (int i = 0; i < ndim; ++i) dst_strides_h[i] = dst.strides[i];

        int64_t *src_strides_d, *dst_strides_d;
        const size_t bytes = sizeof(int64_t) * MAX_DIMS;
        CUDA_CHECK(cudaMalloc(&src_strides_d, bytes));
        CUDA_CHECK(cudaMalloc(&dst_strides_d, bytes));
        CUDA_CHECK(cudaMemcpy(src_strides_d, src_strides_h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst_strides_d, dst_strides_h, bytes, cudaMemcpyHostToDevice));

        strided_to_contiguous_kernel<<<grid_for(total), BLOCK>>>(
            static_cast<const float*>(src.data()),
            static_cast<float*>(dst.data()),
            ndim, src_strides_d, dst_strides_d, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(src_strides_d);
        cudaFree(dst_strides_d);
    }

    void copy_contiguous_into_strided_cuda(const Tensor& src, Tensor& dst) {
        const int ndim = static_cast<int>(src.shape.size());
        if (ndim > MAX_DIMS) {
            throw std::runtime_error("copy_contiguous_into_strided_cuda: rank exceeds MAX_DIMS");
        }
        const size_t total = src.numel();
        if (total == 0) return;
        if (ndim == 0) {
            CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), sizeof(float),
                                   cudaMemcpyDeviceToDevice));
            return;
        }

        int64_t src_strides_h[MAX_DIMS] = {0};
        int64_t dst_strides_h[MAX_DIMS] = {0};
        int64_t s = 1;
        for (int i = ndim - 1; i >= 0; --i) {
            src_strides_h[i] = s;
            s *= src.shape[i];
        }
        for (int i = 0; i < ndim; ++i) dst_strides_h[i] = dst.strides[i];

        int64_t *src_strides_d, *dst_strides_d;
        const size_t bytes = sizeof(int64_t) * MAX_DIMS;
        CUDA_CHECK(cudaMalloc(&src_strides_d, bytes));
        CUDA_CHECK(cudaMalloc(&dst_strides_d, bytes));
        CUDA_CHECK(cudaMemcpy(src_strides_d, src_strides_h, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dst_strides_d, dst_strides_h, bytes, cudaMemcpyHostToDevice));

        contiguous_to_strided_kernel<<<grid_for(total), BLOCK>>>(
            static_cast<const float*>(src.data()),
            static_cast<float*>(dst.data()),
            ndim, src_strides_d, dst_strides_d, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(src_strides_d);
        cudaFree(dst_strides_d);
    }
}
