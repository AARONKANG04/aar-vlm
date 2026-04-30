#include "ops/attention.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        __global__ void causal_mask_kernel(const float* x, float* y, 
                                           int64_t T, size_t total) {
            size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= total) return;
            const int64_t j = idx % T;
            const int64_t i = (idx / T) % T;
            y[idx] = (j > i) ? -INFINITY : x[idx];
        }

        __global__ void causal_mask_backward_kernel(const float* go, float* g,
                                                    int64_t T, size_t total) {
            size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (idx >= total) return;
            const int64_t j = idx % T;
            const int64_t i = (idx / T) % T;
            g[idx] = (j > i) ? 0.0f : go[idx];
        }

        unsigned int grid_for(size_t n) {
            return static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        }
    }

    Tensor apply_causal_mask_cuda(const Tensor& x) {
        Tensor out = Tensor::empty(x.shape, x.dtype, Device::CUDA);
        const size_t n = x.numel();
        const int64_t T = x.shape.back();
        causal_mask_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            T, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor causal_mask_backward_cuda(const Tensor& grad_out) {
        Tensor g = Tensor::empty(grad_out.shape, grad_out.dtype, Device::CUDA);
        const size_t n = grad_out.numel();
        const int64_t T = grad_out.shape.back();
        causal_mask_backward_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<float*>(g.data()),
            T, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return g;
    }
}