#include "ops/embedding.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        __global__ void embedding_forward_kernel(const float* W, const int64_t* I, float* O,
                                                  int64_t N, int64_t D, int64_t V) {
            const int64_t i = blockIdx.y;
            const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N || k >= D) return;
            const int64_t idx = I[i];
            if (idx < 0 || idx >= V) return;
            O[i * D + k] = W[idx * D + k];
        }

        __global__ void embedding_backward_kernel(const float* G, const int64_t* I, float* DW,
                                                   int64_t N, int64_t D) {
            const int64_t i = blockIdx.y;
            const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N || k >= D) return;
            const int64_t idx = I[i];
            atomicAdd(&DW[idx * D + k], G[i * D + k]);
        }
    }

    Tensor embedding_cuda(const Tensor& weight, const Tensor& ids) {
        const int64_t D = weight.shape[1];
        const int64_t V = weight.shape[0];
        std::vector<int64_t> out_shape = ids.shape;
        out_shape.push_back(D);
        Tensor out = Tensor::empty(out_shape, weight.dtype, Device::CUDA);
        const int64_t N = static_cast<int64_t>(ids.numel());
        if (N == 0 || D == 0) return out;
        constexpr int BLOCK = 128;
        dim3 grid(static_cast<unsigned int>((D + BLOCK - 1) / BLOCK),
                  static_cast<unsigned int>(N));
        embedding_forward_kernel<<<grid, BLOCK>>>(
            static_cast<const float*>(weight.data()),
            static_cast<const int64_t*>(ids.data()),
            static_cast<float*>(out.data()),
            N, D, V);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor embedding_backward_cuda(const Tensor& grad_out, const Tensor& ids,
                                    int64_t vocab_size) {
        const int64_t D = grad_out.shape.back();
        Tensor dW = Tensor::zeros({vocab_size, D}, grad_out.dtype, Device::CUDA);
        const int64_t N = static_cast<int64_t>(ids.numel());
        if (N == 0 || D == 0) return dW;
        constexpr int BLOCK = 128;
        dim3 grid(static_cast<unsigned int>((D + BLOCK - 1) / BLOCK),
                  static_cast<unsigned int>(N));
        embedding_backward_kernel<<<grid, BLOCK>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<const int64_t*>(ids.data()),
            static_cast<float*>(dW.data()),
            N, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return dW;
    }
}
