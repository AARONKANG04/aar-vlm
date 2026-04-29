#include "ops/elementwise.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        __global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }

        __global__ void mul_kernel(const float* a, const float* b, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] * b[i];
        }

        __global__ void relu_kernel(const float* a, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] > 0.0f ? a[i] : 0.0f;
        }

        unsigned int grid_for(size_t n) {
            return static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        }
    }

    Tensor add_cuda(const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        add_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor mul_cuda(const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        mul_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor relu_cuda(const Tensor& a) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        relu_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
}
