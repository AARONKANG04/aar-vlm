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

        __global__ void relu_backward_kernel(const float* go, const float* x, float* g, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) g[i] = x[i] > 0.0f ? go[i] : 0.0f;
        }

        __global__ void fill_kernel(float* p, float v, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) p[i] = v;
        }

        __global__ void scaled_add_kernel(float* d, const float* s, float a, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) d[i] += a * s[i];
        }

        __global__ void sub_kernel(const float* a, const float* b, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] - b[i];
        }

        __global__ void neg_kernel(const float* a, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) c[i] = -a[i];
        }

        __global__ void gelu_kernel(const float* a, float* c, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) {
                const float x = a[i];
                c[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
            }
        }

        __global__ void gelu_backward_kernel(const float* go, const float* x, float* g, size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) {
                const float xi = x[i];
                const float cdf = 0.5f * (1.0f + erff(xi * 0.70710678118654752440f));
                const float pdf = 0.39894228040143267794f * expf(-0.5f * xi * xi);
                g[i] = go[i] * (cdf + xi * pdf);
            }
        }

        __global__ void add_bias_kernel(const float* x, const float* b, float* y, size_t n, size_t D) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) y[i] = x[i] + b[i % D];
        }

        __global__ void bias_grad_kernel(const float* g, float* db, size_t n, size_t D) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) atomicAdd(&db[i % D], g[i]);
        }

        __global__ void sum_all_kernel(const float* X, float* out, size_t n) {
            extern __shared__ float shared[];
            const int tid = threadIdx.x;
            const int bsz = blockDim.x;

            float local = 0.0f;
            for (size_t i = tid; i < n; i += bsz) local += X[i];
            shared[tid] = local;
            __syncthreads();

            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }

            if (tid == 0) *out = shared[0];
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

    Tensor relu_backward_cuda(const Tensor& grad_out, const Tensor& x) {
        Tensor g = Tensor::empty(grad_out.shape, grad_out.dtype, Device::CUDA);
        const size_t n = grad_out.numel();
        relu_backward_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<const float*>(x.data()),
            static_cast<float*>(g.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return g;
    }

    void fill_cuda(Tensor& out, float v) {
        const size_t n = out.numel();
        if (n == 0) return;
        fill_kernel<<<grid_for(n), BLOCK>>>(static_cast<float*>(out.data()), v, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void scaled_add_inplace_cuda(Tensor& dst, const Tensor& src, float alpha) {
        const size_t n = dst.numel();
        if (n == 0) return;
        scaled_add_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<float*>(dst.data()),
            static_cast<const float*>(src.data()),
            alpha,
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Tensor sum_all_cuda(const Tensor& a) {
        Tensor out = Tensor::empty({}, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        sum_all_kernel<<<1, BLOCK, BLOCK * sizeof(float)>>>(
            static_cast<const float*>(a.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor sub_cuda(const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        sub_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor neg_cuda(const Tensor& a) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        neg_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor gelu_cuda(const Tensor& a) {
        Tensor out = Tensor::empty(a.shape, a.dtype, Device::CUDA);
        const size_t n = a.numel();
        gelu_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(a.data()),
            static_cast<float*>(out.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor gelu_backward_cuda(const Tensor& grad_out, const Tensor& x) {
        Tensor g = Tensor::empty(x.shape, x.dtype, Device::CUDA);
        const size_t n = x.numel();
        gelu_backward_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<const float*>(x.data()),
            static_cast<float*>(g.data()),
            n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return g;
    }

    Tensor add_bias_cuda(const Tensor& x, const Tensor& bias) {
        Tensor out = Tensor::empty(x.shape, x.dtype, Device::CUDA);
        const size_t n = x.numel();
        const size_t D = bias.numel();
        add_bias_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(x.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(out.data()),
            n, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor bias_grad_cuda(const Tensor& grad_out, int64_t D) {
        Tensor db = Tensor::zeros({D}, grad_out.dtype, Device::CUDA);
        const size_t n = grad_out.numel();
        bias_grad_kernel<<<grid_for(n), BLOCK>>>(
            static_cast<const float*>(grad_out.data()),
            static_cast<float*>(db.data()),
            n, static_cast<size_t>(D));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return db;
    }
}
