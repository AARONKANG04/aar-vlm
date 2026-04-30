#include "ops/adamw.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        __global__ void adamw_kernel(float* p, const float* g, float* mp, float* vp,
                                     float beta1, float beta2,
                                     float one_minus_b1, float one_minus_b2,
                                     float lr, float eps, float wd_factor,
                                     float bc1, float bc2,
                                     size_t n) {
            size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float gi = g[i];
            const float mi = beta1 * mp[i] + one_minus_b1 * gi;
            const float vi = beta2 * vp[i] + one_minus_b2 * gi * gi;
            mp[i] = mi;
            vp[i] = vi;
            const float m_hat = mi / bc1;
            const float v_hat = vi / bc2;
            p[i] = p[i] * wd_factor - lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }

    void adamw_step_cuda(Tensor& param, const Tensor& grad,
                         Tensor& m, Tensor& v,
                         float lr, float beta1, float beta2,
                         float eps, float weight_decay,
                         float bc1, float bc2) {
        const size_t n = param.numel();
        if (n == 0) return;
        const float one_minus_b1 = 1.0f - beta1;
        const float one_minus_b2 = 1.0f - beta2;
        const float wd_factor = 1.0f - lr * weight_decay;
        const unsigned int grid = static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        adamw_kernel<<<grid, BLOCK>>>(
            static_cast<float*>(param.data()),
            static_cast<const float*>(grad.data()),
            static_cast<float*>(m.data()),
            static_cast<float*>(v.data()),
            beta1, beta2, one_minus_b1, one_minus_b2,
            lr, eps, wd_factor, bc1, bc2, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
