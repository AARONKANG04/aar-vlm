#include "ops/dropout.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"
#include "core/rng.hpp"

namespace vlm {
    namespace {
        __global__ void dropout_kernel(const float* X, float* O, float* M,
                                        size_t n, float p, float scale,
                                        uint64_t seed) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const float r = philox_uniform(seed, static_cast<uint64_t>(i));
            const float keep_scaled = (r >= p) ? scale : 0.0f;
            M[i] = keep_scaled;
            O[i] = X[i] * keep_scaled;
        }
    }

    void dropout_cuda(const Tensor& x, float p, uint64_t seed,
                      Tensor& out, Tensor& mask) {
        const size_t n = x.numel();
        if (n == 0) return;
        constexpr int BLOCK = 256;
        const unsigned int grid = static_cast<unsigned int>((n + BLOCK - 1) / BLOCK);
        const float scale = 1.0f / (1.0f - p);
        dropout_kernel<<<grid, BLOCK>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            static_cast<float*>(mask.data()),
            n, p, scale, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
