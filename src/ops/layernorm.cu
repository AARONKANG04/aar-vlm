#include "ops/layernorm.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        __global__ void layernorm_kernel(const float* X, const float* W, const float* B,
                                         float* Y, int last, float eps) {
            extern __shared__ float shared[];
            const int row = blockIdx.x;
            const int tid = threadIdx.x;
            const int bsz = blockDim.x;
            const float D = static_cast<float>(last);

            const float* row_in  = X + static_cast<size_t>(row) * last;
            float*       row_out = Y + static_cast<size_t>(row) * last;

            float lsum = 0.0f;
            for (int j = tid; j < last; j += bsz) lsum += row_in[j];
            shared[tid] = lsum;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            const float mean = shared[0] / D;
            __syncthreads();

            float lsq = 0.0f;
            for (int j = tid; j < last; j += bsz) {
                const float d = row_in[j] - mean;
                lsq += d * d;
            }
            shared[tid] = lsq;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            const float var = shared[0] / D;
            const float rstd = rsqrtf(var + eps);
            __syncthreads();

            for (int j = tid; j < last; j += bsz) {
                row_out[j] = (row_in[j] - mean) * rstd * W[j] + B[j];
            }
        }
    }

    Tensor layernorm_cuda(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
        Tensor out = Tensor::empty(x.shape, x.dtype, Device::CUDA);
        const int last = static_cast<int>(x.shape.back());
        const int64_t outer = x.numel() / last;

        dim3 grid(static_cast<unsigned int>(outer));
        dim3 block(BLOCK);
        layernorm_kernel<<<grid, block, BLOCK * sizeof(float)>>>(
            static_cast<const float*>(x.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(out.data()),
            last, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
}
