#include "ops/softmax.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        __global__ void softmax_kernel(const float* X, float* Y, int last) {
            extern __shared__ float shared[];
            const int row = blockIdx.x;
            const int tid = threadIdx.x;
            const int bsz = blockDim.x;

            const float* row_in  = X + static_cast<size_t>(row) * last;
            float*       row_out = Y + static_cast<size_t>(row) * last;

            float lmax = -INFINITY;
            for (int j = tid; j < last; j += bsz) {
                lmax = fmaxf(lmax, row_in[j]);
            }
            shared[tid] = lmax;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
                __syncthreads();
            }
            const float row_max = shared[0];
            __syncthreads();

            float lsum = 0.0f;
            for (int j = tid; j < last; j += bsz) {
                const float e = __expf(row_in[j] - row_max);
                row_out[j] = e;
                lsum += e;
            }
            shared[tid] = lsum;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            const float inv_sum = 1.0f / shared[0];
            __syncthreads();

            for (int j = tid; j < last; j += bsz) {
                row_out[j] *= inv_sum;
            }
        }
    }

    Tensor softmax_cuda(const Tensor& x) {
        Tensor out = Tensor::empty(x.shape, x.dtype, Device::CUDA);
        const int last = static_cast<int>(x.shape.back());
        const int64_t outer = x.numel() / last;

        dim3 grid(static_cast<unsigned int>(outer));
        dim3 block(BLOCK);
        softmax_kernel<<<grid, block, BLOCK * sizeof(float)>>>(
            static_cast<const float*>(x.data()),
            static_cast<float*>(out.data()),
            last);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
}
