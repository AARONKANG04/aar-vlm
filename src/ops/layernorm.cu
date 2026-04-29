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

        __global__ void layernorm_with_stats_kernel(const float* X, const float* W, const float* B,
                                                    float* Y, float* MEAN, float* RSTD,
                                                    int last, float eps) {
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

            if (tid == 0) {
                MEAN[row] = mean;
                RSTD[row] = rstd;
            }

            for (int j = tid; j < last; j += bsz) {
                row_out[j] = (row_in[j] - mean) * rstd * W[j] + B[j];
            }
        }

        __global__ void layernorm_backward_kernel(
            const float* GY, const float* X, const float* W,
            const float* MEAN, const float* RSTD,
            float* DX, float* DW, float* DB,
            int last
        ) {
            extern __shared__ float shared[];
            const int row = blockIdx.x;
            const int tid = threadIdx.x;
            const int bsz = blockDim.x;
            const float D = static_cast<float>(last);
            const float inv_D = 1.0f / D;

            const float* gy_row = GY + static_cast<size_t>(row) * last;
            const float* x_row  = X  + static_cast<size_t>(row) * last;
            float*       dx_row = DX + static_cast<size_t>(row) * last;
            const float mean_i = MEAN[row];
            const float rstd_i = RSTD[row];

            float lsum_g = 0.0f;
            float lsum_g_x = 0.0f;
            for (int j = tid; j < last; j += bsz) {
                const float normed = (x_row[j] - mean_i) * rstd_i;
                const float g = gy_row[j] * W[j];
                lsum_g += g;
                lsum_g_x += g * normed;
            }

            shared[tid] = lsum_g;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            const float sum_g = shared[0];
            __syncthreads();

            shared[tid] = lsum_g_x;
            __syncthreads();
            for (int s = bsz / 2; s > 0; s >>= 1) {
                if (tid < s) shared[tid] += shared[tid + s];
                __syncthreads();
            }
            const float sum_g_x = shared[0];
            __syncthreads();

            for (int j = tid; j < last; j += bsz) {
                const float normed = (x_row[j] - mean_i) * rstd_i;
                const float g = gy_row[j] * W[j];
                dx_row[j] = rstd_i * (g - sum_g * inv_D - normed * sum_g_x * inv_D);
                atomicAdd(&DW[j], gy_row[j] * normed);
                atomicAdd(&DB[j], gy_row[j]);
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

    void layernorm_with_stats_cuda(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                                    Tensor& y, Tensor& mean, Tensor& rstd) {
        const int last = static_cast<int>(x.shape.back());
        const int64_t outer = x.numel() / last;

        dim3 grid(static_cast<unsigned int>(outer));
        dim3 block(BLOCK);
        layernorm_with_stats_kernel<<<grid, block, BLOCK * sizeof(float)>>>(
            static_cast<const float*>(x.data()),
            static_cast<const float*>(weight.data()),
            static_cast<const float*>(bias.data()),
            static_cast<float*>(y.data()),
            static_cast<float*>(mean.data()),
            static_cast<float*>(rstd.data()),
            last, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void layernorm_backward_cuda(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                  const Tensor& mean, const Tensor& rstd,
                                  Tensor& dx, Tensor& dw, Tensor& db) {
        const int last = static_cast<int>(x.shape.back());
        const int64_t outer = x.numel() / last;

        CUDA_CHECK(cudaMemset(dw.data(), 0, dw.nbytes()));
        CUDA_CHECK(cudaMemset(db.data(), 0, db.nbytes()));

        dim3 grid(static_cast<unsigned int>(outer));
        dim3 block(BLOCK);
        layernorm_backward_kernel<<<grid, block, BLOCK * sizeof(float)>>>(
            static_cast<const float*>(grad_y.data()),
            static_cast<const float*>(x.data()),
            static_cast<const float*>(w.data()),
            static_cast<const float*>(mean.data()),
            static_cast<const float*>(rstd.data()),
            static_cast<float*>(dx.data()),
            static_cast<float*>(dw.data()),
            static_cast<float*>(db.data()),
            last);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
