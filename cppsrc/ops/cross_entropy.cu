#include "ops/cross_entropy.hpp"

#include <cuda_runtime.h>
#include <cmath>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BLOCK = 256;

        // One block per row. Computes per-row loss = lse - logits[t], and writes 0 if
        // t == ignore_index. Also writes 1.0 / 0.0 to active_flag for active reduction.
        __global__ void ce_forward_per_row_kernel(const float* L, const int64_t* T,
                                                   float* row_loss, float* active_flag,
                                                   int64_t V, int64_t ignore_index) {
            const int64_t i = blockIdx.x;
            const int64_t t = T[i];
            __shared__ float buf[BLOCK];
            const float* row = L + i * V;

            if (t == ignore_index) {
                if (threadIdx.x == 0) {
                    row_loss[i] = 0.0f;
                    active_flag[i] = 0.0f;
                }
                return;
            }

            // max reduce
            float m = -INFINITY;
            for (int64_t v = threadIdx.x; v < V; v += BLOCK) {
                m = fmaxf(m, row[v]);
            }
            buf[threadIdx.x] = m;
            __syncthreads();
            for (int s = BLOCK / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]);
                __syncthreads();
            }
            const float row_max = buf[0];

            // sum exp reduce
            float s = 0.0f;
            for (int64_t v = threadIdx.x; v < V; v += BLOCK) {
                s += expf(row[v] - row_max);
            }
            buf[threadIdx.x] = s;
            __syncthreads();
            for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
                __syncthreads();
            }
            const float row_sum = buf[0];

            if (threadIdx.x == 0) {
                const float lse = row_max + logf(row_sum);
                row_loss[i] = lse - row[t];
                active_flag[i] = 1.0f;
            }
        }

        __global__ void ce_reduce_mean_kernel(const float* row_loss, const float* active_flag,
                                               int64_t N, float* out) {
            __shared__ float lbuf[BLOCK];
            __shared__ float abuf[BLOCK];
            float lacc = 0.0f, aacc = 0.0f;
            for (int64_t i = threadIdx.x; i < N; i += BLOCK) {
                lacc += row_loss[i];
                aacc += active_flag[i];
            }
            lbuf[threadIdx.x] = lacc;
            abuf[threadIdx.x] = aacc;
            __syncthreads();
            for (int s = BLOCK / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    lbuf[threadIdx.x] += lbuf[threadIdx.x + s];
                    abuf[threadIdx.x] += abuf[threadIdx.x + s];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                out[0] = abuf[0] > 0.0f ? lbuf[0] / abuf[0] : 0.0f;
            }
        }

        // One block per row. Writes (softmax(row) - one_hot(t)) * (go / active) into dL row,
        // or zeros if t == ignore_index.
        __global__ void ce_backward_per_row_kernel(const float* L, const int64_t* T,
                                                    float* DL,
                                                    int64_t V, int64_t ignore_index,
                                                    float scale) {
            const int64_t i = blockIdx.x;
            const int64_t t = T[i];
            __shared__ float buf[BLOCK];
            const float* row = L + i * V;
            float* drow = DL + i * V;

            if (t == ignore_index) {
                for (int64_t v = threadIdx.x; v < V; v += BLOCK) drow[v] = 0.0f;
                return;
            }

            float m = -INFINITY;
            for (int64_t v = threadIdx.x; v < V; v += BLOCK) {
                m = fmaxf(m, row[v]);
            }
            buf[threadIdx.x] = m;
            __syncthreads();
            for (int s = BLOCK / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) buf[threadIdx.x] = fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]);
                __syncthreads();
            }
            const float row_max = buf[0];

            float s = 0.0f;
            for (int64_t v = threadIdx.x; v < V; v += BLOCK) {
                s += expf(row[v] - row_max);
            }
            buf[threadIdx.x] = s;
            __syncthreads();
            for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
                __syncthreads();
            }
            const float inv_sum = 1.0f / buf[0];

            for (int64_t v = threadIdx.x; v < V; v += BLOCK) {
                drow[v] = expf(row[v] - row_max) * inv_sum * scale;
            }
            __syncthreads();
            if (threadIdx.x == 0) drow[t] -= scale;
        }

        __global__ void count_active_kernel(const int64_t* T, int64_t N,
                                             int64_t ignore_index, int* active_out) {
            __shared__ int buf[BLOCK];
            int acc = 0;
            for (int64_t i = threadIdx.x; i < N; i += BLOCK) {
                if (T[i] != ignore_index) ++acc;
            }
            buf[threadIdx.x] = acc;
            __syncthreads();
            for (int s = BLOCK / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) active_out[0] = buf[0];
        }
    }

    Tensor cross_entropy_cuda(const Tensor& logits, const Tensor& targets,
                               int64_t ignore_index) {
        const int64_t N = logits.shape[0];
        const int64_t V = logits.shape[1];
        Tensor row_loss = Tensor::empty({N}, DType::Fp32, Device::CUDA);
        Tensor active = Tensor::empty({N}, DType::Fp32, Device::CUDA);
        ce_forward_per_row_kernel<<<static_cast<unsigned int>(N), BLOCK>>>(
            static_cast<const float*>(logits.data()),
            static_cast<const int64_t*>(targets.data()),
            static_cast<float*>(row_loss.data()),
            static_cast<float*>(active.data()),
            V, ignore_index);
        CUDA_CHECK(cudaGetLastError());

        Tensor out = Tensor::empty({}, DType::Fp32, Device::CUDA);
        ce_reduce_mean_kernel<<<1, BLOCK>>>(
            static_cast<const float*>(row_loss.data()),
            static_cast<const float*>(active.data()),
            N, static_cast<float*>(out.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor cross_entropy_backward_cuda(const Tensor& grad_output,
                                        const Tensor& logits,
                                        const Tensor& targets,
                                        int64_t ignore_index) {
        const int64_t N = logits.shape[0];
        const int64_t V = logits.shape[1];
        Tensor dL = Tensor::empty({N, V}, DType::Fp32, Device::CUDA);

        // Copy go scalar to host; count active rows via kernel.
        float go;
        copy_bytes(&go, Device::CPU, grad_output.data(), Device::CUDA, sizeof(float));

        int* active_d = nullptr;
        CUDA_CHECK(cudaMalloc(&active_d, sizeof(int)));
        count_active_kernel<<<1, BLOCK>>>(
            static_cast<const int64_t*>(targets.data()), N, ignore_index, active_d);
        int active_h = 0;
        CUDA_CHECK(cudaMemcpy(&active_h, active_d, sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(active_d);

        if (active_h == 0) {
            CUDA_CHECK(cudaMemset(dL.data(), 0, dL.nbytes()));
            return dL;
        }
        const float scale = go / static_cast<float>(active_h);
        ce_backward_per_row_kernel<<<static_cast<unsigned int>(N), BLOCK>>>(
            static_cast<const float*>(logits.data()),
            static_cast<const int64_t*>(targets.data()),
            static_cast<float*>(dL.data()),
            V, ignore_index, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return dL;
    }
}
