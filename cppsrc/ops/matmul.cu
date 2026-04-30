#include "ops/matmul.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        __global__ void matmul_naive_fp32(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int N, int K
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }

        __global__ void matmul_a_bt_naive_fp32(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int N, int K
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[row * K + k] * B[col * K + k];
            }
            C[row * N + col] = acc;
        }

        __global__ void matmul_at_b_naive_fp32(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            int M, int N, int K
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) return;

            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[k * M + row] * B[k * N + col];
            }
            C[row * N + col] = acc;
        }
    }

    Tensor matmul_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        const auto* A = static_cast<const float*>(a.data());
        const auto* B = static_cast<const float*>(b.data());
        auto* C = static_cast<float*>(out.data());

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_naive_fp32<<<grid, block>>>(A, B, C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return out;
    }

    Tensor matmul_a_bt_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[0]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        const auto* A = static_cast<const float*>(a.data());
        const auto* B = static_cast<const float*>(b.data());
        auto* C = static_cast<float*>(out.data());

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_a_bt_naive_fp32<<<grid, block>>>(A, B, C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return out;
    }

    Tensor matmul_at_b_cuda(const Tensor& a, const Tensor& b) {
        const int K = static_cast<int>(a.shape[0]);
        const int M = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        const auto* A = static_cast<const float*>(a.data());
        const auto* B = static_cast<const float*>(b.data());
        auto* C = static_cast<float*>(out.data());

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_at_b_naive_fp32<<<grid, block>>>(A, B, C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return out;
    }
}