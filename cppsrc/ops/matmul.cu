#include "ops/matmul.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BM = 128;
        constexpr int BN = 128;
        constexpr int BK = 16;
        constexpr int TM = 8;
        constexpr int TN = 8;
        constexpr int TX = BN / TN;
        constexpr int TY = BM / TM;
        constexpr int THREADS = TX * TY;
        constexpr int A_LOADS = (BM * BK) / THREADS;
        constexpr int B_LOADS = (BK * BN) / THREADS;

        __device__ __forceinline__ void mma_compute(
                const float (&As)[BM][BK],
                const float (&Bs)[BK][BN],
                int ty, int tx,
                float (&acc)[TM][TN]) {
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a_reg[TM];
                float b_reg[TN];
                #pragma unroll
                for (int i = 0; i < TM; ++i) a_reg[i] = As[ty * TM + i][kk];
                #pragma unroll
                for (int j = 0; j < TN; ++j) b_reg[j] = Bs[kk][tx * TN + j];
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] += a_reg[i] * b_reg[j];
                    }
                }
            }
        }

        __device__ __forceinline__ void write_back(
                float* C, int M, int N,
                int by, int bx, int ty, int tx,
                const float (&acc)[TM][TN]) {
            const int row_base = by * BM + ty * TM;
            const int col_base = bx * BN + tx * TN;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int r = row_base + i;
                if (r >= M) continue;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int c = col_base + j;
                    if (c < N) C[r * N + c] = acc[i][j];
                }
            }
        }

        __global__ void matmul_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];

            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * TX + tx;

            float acc[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

            const int num_k_blocks = (K + BK - 1) / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int m_local = idx / BK;
                    const int k_local = idx % BK;
                    const int gm = by * BM + m_local;
                    const int gk = k_start + k_local;
                    As[m_local][k_local] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
                }
                #pragma unroll
                for (int s = 0; s < B_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int k_local = idx / BN;
                    const int n_local = idx % BN;
                    const int gk = k_start + k_local;
                    const int gn = bx * BN + n_local;
                    Bs[k_local][n_local] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
                }
                __syncthreads();

                mma_compute(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back(C, M, N, by, bx, ty, tx, acc);
        }

        __global__ void matmul_a_bt_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];

            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * TX + tx;

            float acc[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

            const int num_k_blocks = (K + BK - 1) / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int m_local = idx / BK;
                    const int k_local = idx % BK;
                    const int gm = by * BM + m_local;
                    const int gk = k_start + k_local;
                    As[m_local][k_local] = (gm < M && gk < K) ? A[gm * K + gk] : 0.0f;
                }
                #pragma unroll
                for (int s = 0; s < B_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int n_local = idx / BK;
                    const int k_local = idx % BK;
                    const int gn = bx * BN + n_local;
                    const int gk = k_start + k_local;
                    Bs[k_local][n_local] = (gn < N && gk < K) ? B[gn * K + gk] : 0.0f;
                }
                __syncthreads();

                mma_compute(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back(C, M, N, by, bx, ty, tx, acc);
        }

        __global__ void matmul_at_b_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];

            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * TX + tx;

            float acc[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

            const int num_k_blocks = (K + BK - 1) / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int k_local = idx / BM;
                    const int m_local = idx % BM;
                    const int gk = k_start + k_local;
                    const int gm = by * BM + m_local;
                    As[m_local][k_local] = (gk < K && gm < M) ? A[gk * M + gm] : 0.0f;
                }
                #pragma unroll
                for (int s = 0; s < B_LOADS; ++s) {
                    const int idx = s * THREADS + tid;
                    const int k_local = idx / BN;
                    const int n_local = idx % BN;
                    const int gk = k_start + k_local;
                    const int gn = bx * BN + n_local;
                    Bs[k_local][n_local] = (gk < K && gn < N) ? B[gk * N + gn] : 0.0f;
                }
                __syncthreads();

                mma_compute(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back(C, M, N, by, bx, ty, tx, acc);
        }
    }

    Tensor matmul_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        dim3 block(TX, TY);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor matmul_a_bt_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[0]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        dim3 block(TX, TY);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_a_bt_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor matmul_at_b_cuda(const Tensor& a, const Tensor& b) {
        const int K = static_cast<int>(a.shape[0]);
        const int M = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        dim3 block(TX, TY);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_at_b_tiled_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
}
