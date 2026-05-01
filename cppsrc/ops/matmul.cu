#include "ops/matmul.hpp"

#include <algorithm>
#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        template <int BM, int BN, int BK, int TM, int TN>
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

        template <int BM, int BN, int TM, int TN>
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

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_LOADS = (BM * BK) / THREADS;
            constexpr int B_LOADS = (BK * BN) / THREADS;
            static_assert((BM * BK) % THREADS == 0, "BM*BK must be divisible by THREADS");
            static_assert((BK * BN) % THREADS == 0, "BK*BN must be divisible by THREADS");

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

                mma_compute<BM, BN, BK, TM, TN>(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_a_bt_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_LOADS = (BM * BK) / THREADS;
            constexpr int B_LOADS = (BK * BN) / THREADS;
            static_assert((BM * BK) % THREADS == 0, "BM*BK must be divisible by THREADS");
            static_assert((BK * BN) % THREADS == 0, "BK*BN must be divisible by THREADS");

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

                mma_compute<BM, BN, BK, TM, TN>(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_at_b_tiled_kernel(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_LOADS = (BM * BK) / THREADS;
            constexpr int B_LOADS = (BK * BN) / THREADS;
            static_assert((BM * BK) % THREADS == 0, "BM*BK must be divisible by THREADS");
            static_assert((BK * BN) % THREADS == 0, "BK*BN must be divisible by THREADS");

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

                mma_compute<BM, BN, BK, TM, TN>(As, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __device__ __forceinline__ void mma_compute_v3(
                const float (&AsT)[BK][BM],
                const float (&Bs)[BK][BN],
                int ty, int tx,
                float (&acc)[TM][TN]) {
            static_assert(TM % 4 == 0, "TM must be divisible by 4 for float4 LDS");
            static_assert(TN % 4 == 0, "TN must be divisible by 4 for float4 LDS");
            constexpr int TM4 = TM / 4;
            constexpr int TN4 = TN / 4;

            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float a_reg[TM];
                float b_reg[TN];
                #pragma unroll
                for (int i = 0; i < TM4; ++i) {
                    float4 v = *reinterpret_cast<const float4*>(&AsT[kk][ty * TM + i * 4]);
                    a_reg[i * 4 + 0] = v.x;
                    a_reg[i * 4 + 1] = v.y;
                    a_reg[i * 4 + 2] = v.z;
                    a_reg[i * 4 + 3] = v.w;
                }
                #pragma unroll
                for (int j = 0; j < TN4; ++j) {
                    float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tx * TN + j * 4]);
                    b_reg[j * 4 + 0] = v.x;
                    b_reg[j * 4 + 1] = v.y;
                    b_reg[j * 4 + 2] = v.z;
                    b_reg[j * 4 + 3] = v.w;
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] += a_reg[i] * b_reg[j];
                    }
                }
            }
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_tiled_kernel_v3(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_F4_LOADS = (BM * BK) / (THREADS * 4);
            constexpr int B_F4_LOADS = (BK * BN) / (THREADS * 4);
            static_assert((BM * BK) % (THREADS * 4) == 0, "BM*BK must be divisible by THREADS*4");
            static_assert((BK * BN) % (THREADS * 4) == 0, "BK*BN must be divisible by THREADS*4");
            static_assert(BK % 4 == 0, "BK must be divisible by 4");
            static_assert(BN % 4 == 0, "BN must be divisible by 4");
            static_assert(TM % 4 == 0, "TM must be divisible by 4");
            static_assert(TN % 4 == 0, "TN must be divisible by 4");

            __shared__ float AsT[BK][BM];
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

            const int num_k_blocks = K / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int m_local = elem_idx / BK;
                    const int k_local = elem_idx % BK;
                    const int gm = by * BM + m_local;
                    const int gk = k_start + k_local;
                    float4 v = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                    AsT[k_local + 0][m_local] = v.x;
                    AsT[k_local + 1][m_local] = v.y;
                    AsT[k_local + 2][m_local] = v.z;
                    AsT[k_local + 3][m_local] = v.w;
                }
                #pragma unroll
                for (int s = 0; s < B_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int k_local = elem_idx / BN;
                    const int n_local = elem_idx % BN;
                    const int gk = k_start + k_local;
                    const int gn = bx * BN + n_local;
                    float4 v = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
                    *reinterpret_cast<float4*>(&Bs[k_local][n_local]) = v;
                }
                __syncthreads();

                mma_compute_v3<BM, BN, BK, TM, TN>(AsT, Bs, ty, tx, acc);
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_tiled_kernel_v4(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_F4_LOADS = (BM * BK) / (THREADS * 4);
            constexpr int B_F4_LOADS = (BK * BN) / (THREADS * 4);
            static_assert((BM * BK) % (THREADS * 4) == 0, "BM*BK must be divisible by THREADS*4");
            static_assert((BK * BN) % (THREADS * 4) == 0, "BK*BN must be divisible by THREADS*4");
            static_assert(BK % 4 == 0, "BK must be divisible by 4");
            static_assert(BN % 4 == 0, "BN must be divisible by 4");
            static_assert(TM % 4 == 0, "TM must be divisible by 4");
            static_assert(TN % 4 == 0, "TN must be divisible by 4");

            __shared__ float AsT[BK][BM + 4];
            __shared__ float Bs[BK][BN + 4];

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

            const int num_k_blocks = K / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int m_local = elem_idx / BK;
                    const int k_local = elem_idx % BK;
                    const int gm = by * BM + m_local;
                    const int gk = k_start + k_local;
                    float4 v = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                    AsT[k_local + 0][m_local] = v.x;
                    AsT[k_local + 1][m_local] = v.y;
                    AsT[k_local + 2][m_local] = v.z;
                    AsT[k_local + 3][m_local] = v.w;
                }
                #pragma unroll
                for (int s = 0; s < B_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int k_local = elem_idx / BN;
                    const int n_local = elem_idx % BN;
                    const int gk = k_start + k_local;
                    const int gn = bx * BN + n_local;
                    float4 v = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
                    *reinterpret_cast<float4*>(&Bs[k_local][n_local]) = v;
                }
                __syncthreads();

                constexpr int TM4 = TM / 4;
                constexpr int TN4 = TN / 4;

                #pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float a_reg[TM];
                    float b_reg[TN];
                    #pragma unroll
                    for (int i = 0; i < TM4; ++i) {
                        float4 v = *reinterpret_cast<const float4*>(&AsT[kk][ty * TM + i * 4]);
                        a_reg[i * 4 + 0] = v.x;
                        a_reg[i * 4 + 1] = v.y;
                        a_reg[i * 4 + 2] = v.z;
                        a_reg[i * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int j = 0; j < TN4; ++j) {
                        float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tx * TN + j * 4]);
                        b_reg[j * 4 + 0] = v.x;
                        b_reg[j * 4 + 1] = v.y;
                        b_reg[j * 4 + 2] = v.z;
                        b_reg[j * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] += a_reg[i] * b_reg[j];
                        }
                    }
                }
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_a_bt_tiled_kernel_v4(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_F4_LOADS = (BM * BK) / (THREADS * 4);
            constexpr int B_F4_LOADS = (BK * BN) / (THREADS * 4);
            static_assert((BM * BK) % (THREADS * 4) == 0, "BM*BK must be divisible by THREADS*4");
            static_assert((BK * BN) % (THREADS * 4) == 0, "BK*BN must be divisible by THREADS*4");
            static_assert(BK % 4 == 0, "BK must be divisible by 4");
            static_assert(TM % 4 == 0, "TM must be divisible by 4");
            static_assert(TN % 4 == 0, "TN must be divisible by 4");

            __shared__ float AsT[BK][BM + 4];
            __shared__ float BsT[BK][BN + 4];

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

            const int num_k_blocks = K / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int m_local = elem_idx / BK;
                    const int k_local = elem_idx % BK;
                    const int gm = by * BM + m_local;
                    const int gk = k_start + k_local;
                    float4 v = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                    AsT[k_local + 0][m_local] = v.x;
                    AsT[k_local + 1][m_local] = v.y;
                    AsT[k_local + 2][m_local] = v.z;
                    AsT[k_local + 3][m_local] = v.w;
                }
                #pragma unroll
                for (int s = 0; s < B_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int n_local = elem_idx / BK;
                    const int k_local = elem_idx % BK;
                    const int gn = bx * BN + n_local;
                    const int gk = k_start + k_local;
                    float4 v = *reinterpret_cast<const float4*>(&B[gn * K + gk]);
                    BsT[k_local + 0][n_local] = v.x;
                    BsT[k_local + 1][n_local] = v.y;
                    BsT[k_local + 2][n_local] = v.z;
                    BsT[k_local + 3][n_local] = v.w;
                }
                __syncthreads();

                constexpr int TM4 = TM / 4;
                constexpr int TN4 = TN / 4;

                #pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float a_reg[TM];
                    float b_reg[TN];
                    #pragma unroll
                    for (int i = 0; i < TM4; ++i) {
                        float4 v = *reinterpret_cast<const float4*>(&AsT[kk][ty * TM + i * 4]);
                        a_reg[i * 4 + 0] = v.x;
                        a_reg[i * 4 + 1] = v.y;
                        a_reg[i * 4 + 2] = v.z;
                        a_reg[i * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int j = 0; j < TN4; ++j) {
                        float4 v = *reinterpret_cast<const float4*>(&BsT[kk][tx * TN + j * 4]);
                        b_reg[j * 4 + 0] = v.x;
                        b_reg[j * 4 + 1] = v.y;
                        b_reg[j * 4 + 2] = v.z;
                        b_reg[j * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] += a_reg[i] * b_reg[j];
                        }
                    }
                }
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }

        template <int BM, int BN, int BK, int TM, int TN>
        __global__ void matmul_at_b_tiled_kernel_v4(
                const float* __restrict__ A,
                const float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {
            constexpr int TX = BN / TN;
            constexpr int TY = BM / TM;
            constexpr int THREADS = TX * TY;
            constexpr int A_F4_LOADS = (BM * BK) / (THREADS * 4);
            constexpr int B_F4_LOADS = (BK * BN) / (THREADS * 4);
            static_assert((BM * BK) % (THREADS * 4) == 0, "BM*BK must be divisible by THREADS*4");
            static_assert((BK * BN) % (THREADS * 4) == 0, "BK*BN must be divisible by THREADS*4");
            static_assert(BM % 4 == 0, "BM must be divisible by 4");
            static_assert(BN % 4 == 0, "BN must be divisible by 4");
            static_assert(TM % 4 == 0, "TM must be divisible by 4");
            static_assert(TN % 4 == 0, "TN must be divisible by 4");

            __shared__ float As[BK][BM + 4];
            __shared__ float Bs[BK][BN + 4];

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

            const int num_k_blocks = K / BK;
            for (int kb = 0; kb < num_k_blocks; ++kb) {
                const int k_start = kb * BK;

                #pragma unroll
                for (int s = 0; s < A_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int k_local = elem_idx / BM;
                    const int m_local = elem_idx % BM;
                    const int gk = k_start + k_local;
                    const int gm = by * BM + m_local;
                    float4 v = *reinterpret_cast<const float4*>(&A[gk * M + gm]);
                    *reinterpret_cast<float4*>(&As[k_local][m_local]) = v;
                }
                #pragma unroll
                for (int s = 0; s < B_F4_LOADS; ++s) {
                    const int f4_idx = s * THREADS + tid;
                    const int elem_idx = f4_idx * 4;
                    const int k_local = elem_idx / BN;
                    const int n_local = elem_idx % BN;
                    const int gk = k_start + k_local;
                    const int gn = bx * BN + n_local;
                    float4 v = *reinterpret_cast<const float4*>(&B[gk * N + gn]);
                    *reinterpret_cast<float4*>(&Bs[k_local][n_local]) = v;
                }
                __syncthreads();

                constexpr int TM4 = TM / 4;
                constexpr int TN4 = TN / 4;

                #pragma unroll
                for (int kk = 0; kk < BK; ++kk) {
                    float a_reg[TM];
                    float b_reg[TN];
                    #pragma unroll
                    for (int i = 0; i < TM4; ++i) {
                        float4 v = *reinterpret_cast<const float4*>(&As[kk][ty * TM + i * 4]);
                        a_reg[i * 4 + 0] = v.x;
                        a_reg[i * 4 + 1] = v.y;
                        a_reg[i * 4 + 2] = v.z;
                        a_reg[i * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int j = 0; j < TN4; ++j) {
                        float4 v = *reinterpret_cast<const float4*>(&Bs[kk][tx * TN + j * 4]);
                        b_reg[j * 4 + 0] = v.x;
                        b_reg[j * 4 + 1] = v.y;
                        b_reg[j * 4 + 2] = v.z;
                        b_reg[j * 4 + 3] = v.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] += a_reg[i] * b_reg[j];
                        }
                    }
                }
                __syncthreads();
            }

            write_back<BM, BN, TM, TN>(C, M, N, by, bx, ty, tx, acc);
        }
    }

    void matmul_v4_launch(const float*, const float*, float*, int, int, int, cudaStream_t);
    void matmul_a_bt_v4_launch(const float*, const float*, float*, int, int, int, cudaStream_t);
    void matmul_at_b_v4_launch(const float*, const float*, float*, int, int, int, cudaStream_t);

    Tensor matmul_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        matmul_v4_launch(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor matmul_a_bt_cuda(const Tensor& a, const Tensor& b) {
        const int M = static_cast<int>(a.shape[0]);
        const int K = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[0]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        matmul_a_bt_v4_launch(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor matmul_at_b_cuda(const Tensor& a, const Tensor& b) {
        const int K = static_cast<int>(a.shape[0]);
        const int M = static_cast<int>(a.shape[1]);
        const int N = static_cast<int>(b.shape[1]);

        Tensor out = Tensor::empty({M, N}, a.dtype, Device::CUDA);

        matmul_at_b_v4_launch(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            M, N, K, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    void matmul_v1_launch(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
        constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
        dim3 block(BN / TN, BM / TM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        matmul_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }

    void matmul_v2_launch(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
        constexpr int SM_TARGET = 108;
        auto blocks = [&](int bm, int bn) {
            return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        };
        const int small = std::min(M, N);

        int tile;
        if (small <= 128) {
            tile = 32;
        } else if (small <= 384 && blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else if (blocks(128, 128) >= 2 * SM_TARGET) {
            tile = 128;
        } else if (blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else {
            tile = 32;
        }

        if (tile == 32) {
            constexpr int BM = 32, BN = 32, TM = 2, TN = 2;
            dim3 block(BN / TN, BM / TM);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            matmul_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        } else if (tile == 64) {
            constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            matmul_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        } else {
            constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
            dim3 block(BN / TN, BM / TM);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            matmul_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        }
    }

    void matmul_v3_launch(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
        constexpr int SM_TARGET = 108;
        auto blocks = [&](int bm, int bn) {
            return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        };
        const int small = std::min(M, N);

        int tile;
        if (small <= 128) {
            tile = 32;
        } else if (small <= 384 && blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else if (blocks(128, 128) >= 2 * SM_TARGET) {
            tile = 128;
        } else if (blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else {
            tile = 32;
        }

        const bool aligned4 = (K % 4 == 0) && (N % 4 == 0);
        const bool tile_exact = (M % tile == 0) && (N % tile == 0) && (K % 16 == 0);

        if (tile == 32 || !aligned4 || !tile_exact) {
            matmul_v2_launch(A, B, C, M, N, K, stream);
            return;
        }

        if (tile == 64) {
            constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            dim3 grid(N / BN, M / BM);
            matmul_tiled_kernel_v3<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        } else {
            constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
            dim3 block(BN / TN, BM / TM);
            dim3 grid(N / BN, M / BM);
            matmul_tiled_kernel_v3<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        }
    }

    void matmul_v4_launch(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream) {
        constexpr int SM_TARGET = 108;
        auto blocks = [&](int bm, int bn) {
            return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        };
        const int small = std::min(M, N);

        int tile;
        if (small <= 128) {
            tile = 32;
        } else if (small <= 384 && blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else if (blocks(128, 128) >= 2 * SM_TARGET) {
            tile = 128;
        } else if (blocks(64, 64) >= SM_TARGET) {
            tile = 64;
        } else {
            tile = 32;
        }

        const bool aligned4 = (K % 4 == 0) && (N % 4 == 0);
        const bool tile_exact = (M % tile == 0) && (N % tile == 0) && (K % 16 == 0);

        if (tile == 32 || !aligned4 || !tile_exact) {
            matmul_v2_launch(A, B, C, M, N, K, stream);
            return;
        }

        if (tile == 64) {
            constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
            dim3 block(BN / TN, BM / TM);
            dim3 grid(N / BN, M / BM);
            matmul_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        } else {
            constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
            dim3 block(BN / TN, BM / TM);
            dim3 grid(N / BN, M / BM);
            matmul_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        }
    }

    void matmul_a_bt_v4_launch(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream) {
        constexpr int SM_TARGET = 108;
        auto blocks_count = [&](int bm, int bn) {
            return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        };
        const int small = std::min(M, N);

        int tile;
        if (small <= 128) {
            tile = 32;
        } else if (small <= 384 && blocks_count(64, 64) >= SM_TARGET) {
            tile = 64;
        } else if (blocks_count(128, 128) >= 2 * SM_TARGET) {
            tile = 128;
        } else if (blocks_count(64, 64) >= SM_TARGET) {
            tile = 64;
        } else {
            tile = 32;
        }

        const bool aligned4 = (K % 4 == 0);
        const bool tile_exact = (M % tile == 0) && (N % tile == 0) && (K % 16 == 0);
        const bool fast_path = (tile != 32) && aligned4 && tile_exact;

        if (fast_path) {
            if (tile == 64) {
                constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
                dim3 block(BN / TN, BM / TM);
                dim3 grid(N / BN, M / BM);
                matmul_a_bt_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else {
                constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
                dim3 block(BN / TN, BM / TM);
                dim3 grid(N / BN, M / BM);
                matmul_a_bt_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            }
        } else {
            if (tile == 32) {
                constexpr int BM = 32, BN = 32, TM = 2, TN = 2;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_a_bt_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else if (tile == 64) {
                constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_a_bt_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else {
                constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_a_bt_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            }
        }
    }

    void matmul_at_b_v4_launch(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream) {
        constexpr int SM_TARGET = 108;
        auto blocks_count = [&](int bm, int bn) {
            return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        };
        const int small = std::min(M, N);

        int tile;
        if (small <= 128) {
            tile = 32;
        } else if (small <= 384 && blocks_count(64, 64) >= SM_TARGET) {
            tile = 64;
        } else if (blocks_count(128, 128) >= 2 * SM_TARGET) {
            tile = 128;
        } else if (blocks_count(64, 64) >= SM_TARGET) {
            tile = 64;
        } else {
            tile = 32;
        }

        const bool aligned4 = (M % 4 == 0) && (N % 4 == 0);
        const bool tile_exact = (M % tile == 0) && (N % tile == 0) && (K % 16 == 0);
        const bool fast_path = (tile != 32) && aligned4 && tile_exact;

        if (fast_path) {
            if (tile == 64) {
                constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
                dim3 block(BN / TN, BM / TM);
                dim3 grid(N / BN, M / BM);
                matmul_at_b_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else {
                constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
                dim3 block(BN / TN, BM / TM);
                dim3 grid(N / BN, M / BM);
                matmul_at_b_tiled_kernel_v4<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            }
        } else {
            if (tile == 32) {
                constexpr int BM = 32, BN = 32, TM = 2, TN = 2;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_at_b_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else if (tile == 64) {
                constexpr int BM = 64, BN = 64, TM = 4, TN = 4;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_at_b_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            } else {
                constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
                dim3 block(BN / TN, BM / TM);
                dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
                matmul_at_b_tiled_kernel<BM, BN, 16, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            }
        }
    }
}
