#include "ops/flash_attention.hpp"

#include <cuda_runtime.h>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int BR = 32;
        constexpr int BC = 32;
        constexpr int D_MAX = 128;

        __global__ void flash_attn_fwd_kernel(
                const float* __restrict__ Q,
                const float* __restrict__ K,
                const float* __restrict__ V,
                float* __restrict__ O,
                float* __restrict__ L,
                int T, int d, bool causal, float scale) {
            extern __shared__ float smem[];
            float* K_tile = smem;                // [BC, d]
            float* V_tile = smem + BC * d;       // [BC, d]

            const int tid = threadIdx.x;
            const int q_block = blockIdx.x;
            const int bh = blockIdx.y;

            const int row = q_block * BR + tid;
            const bool valid_row = row < T;

            const float* Q_bh = Q + bh * T * d;
            const float* K_bh = K + bh * T * d;
            const float* V_bh = V + bh * T * d;
            float* O_bh = O + bh * T * d;
            float* L_bh = L + bh * T;

            float q_row[D_MAX];
            float O_row[D_MAX];
            #pragma unroll
            for (int e = 0; e < D_MAX; ++e) {
                q_row[e] = 0.0f;
                O_row[e] = 0.0f;
            }
            if (valid_row) {
                for (int e = 0; e < d; ++e) q_row[e] = Q_bh[row * d + e];
            }

            float m_i = -INFINITY;
            float l_i = 0.0f;

            const int num_kv_blocks = (T + BC - 1) / BC;
            for (int j = 0; j < num_kv_blocks; ++j) {
                const int kv_start = j * BC;
                const int kv_end = min(kv_start + BC, T);
                const int kv_count = kv_end - kv_start;

                if (causal && kv_start > q_block * BR + BR - 1) break;

                for (int idx = tid; idx < BC * d; idx += BR) {
                    const int c = idx / d;
                    const int e = idx % d;
                    const int row_k = kv_start + c;
                    K_tile[c * d + e] = (row_k < T) ? K_bh[row_k * d + e] : 0.0f;
                    V_tile[c * d + e] = (row_k < T) ? V_bh[row_k * d + e] : 0.0f;
                }
                __syncthreads();

                if (!valid_row) { __syncthreads(); continue; }

                float S_row[BC];
                float m_ij = -INFINITY;
                for (int c = 0; c < BC; ++c) {
                    if (c >= kv_count) { S_row[c] = -INFINITY; continue; }
                    const int col = kv_start + c;
                    if (causal && col > row) { S_row[c] = -INFINITY; continue; }
                    float s = 0.0f;
                    for (int e = 0; e < d; ++e) s += q_row[e] * K_tile[c * d + e];
                    s *= scale;
                    S_row[c] = s;
                    if (s > m_ij) m_ij = s;
                }

                const float m_new = fmaxf(m_i, m_ij);
                const float alpha = expf(m_i - m_new);
                float l_ij = 0.0f;
                for (int c = 0; c < BC; ++c) {
                    const float p = (S_row[c] == -INFINITY) ? 0.0f : expf(S_row[c] - m_new);
                    S_row[c] = p;
                    l_ij += p;
                }
                const float l_new = alpha * l_i + l_ij;

                for (int e = 0; e < d; ++e) {
                    float pv = 0.0f;
                    for (int c = 0; c < BC; ++c) pv += S_row[c] * V_tile[c * d + e];
                    O_row[e] = alpha * O_row[e] + pv;
                }

                m_i = m_new;
                l_i = l_new;
                __syncthreads();
            }

            if (valid_row) {
                const float inv_l = 1.0f / l_i;
                for (int e = 0; e < d; ++e) {
                    O_bh[row * d + e] = O_row[e] * inv_l;
                }
                L_bh[row] = m_i + logf(l_i);
            }
        }

        __global__ void flash_attn_bwd_kernel(
                const float* __restrict__ Q,
                const float* __restrict__ K,
                const float* __restrict__ V,
                const float* __restrict__ O,
                const float* __restrict__ L,
                const float* __restrict__ dO,
                float* __restrict__ dQ,
                float* __restrict__ dK,
                float* __restrict__ dV,
                int T, int d, bool causal, float scale) {
            extern __shared__ float smem[];
            float* K_tile = smem;                // [BC, d]
            float* V_tile = smem + BC * d;       // [BC, d]

            const int tid = threadIdx.x;
            const int q_block = blockIdx.x;
            const int bh = blockIdx.y;

            const int row = q_block * BR + tid;
            const bool valid_row = row < T;

            const float* Q_bh = Q + bh * T * d;
            const float* K_bh = K + bh * T * d;
            const float* V_bh = V + bh * T * d;
            const float* O_bh = O + bh * T * d;
            const float* L_bh = L + bh * T;
            const float* dO_bh = dO + bh * T * d;
            float* dQ_bh = dQ + bh * T * d;
            float* dK_bh = dK + bh * T * d;
            float* dV_bh = dV + bh * T * d;

            float q_row[D_MAX];
            float dO_row[D_MAX];
            float dQ_row[D_MAX];
            #pragma unroll
            for (int e = 0; e < D_MAX; ++e) {
                q_row[e] = 0.0f; dO_row[e] = 0.0f; dQ_row[e] = 0.0f;
            }
            float L_row = 0.0f;
            float D_row = 0.0f;
            if (valid_row) {
                for (int e = 0; e < d; ++e) {
                    q_row[e] = Q_bh[row * d + e];
                    dO_row[e] = dO_bh[row * d + e];
                    D_row += dO_row[e] * O_bh[row * d + e];
                }
                L_row = L_bh[row];
            }

            const int num_kv_blocks = (T + BC - 1) / BC;
            for (int j = 0; j < num_kv_blocks; ++j) {
                const int kv_start = j * BC;
                const int kv_end = min(kv_start + BC, T);
                const int kv_count = kv_end - kv_start;

                if (causal && kv_start > q_block * BR + BR - 1) break;

                for (int idx = tid; idx < BC * d; idx += BR) {
                    const int c = idx / d;
                    const int e = idx % d;
                    const int row_k = kv_start + c;
                    K_tile[c * d + e] = (row_k < T) ? K_bh[row_k * d + e] : 0.0f;
                    V_tile[c * d + e] = (row_k < T) ? V_bh[row_k * d + e] : 0.0f;
                }
                __syncthreads();

                float P_row[BC];
                float dS_row[BC];
                if (valid_row) {
                    for (int c = 0; c < BC; ++c) {
                        if (c >= kv_count) { P_row[c] = 0.0f; continue; }
                        const int col = kv_start + c;
                        if (causal && col > row) { P_row[c] = 0.0f; continue; }
                        float s = 0.0f;
                        for (int e = 0; e < d; ++e) s += q_row[e] * K_tile[c * d + e];
                        s *= scale;
                        P_row[c] = expf(s - L_row);
                    }
                    for (int c = 0; c < BC; ++c) {
                        float dp = 0.0f;
                        for (int e = 0; e < d; ++e) dp += dO_row[e] * V_tile[c * d + e];
                        dS_row[c] = P_row[c] * (dp - D_row);
                    }
                    for (int e = 0; e < d; ++e) {
                        float acc = 0.0f;
                        for (int c = 0; c < BC; ++c) acc += dS_row[c] * K_tile[c * d + e];
                        dQ_row[e] += acc * scale;
                    }
                }
                __syncthreads();

                if (valid_row) {
                    for (int c = 0; c < kv_count; ++c) {
                        const int col = kv_start + c;
                        const float p = P_row[c];
                        const float ds = dS_row[c];
                        for (int e = 0; e < d; ++e) {
                            atomicAdd(&dV_bh[col * d + e], p * dO_row[e]);
                            atomicAdd(&dK_bh[col * d + e], ds * q_row[e] * scale);
                        }
                    }
                }
                __syncthreads();
            }

            if (valid_row) {
                for (int e = 0; e < d; ++e) {
                    dQ_bh[row * d + e] = dQ_row[e];
                }
            }
        }
    }

    void flash_attention_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v,
                                      Tensor& out, Tensor& lse, bool causal, float scale) {
        const int d = static_cast<int>(q.shape.back());
        const int T = static_cast<int>(q.shape[q.shape.size() - 2]);
        int64_t bh = 1;
        for (size_t i = 0; i + 2 < q.shape.size(); ++i) bh *= q.shape[i];
        if (d > D_MAX) {
            throw std::invalid_argument("flash_attention: d_head exceeds D_MAX (128)");
        }
        const int num_q_blocks = (T + BR - 1) / BR;
        dim3 grid(num_q_blocks, static_cast<unsigned int>(bh));
        dim3 block(BR);
        const size_t smem_bytes = 2 * BC * d * sizeof(float);
        flash_attn_fwd_kernel<<<grid, block, smem_bytes>>>(
            static_cast<const float*>(q.data()),
            static_cast<const float*>(k.data()),
            static_cast<const float*>(v.data()),
            static_cast<float*>(out.data()),
            static_cast<float*>(lse.data()),
            T, d, causal, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void flash_attention_backward_cuda(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& o, const Tensor& lse,
                                       const Tensor& grad_out,
                                       Tensor& dq, Tensor& dk, Tensor& dv,
                                       bool causal, float scale) {
        const int d = static_cast<int>(q.shape.back());
        const int T = static_cast<int>(q.shape[q.shape.size() - 2]);
        int64_t bh = 1;
        for (size_t i = 0; i + 2 < q.shape.size(); ++i) bh *= q.shape[i];
        if (d > D_MAX) {
            throw std::invalid_argument("flash_attention: d_head exceeds D_MAX (128)");
        }
        const int num_q_blocks = (T + BR - 1) / BR;
        dim3 grid(num_q_blocks, static_cast<unsigned int>(bh));
        dim3 block(BR);
        const size_t smem_bytes = 2 * BC * d * sizeof(float);
        flash_attn_bwd_kernel<<<grid, block, smem_bytes>>>(
            static_cast<const float*>(q.data()),
            static_cast<const float*>(k.data()),
            static_cast<const float*>(v.data()),
            static_cast<const float*>(o.data()),
            static_cast<const float*>(lse.data()),
            static_cast<const float*>(grad_out.data()),
            static_cast<float*>(dq.data()),
            static_cast<float*>(dk.data()),
            static_cast<float*>(dv.data()),
            T, d, causal, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
