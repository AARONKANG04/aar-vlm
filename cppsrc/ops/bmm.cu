#include "ops/bmm.hpp"

#include <cuda_runtime.h>
#include <vector>

#include "core/cuda_check.hpp"

namespace vlm {
    namespace {
        constexpr int TILE = 16;

        __global__ void bmm_kernel(const float* A, const float* B, float* C,
                                   const int64_t* a_off, const int64_t* b_off,
                                   int64_t M, int64_t K, int64_t N,
                                   int64_t a_rs, int64_t a_cs,
                                   int64_t b_rs, int64_t b_cs) {
            const int64_t bi = blockIdx.z;
            const int64_t i = blockIdx.y * TILE + threadIdx.y;
            const int64_t j = blockIdx.x * TILE + threadIdx.x;
            if (i >= M || j >= N) return;
            const float* Ab = A + a_off[bi];
            const float* Bb = B + b_off[bi];
            float* Cb = C + bi * M * N;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += Ab[i * a_rs + k * a_cs] * Bb[k * b_rs + j * b_cs];
            }
            Cb[i * N + j] = acc;
        }

        __global__ void bmm_a_bt_kernel(const float* A, const float* B, float* C,
                                        const int64_t* a_off, const int64_t* b_off,
                                        int64_t M, int64_t K, int64_t N,
                                        int64_t a_rs, int64_t a_cs,
                                        int64_t b_rs, int64_t b_cs) {
            const int64_t bi = blockIdx.z;
            const int64_t i = blockIdx.y * TILE + threadIdx.y;
            const int64_t j = blockIdx.x * TILE + threadIdx.x;
            if (i >= M || j >= N) return;
            const float* Ab = A + a_off[bi];
            const float* Bb = B + b_off[bi];
            float* Cb = C + bi * M * N;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += Ab[i * a_rs + k * a_cs] * Bb[j * b_rs + k * b_cs];
            }
            Cb[i * N + j] = acc;
        }

        __global__ void bmm_at_b_kernel(const float* A, const float* B, float* C,
                                        const int64_t* a_off, const int64_t* b_off,
                                        int64_t M, int64_t K, int64_t N,
                                        int64_t a_rs, int64_t a_cs,
                                        int64_t b_rs, int64_t b_cs) {
            const int64_t bi = blockIdx.z;
            const int64_t i = blockIdx.y * TILE + threadIdx.y;
            const int64_t j = blockIdx.x * TILE + threadIdx.x;
            if (i >= M || j >= N) return;
            const float* Ab = A + a_off[bi];
            const float* Bb = B + b_off[bi];
            float* Cb = C + bi * M * N;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += Ab[k * a_rs + i * a_cs] * Bb[k * b_rs + j * b_cs];
            }
            Cb[i * N + j] = acc;
        }

        int64_t leading_product(const Tensor& a) {
            int64_t p = 1;
            for (size_t i = 0; i + 2 < a.shape.size(); ++i) p *= a.shape[i];
            return p;
        }

        std::vector<int64_t> with_last_two(const Tensor& a, int64_t last_minus_1, int64_t last) {
            std::vector<int64_t> out(a.shape.begin(), a.shape.end() - 2);
            out.push_back(last_minus_1);
            out.push_back(last);
            return out;
        }

        std::vector<int64_t> batch_offsets(const Tensor& t, int64_t B) {
            std::vector<int64_t> offs(static_cast<size_t>(B), 0);
            const size_t nd = t.shape.size();
            if (nd <= 2 || B == 0) return offs;
            const size_t bdims = nd - 2;
            std::vector<int64_t> idx(bdims, 0);
            for (int64_t bi = 0; bi < B; ++bi) {
                int64_t off = 0;
                for (size_t d = 0; d < bdims; ++d) off += idx[d] * t.strides[d];
                offs[bi] = off;
                for (int d = static_cast<int>(bdims) - 1; d >= 0; --d) {
                    if (++idx[d] < t.shape[d]) break;
                    idx[d] = 0;
                }
            }
            return offs;
        }

        dim3 grid_for(int64_t M, int64_t N, int64_t B) {
            return dim3(static_cast<unsigned int>((N + TILE - 1) / TILE),
                        static_cast<unsigned int>((M + TILE - 1) / TILE),
                        static_cast<unsigned int>(B));
        }

        struct BatchOffsetsDevice {
            int64_t* a = nullptr;
            int64_t* b = nullptr;
            BatchOffsetsDevice(const std::vector<int64_t>& a_off,
                               const std::vector<int64_t>& b_off) {
                const size_t bytes = a_off.size() * sizeof(int64_t);
                CUDA_CHECK(cudaMalloc(&a, bytes));
                CUDA_CHECK(cudaMalloc(&b, bytes));
                CUDA_CHECK(cudaMemcpy(a, a_off.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(b, b_off.data(), bytes, cudaMemcpyHostToDevice));
            }
            ~BatchOffsetsDevice() {
                if (a) cudaFree(a);
                if (b) cudaFree(b);
            }
        };
    }

    Tensor bmm_cuda(const Tensor& a, const Tensor& b) {
        const size_t nd = a.shape.size();
        const int64_t M = a.shape[nd - 2];
        const int64_t K = a.shape[nd - 1];
        const int64_t N = b.shape[nd - 1];
        const int64_t B = leading_product(a);
        Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, Device::CUDA);
        BatchOffsetsDevice offs(batch_offsets(a, B), batch_offsets(b, B));
        dim3 block(TILE, TILE);
        dim3 grid = grid_for(M, N, B);
        bmm_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            offs.a, offs.b,
            M, K, N,
            a.strides[nd - 2], a.strides[nd - 1],
            b.strides[nd - 2], b.strides[nd - 1]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor bmm_a_bt_cuda(const Tensor& a, const Tensor& b) {
        const size_t nd = a.shape.size();
        const int64_t M = a.shape[nd - 2];
        const int64_t K = a.shape[nd - 1];
        const int64_t N = b.shape[nd - 2];
        const int64_t B = leading_product(a);
        Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, Device::CUDA);
        BatchOffsetsDevice offs(batch_offsets(a, B), batch_offsets(b, B));
        dim3 block(TILE, TILE);
        dim3 grid = grid_for(M, N, B);
        bmm_a_bt_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            offs.a, offs.b,
            M, K, N,
            a.strides[nd - 2], a.strides[nd - 1],
            b.strides[nd - 2], b.strides[nd - 1]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    Tensor bmm_at_b_cuda(const Tensor& a, const Tensor& b) {
        const size_t nd = a.shape.size();
        const int64_t K = a.shape[nd - 2];
        const int64_t M = a.shape[nd - 1];
        const int64_t N = b.shape[nd - 1];
        const int64_t B = leading_product(a);
        Tensor out = Tensor::empty(with_last_two(a, M, N), a.dtype, Device::CUDA);
        BatchOffsetsDevice offs(batch_offsets(a, B), batch_offsets(b, B));
        dim3 block(TILE, TILE);
        dim3 grid = grid_for(M, N, B);
        bmm_at_b_kernel<<<grid, block>>>(
            static_cast<const float*>(a.data()),
            static_cast<const float*>(b.data()),
            static_cast<float*>(out.data()),
            offs.a, offs.b,
            M, K, N,
            a.strides[nd - 2], a.strides[nd - 1],
            b.strides[nd - 2], b.strides[nd - 1]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
}
