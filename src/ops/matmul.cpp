#include "ops/matmul.hpp"

#include <stdexcept>

namespace vlm {
    Tensor matmul_cuda(const Tensor& a, const Tensor& b);

    namespace {
        Tensor matmul_cpu(const Tensor& a, const Tensor& b) {
            const int64_t M = a.shape[0];
            const int64_t K = a.shape[1];
            const int64_t N = b.shape[1];

            Tensor out = Tensor::empty({M, N}, a.dtype, a.device);

            const float* A = static_cast<const float*>(a.data());
            const float* B = static_cast<const float*>(b.data());
            float* C = static_cast<float*>(out.data());

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        acc += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = acc;
                }
            }
            return out;
        }
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        if (a.shape.size() != 2 || b.shape.size() != 2) {
            throw std::invalid_argument("matmul: inputs must be 2D");
        }
        if (a.dtype != b.dtype) {
            throw std::invalid_argument("matmul: dtype mismatch");
        }
        if (a.device != b.device) {
            throw std::invalid_argument("matmul: device mismatch");
        }
        if (a.dtype != DType::Fp32) {
            throw std::invalid_argument("matmul: only Fp32 supported");
        }
        if (b.shape[0] != a.shape[1]) {
            throw std::invalid_argument("matmul: inner dimensions mismatch");
        }
        switch (a.device) {
            case Device::CPU: return matmul_cpu(a, b);
            case Device::CUDA: return matmul_cuda(a, b);
        }
        throw std::runtime_error("matmul: unsupported device");
    }
}
