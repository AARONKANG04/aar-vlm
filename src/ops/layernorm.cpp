#include "ops/layernorm.hpp"

#include <cmath>
#include <stdexcept>

namespace vlm {
    Tensor layernorm_cuda(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps);

    namespace {
        Tensor layernorm_cpu(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            const float* W = static_cast<const float*>(weight.data());
            const float* B = static_cast<const float*>(bias.data());
            float* Y = static_cast<float*>(out.data());
            const int64_t last = x.shape.back();
            const int64_t outer = x.numel() / last;
            const float D = static_cast<float>(last);

            for (int64_t i = 0; i < outer; ++i) {
                const float* row_in = X + i * last;
                float* row_out = Y + i * last;

                float sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) sum += row_in[j];
                const float mean = sum / D;

                float sq_sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) {
                    const float d = row_in[j] - mean;
                    sq_sum += d * d;
                }
                const float var = sq_sum / D;
                const float rstd = 1.0f / std::sqrt(var + eps);

                for (int64_t j = 0; j < last; ++j) {
                    row_out[j] = (row_in[j] - mean) * rstd * W[j] + B[j];
                }
            }
            return out;
        }
    }

    Tensor layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
        if (x.shape.empty()) {
            throw std::invalid_argument("layernorm: input must have at least 1 dim");
        }
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("layernorm: only Fp32 supported");
        }
        const int64_t last = x.shape.back();
        if (last == 0) {
            throw std::invalid_argument("layernorm: last dim must be > 0");
        }
        if (weight.shape.size() != 1 || weight.shape[0] != last) {
            throw std::invalid_argument("layernorm: weight must be 1D matching last dim");
        }
        if (bias.shape.size() != 1 || bias.shape[0] != last) {
            throw std::invalid_argument("layernorm: bias must be 1D matching last dim");
        }
        if (weight.dtype != DType::Fp32 || bias.dtype != DType::Fp32) {
            throw std::invalid_argument("layernorm: weight/bias must be Fp32");
        }
        if (x.device != weight.device || x.device != bias.device) {
            throw std::invalid_argument("layernorm: device mismatch");
        }
        if (eps <= 0.0f) {
            throw std::invalid_argument("layernorm: eps must be > 0");
        }
        switch (x.device) {
            case Device::CPU: return layernorm_cpu(x, weight, bias, eps);
            case Device::CUDA: return layernorm_cuda(x, weight, bias, eps);
        }
        throw std::runtime_error("layernorm: unsupported device");
    }
}
