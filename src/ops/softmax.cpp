#include "ops/softmax.hpp"

#include <cmath>
#include <stdexcept>

namespace vlm {
    Tensor softmax_cuda(const Tensor& x);

    namespace {
        Tensor softmax_cpu(const Tensor& x) {
            Tensor out = Tensor::empty(x.shape, x.dtype, x.device);
            const float* X = static_cast<const float*>(x.data());
            float* Y = static_cast<float*>(out.data());
            const int64_t last = x.shape.back();
            const int64_t outer = x.numel() / last;
            for (int64_t i = 0; i < outer; ++i) {
                const float* row = X + i * last;
                float* out_row = Y + i * last;
                float max_val = row[0];
                for (int64_t j = 1; j < last; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) {
                    out_row[j] = std::exp(row[j] - max_val);
                    sum += out_row[j];
                }
                const float inv_sum = 1.0f / sum;
                for (int64_t j = 0; j < last; ++j) out_row[j] *= inv_sum;
            }
            return out;
        }
    }

    Tensor softmax(const Tensor& x) {
        if (x.shape.empty()) {
            throw std::invalid_argument("softmax: input must have at least 1 dim");
        }
        if (x.dtype != DType::Fp32) {
            throw std::invalid_argument("softmax: only Fp32 supported");
        }
        if (x.shape.back() == 0) {
            throw std::invalid_argument("softmax: last dim must be > 0");
        }
        switch (x.device) {
            case Device::CPU: return softmax_cpu(x);
            case Device::CUDA: return softmax_cuda(x);
        }
        throw std::runtime_error("softmax: unsupported device");
    }
}
