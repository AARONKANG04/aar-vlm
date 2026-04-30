#include "ops/layernorm.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "autograd/function.hpp"

namespace vlm {
    Tensor layernorm_cuda(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps);
    void layernorm_with_stats_cuda(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                                    Tensor& y, Tensor& mean, Tensor& rstd);
    void layernorm_backward_cuda(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                  const Tensor& mean, const Tensor& rstd,
                                  Tensor& dx, Tensor& dw, Tensor& db);

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

        void layernorm_with_stats_cpu(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps,
                                       Tensor& y, Tensor& mean, Tensor& rstd) {
            const float* X = static_cast<const float*>(x.data());
            const float* W = static_cast<const float*>(weight.data());
            const float* B = static_cast<const float*>(bias.data());
            float* Y = static_cast<float*>(y.data());
            float* M = static_cast<float*>(mean.data());
            float* R = static_cast<float*>(rstd.data());
            const int64_t last = x.shape.back();
            const int64_t outer = x.numel() / last;
            const float D = static_cast<float>(last);

            for (int64_t i = 0; i < outer; ++i) {
                const float* row_in = X + i * last;
                float* row_out = Y + i * last;

                float sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) sum += row_in[j];
                const float mean_i = sum / D;

                float sq_sum = 0.0f;
                for (int64_t j = 0; j < last; ++j) {
                    const float d = row_in[j] - mean_i;
                    sq_sum += d * d;
                }
                const float var = sq_sum / D;
                const float rstd_i = 1.0f / std::sqrt(var + eps);

                M[i] = mean_i;
                R[i] = rstd_i;

                for (int64_t j = 0; j < last; ++j) {
                    row_out[j] = (row_in[j] - mean_i) * rstd_i * W[j] + B[j];
                }
            }
        }

        void layernorm_backward_cpu(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                     const Tensor& mean, const Tensor& rstd,
                                     Tensor& dx, Tensor& dw, Tensor& db) {
            const float* GY = static_cast<const float*>(grad_y.data());
            const float* X = static_cast<const float*>(x.data());
            const float* W = static_cast<const float*>(w.data());
            const float* M = static_cast<const float*>(mean.data());
            const float* R = static_cast<const float*>(rstd.data());
            float* DX = static_cast<float*>(dx.data());
            float* DW = static_cast<float*>(dw.data());
            float* DB = static_cast<float*>(db.data());

            const int64_t last = x.shape.back();
            const int64_t outer = x.numel() / last;
            const float D = static_cast<float>(last);
            const float inv_D = 1.0f / D;

            std::memset(DW, 0, last * sizeof(float));
            std::memset(DB, 0, last * sizeof(float));

            for (int64_t i = 0; i < outer; ++i) {
                const float* gy_row = GY + i * last;
                const float* x_row = X + i * last;
                float* dx_row = DX + i * last;
                const float mean_i = M[i];
                const float rstd_i = R[i];

                float sum_g = 0.0f;
                float sum_g_x_normed = 0.0f;
                for (int64_t j = 0; j < last; ++j) {
                    const float normed = (x_row[j] - mean_i) * rstd_i;
                    const float g = gy_row[j] * W[j];
                    sum_g += g;
                    sum_g_x_normed += g * normed;
                    DW[j] += gy_row[j] * normed;
                    DB[j] += gy_row[j];
                }

                for (int64_t j = 0; j < last; ++j) {
                    const float normed = (x_row[j] - mean_i) * rstd_i;
                    const float g = gy_row[j] * W[j];
                    dx_row[j] = rstd_i * (g - sum_g * inv_D - normed * sum_g_x_normed * inv_D);
                }
            }
        }

        Tensor layernorm_no_grad(const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
            return x.device == Device::CPU ? layernorm_cpu(x, w, b, eps) : layernorm_cuda(x, w, b, eps);
        }

        void layernorm_with_stats_no_grad(const Tensor& x, const Tensor& w, const Tensor& b, float eps,
                                           Tensor& y, Tensor& mean, Tensor& rstd) {
            if (x.device == Device::CPU) {
                layernorm_with_stats_cpu(x, w, b, eps, y, mean, rstd);
            } else {
                layernorm_with_stats_cuda(x, w, b, eps, y, mean, rstd);
            }
        }

        void layernorm_backward_no_grad(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                         const Tensor& mean, const Tensor& rstd,
                                         Tensor& dx, Tensor& dw, Tensor& db) {
            if (x.device == Device::CPU) {
                layernorm_backward_cpu(grad_y, x, w, mean, rstd, dx, dw, db);
            } else {
                layernorm_backward_cuda(grad_y, x, w, mean, rstd, dx, dw, db);
            }
        }

        class LayerNormFunction : public Function {
        public:
            Tensor saved_x, saved_w, saved_mean, saved_rstd;
            std::vector<Tensor> backward(const Tensor& grad_y) override {
                Tensor dx = Tensor::empty(saved_x.shape, saved_x.dtype, saved_x.device);
                Tensor dw = Tensor::empty(saved_w.shape, saved_w.dtype, saved_w.device);
                Tensor db = Tensor::empty(saved_w.shape, saved_w.dtype, saved_w.device);
                layernorm_backward_no_grad(grad_y, saved_x, saved_w, saved_mean, saved_rstd,
                                            dx, dw, db);
                return {dx, dw, db};
            }
            const char* name() const override { return "LayerNormFunction"; }
        };
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

        if (!x.requires_grad && !weight.requires_grad && !bias.requires_grad) {
            return layernorm_no_grad(x, weight, bias, eps);
        }

        const int64_t outer = x.numel() / last;
        Tensor y = Tensor::empty(x.shape, x.dtype, x.device);
        Tensor mean = Tensor::empty({outer}, x.dtype, x.device);
        Tensor rstd = Tensor::empty({outer}, x.dtype, x.device);
        layernorm_with_stats_no_grad(x, weight, bias, eps, y, mean, rstd);

        auto fn = std::make_shared<LayerNormFunction>();
        fn->record_input(x);
        fn->record_input(weight);
        fn->record_input(bias);
        fn->saved_x = x;
        fn->saved_w = weight;
        fn->saved_mean = mean;
        fn->saved_rstd = rstd;

        y.requires_grad = true;
        y.grad_fn = fn;
        return y;
    }
}
