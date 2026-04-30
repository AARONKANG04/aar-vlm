#include "ops/adamw.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vlm {
    void adamw_step_cuda(Tensor& param, const Tensor& grad,
                         Tensor& m, Tensor& v,
                         float lr, float beta1, float beta2,
                         float eps, float weight_decay,
                         float bc1, float bc2);

    namespace {
        void check_match(const Tensor& a, const Tensor& b, const char* what) {
            if (a.shape != b.shape) {
                throw std::invalid_argument(std::string("adamw_step: ") + what + " shape mismatch");
            }
            if (a.dtype != b.dtype) {
                throw std::invalid_argument(std::string("adamw_step: ") + what + " dtype mismatch");
            }
            if (a.device != b.device) {
                throw std::invalid_argument(std::string("adamw_step: ") + what + " device mismatch");
            }
        }

        void adamw_step_cpu(Tensor& param, const Tensor& grad,
                            Tensor& m, Tensor& v,
                            float lr, float beta1, float beta2,
                            float eps, float weight_decay,
                            float bc1, float bc2) {
            float* p = static_cast<float*>(param.data());
            const float* g = static_cast<const float*>(grad.data());
            float* mp = static_cast<float*>(m.data());
            float* vp = static_cast<float*>(v.data());
            const size_t n = param.numel();
            const float one_minus_b1 = 1.0f - beta1;
            const float one_minus_b2 = 1.0f - beta2;
            const float wd_factor = 1.0f - lr * weight_decay;
            for (size_t i = 0; i < n; ++i) {
                const float gi = g[i];
                const float mi = beta1 * mp[i] + one_minus_b1 * gi;
                const float vi = beta2 * vp[i] + one_minus_b2 * gi * gi;
                mp[i] = mi;
                vp[i] = vi;
                const float m_hat = mi / bc1;
                const float v_hat = vi / bc2;
                p[i] = p[i] * wd_factor - lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }

    void adamw_step(Tensor& param, const Tensor& grad,
                    Tensor& m, Tensor& v,
                    float lr, float beta1, float beta2,
                    float eps, float weight_decay,
                    float bias_correction1, float bias_correction2) {
        if (param.dtype != DType::Fp32) {
            throw std::invalid_argument("adamw_step: only Fp32 supported");
        }
        check_match(param, grad, "grad");
        check_match(param, m, "m");
        check_match(param, v, "v");
        if (!param.is_contiguous() || !grad.is_contiguous() ||
            !m.is_contiguous() || !v.is_contiguous()) {
            throw std::invalid_argument("adamw_step: all tensors must be contiguous");
        }
        if (param.device == Device::CUDA) {
            adamw_step_cuda(param, grad, m, v, lr, beta1, beta2,
                            eps, weight_decay, bias_correction1, bias_correction2);
            return;
        }
        adamw_step_cpu(param, grad, m, v, lr, beta1, beta2,
                       eps, weight_decay, bias_correction1, bias_correction2);
    }
}
