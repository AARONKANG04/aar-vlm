#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/tensor.hpp"
#include "ops/layernorm.hpp"

using namespace vlm;

namespace {
    void fill_fp32(Tensor& t, std::initializer_list<float> values) {
        auto* p = static_cast<float*>(t.data());
        size_t i = 0;
        for (float v : values) p[i++] = v;
    }

    Tensor full(int64_t n, float v) {
        Tensor t = Tensor::empty({n}, DType::Fp32);
        auto* p = static_cast<float*>(t.data());
        for (int64_t i = 0; i < n; ++i) p[i] = v;
        return t;
    }
}

TEST(LayerNorm, IdentityAffineGivesZeroMeanUnitVar) {
    Tensor x = Tensor::empty({8}, DType::Fp32);
    fill_fp32(x, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor w = full(8, 1.0f);
    Tensor b = full(8, 0.0f);
    Tensor y = layernorm(x, w, b);
    auto* p = static_cast<const float*>(y.data());

    float mean = 0.0f;
    for (int i = 0; i < 8; ++i) mean += p[i];
    mean /= 8.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);

    float var = 0.0f;
    for (int i = 0; i < 8; ++i) var += (p[i] - mean) * (p[i] - mean);
    var /= 8.0f;
    EXPECT_NEAR(var, 1.0f, 1e-3f);
}

TEST(LayerNorm, AffineScalesAndShifts) {
    Tensor x = Tensor::empty({8}, DType::Fp32);
    fill_fp32(x, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor w = full(8, 2.0f);
    Tensor b = full(8, 3.0f);
    Tensor y = layernorm(x, w, b);
    auto* p = static_cast<const float*>(y.data());

    float mean = 0.0f;
    for (int i = 0; i < 8; ++i) mean += p[i];
    mean /= 8.0f;
    EXPECT_NEAR(mean, 3.0f, 1e-5f);

    float var = 0.0f;
    for (int i = 0; i < 8; ++i) var += (p[i] - mean) * (p[i] - mean);
    var /= 8.0f;
    EXPECT_NEAR(var, 4.0f, 1e-2f);
}

TEST(LayerNorm, ConstantInputProducesBias) {
    Tensor x = full(6, 7.0f);
    Tensor w = full(6, 5.0f);
    Tensor b = full(6, -1.0f);
    Tensor y = layernorm(x, w, b);
    auto* p = static_cast<const float*>(y.data());
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(p[i], -1.0f, 1e-3f);
}

TEST(LayerNorm, MultiRowEachIndependent) {
    Tensor x = Tensor::empty({3, 4}, DType::Fp32);
    fill_fp32(x, {
        1, 2, 3, 4,
        10, 20, 30, 40,
        -5, -3, -1, 1,
    });
    Tensor w = full(4, 1.0f);
    Tensor b = full(4, 0.0f);
    Tensor y = layernorm(x, w, b);
    auto* p = static_cast<const float*>(y.data());
    for (int row = 0; row < 3; ++row) {
        float mean = 0.0f;
        for (int j = 0; j < 4; ++j) mean += p[row * 4 + j];
        mean /= 4.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-5f);
        float var = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const float d = p[row * 4 + j] - mean;
            var += d * d;
        }
        var /= 4.0f;
        EXPECT_NEAR(var, 1.0f, 1e-3f);
    }
}

TEST(LayerNorm, RejectsWeightShapeMismatch) {
    Tensor x = Tensor::empty({4}, DType::Fp32);
    Tensor w = Tensor::empty({3}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    EXPECT_THROW(layernorm(x, w, b), std::invalid_argument);
}

TEST(LayerNorm, RejectsScalar) {
    Tensor x = Tensor::empty({}, DType::Fp32);
    Tensor w = Tensor::empty({1}, DType::Fp32);
    Tensor b = Tensor::empty({1}, DType::Fp32);
    EXPECT_THROW(layernorm(x, w, b), std::invalid_argument);
}

TEST(LayerNorm, CudaMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP();
#else
    constexpr int rows = 32;
    constexpr int cols = 257;
    Tensor x = Tensor::empty({rows, cols}, DType::Fp32);
    Tensor w = Tensor::empty({cols}, DType::Fp32);
    Tensor b = Tensor::empty({cols}, DType::Fp32);
    auto* xp = static_cast<float*>(x.data());
    auto* wp = static_cast<float*>(w.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < rows * cols; ++i) {
        xp[i] = std::sin(i * 0.13f) * 5.0f + (i % 7) * 0.5f;
    }
    for (int j = 0; j < cols; ++j) {
        wp[j] = 1.0f + 0.1f * std::cos(j * 0.2f);
        bp[j] = 0.05f * j;
    }

    Tensor y_cpu = layernorm(x, w, b);
    Tensor y_gpu = layernorm(x.to(Device::CUDA), w.to(Device::CUDA), b.to(Device::CUDA))
                       .to(Device::CPU);

    auto* cp = static_cast<const float*>(y_cpu.data());
    auto* gp = static_cast<const float*>(y_gpu.data());
    for (int i = 0; i < rows * cols; ++i) {
        const float tol = 1e-4f * std::max(std::abs(cp[i]), 1.0f) + 1e-5f;
        EXPECT_NEAR(cp[i], gp[i], tol);
    }
#endif
}
