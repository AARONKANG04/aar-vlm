#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/tensor.hpp"
#include "ops/softmax.hpp"

using namespace vlm;

namespace {
    void fill_fp32(Tensor& t, std::initializer_list<float> values) {
        auto* p = static_cast<float*>(t.data());
        size_t i = 0;
        for (float v : values) p[i++] = v;
    }
}

TEST(Softmax, SumsToOne) {
    Tensor x = Tensor::empty({4}, DType::Fp32);
    fill_fp32(x, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = softmax(x);
    auto* p = static_cast<const float*>(y.data());
    float sum = p[0] + p[1] + p[2] + p[3];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    for (int i = 0; i < 4; ++i) EXPECT_GT(p[i], 0.0f);
}

TEST(Softmax, UniformInputIsUniformOutput) {
    Tensor x = Tensor::zeros({5}, DType::Fp32);
    Tensor y = softmax(x);
    auto* p = static_cast<const float*>(y.data());
    for (int i = 0; i < 5; ++i) EXPECT_FLOAT_EQ(p[i], 0.2f);
}

TEST(Softmax, NumericallyStableForLargeInputs) {
    Tensor x = Tensor::empty({3}, DType::Fp32);
    fill_fp32(x, {1000.0f, 1001.0f, 1002.0f});
    Tensor y = softmax(x);
    auto* p = static_cast<const float*>(y.data());
    EXPECT_TRUE(std::isfinite(p[0]));
    EXPECT_TRUE(std::isfinite(p[1]));
    EXPECT_TRUE(std::isfinite(p[2]));
    EXPECT_NEAR(p[0] + p[1] + p[2], 1.0f, 1e-6f);
    EXPECT_LT(p[0], p[1]);
    EXPECT_LT(p[1], p[2]);
}

TEST(Softmax, MultiRowEachSumsToOne) {
    Tensor x = Tensor::empty({3, 4}, DType::Fp32);
    fill_fp32(x, {
        1, 2, 3, 4,
        -1, -2, -3, -4,
        100, 100, 100, 100,
    });
    Tensor y = softmax(x);
    auto* p = static_cast<const float*>(y.data());
    for (int row = 0; row < 3; ++row) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) sum += p[row * 4 + j];
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
    for (int j = 0; j < 4; ++j) EXPECT_FLOAT_EQ(p[2 * 4 + j], 0.25f);
}

TEST(Softmax, RejectsScalar) {
    Tensor x = Tensor::empty({}, DType::Fp32);
    EXPECT_THROW(softmax(x), std::invalid_argument);
}

TEST(Softmax, CudaMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP();
#else
    constexpr int rows = 32;
    constexpr int cols = 257;
    Tensor x = Tensor::empty({rows, cols}, DType::Fp32);
    auto* xp = static_cast<float*>(x.data());
    for (int i = 0; i < rows * cols; ++i) {
        xp[i] = std::sin(i * 0.13f) * 5.0f + (i % 7) * 0.5f;
    }

    Tensor y_cpu = softmax(x);
    Tensor y_gpu = softmax(x.to(Device::CUDA)).to(Device::CPU);

    auto* cp = static_cast<const float*>(y_cpu.data());
    auto* gp = static_cast<const float*>(y_gpu.data());
    for (int i = 0; i < rows * cols; ++i) {
        const float tol = 1e-5f * std::max(std::abs(cp[i]), 1.0f) + 1e-6f;
        EXPECT_NEAR(cp[i], gp[i], tol);
    }
#endif
}
