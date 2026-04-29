#include <gtest/gtest.h>

#include <stdexcept>

#include "core/tensor.hpp"
#include "ops/elementwise.hpp"

using namespace vlm;

namespace {
    void fill_fp32(Tensor& t, std::initializer_list<float> values) {
        auto* p = static_cast<float*>(t.data());
        size_t i = 0;
        for (float v : values) p[i++] = v;
    }
}

TEST(Elementwise, AddBasic) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    fill_fp32(a, {1, 2, 3, 4});
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(b, {10, 20, 30, 40});
    Tensor c = add(a, b);
    auto* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 11.0f);
    EXPECT_FLOAT_EQ(p[1], 22.0f);
    EXPECT_FLOAT_EQ(p[2], 33.0f);
    EXPECT_FLOAT_EQ(p[3], 44.0f);
}

TEST(Elementwise, MulBasic) {
    Tensor a = Tensor::empty({3}, DType::Fp32);
    fill_fp32(a, {2, 3, 4});
    Tensor b = Tensor::empty({3}, DType::Fp32);
    fill_fp32(b, {5, 6, 7});
    Tensor c = mul(a, b);
    auto* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 10.0f);
    EXPECT_FLOAT_EQ(p[1], 18.0f);
    EXPECT_FLOAT_EQ(p[2], 28.0f);
}

TEST(Elementwise, ReluBasic) {
    Tensor a = Tensor::empty({5}, DType::Fp32);
    fill_fp32(a, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    Tensor c = relu(a);
    auto* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
    EXPECT_FLOAT_EQ(p[3], 0.5f);
    EXPECT_FLOAT_EQ(p[4], 2.0f);
}

TEST(Elementwise, AddRejectsShapeMismatch) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({3}, DType::Fp32);
    EXPECT_THROW(add(a, b), std::invalid_argument);
}

TEST(Elementwise, MulRejectsDtypeMismatch) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp16);
    EXPECT_THROW(mul(a, b), std::invalid_argument);
}

TEST(Elementwise, CudaAddMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP();
#else
    constexpr int N = 5000;
    Tensor a = Tensor::empty({N}, DType::Fp32);
    Tensor b = Tensor::empty({N}, DType::Fp32);
    auto* ap = static_cast<float*>(a.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < N; ++i) {
        ap[i] = i * 0.01f - 25.0f;
        bp[i] = i * 0.02f + 1.0f;
    }
    Tensor c_cpu = add(a, b);
    Tensor c_gpu = add(a.to(Device::CUDA), b.to(Device::CUDA)).to(Device::CPU);
    auto* cp = static_cast<const float*>(c_cpu.data());
    auto* gp = static_cast<const float*>(c_gpu.data());
    for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(cp[i], gp[i]);
#endif
}

TEST(Elementwise, CudaMulMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP();
#else
    constexpr int N = 5000;
    Tensor a = Tensor::empty({N}, DType::Fp32);
    Tensor b = Tensor::empty({N}, DType::Fp32);
    auto* ap = static_cast<float*>(a.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < N; ++i) {
        ap[i] = i * 0.01f - 25.0f;
        bp[i] = i * 0.02f + 1.0f;
    }
    Tensor c_cpu = mul(a, b);
    Tensor c_gpu = mul(a.to(Device::CUDA), b.to(Device::CUDA)).to(Device::CPU);
    auto* cp = static_cast<const float*>(c_cpu.data());
    auto* gp = static_cast<const float*>(c_gpu.data());
    for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(cp[i], gp[i]);
#endif
}

TEST(Elementwise, CudaReluMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP();
#else
    constexpr int N = 5000;
    Tensor a = Tensor::empty({N}, DType::Fp32);
    auto* ap = static_cast<float*>(a.data());
    for (int i = 0; i < N; ++i) ap[i] = (i * 0.01f) - 25.0f;
    Tensor c_cpu = relu(a);
    Tensor c_gpu = relu(a.to(Device::CUDA)).to(Device::CPU);
    auto* cp = static_cast<const float*>(c_cpu.data());
    auto* gp = static_cast<const float*>(c_gpu.data());
    for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(cp[i], gp[i]);
#endif
}
