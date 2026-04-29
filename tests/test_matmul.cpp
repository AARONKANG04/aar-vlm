#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "core/tensor.hpp"
#include "ops/matmul.hpp"

using namespace vlm;

namespace {
    void fill_fp32(Tensor& t, std::initializer_list<float> values) {
        auto* p = static_cast<float*>(t.data());
        size_t i = 0;
        for (float v : values) p[i++] = v;
    }
}

TEST(Matmul, BasicFp32) {
    Tensor a = Tensor::empty({2, 3}, DType::Fp32);
    fill_fp32(a, {1, 2, 3, 4, 5, 6});

    Tensor b = Tensor::empty({3, 2}, DType::Fp32);
    fill_fp32(b, {7, 8, 9, 10, 11, 12});

    Tensor c = matmul(a, b);

    ASSERT_EQ(c.shape, (std::vector<int64_t>{2, 2}));
    auto* p = static_cast<const float*>(c.data());
    EXPECT_FLOAT_EQ(p[0], 58.0f);
    EXPECT_FLOAT_EQ(p[1], 64.0f);
    EXPECT_FLOAT_EQ(p[2], 139.0f);
    EXPECT_FLOAT_EQ(p[3], 154.0f);
}

TEST(Matmul, IdentityIsIdentity) {
    Tensor a = Tensor::empty({3, 3}, DType::Fp32);
    fill_fp32(a, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    Tensor I = Tensor::zeros({3, 3}, DType::Fp32);
    auto* ip = static_cast<float*>(I.data());
    ip[0] = ip[4] = ip[8] = 1.0f;

    Tensor c = matmul(a, I);
    auto* ap = static_cast<const float*>(a.data());
    auto* cp = static_cast<const float*>(c.data());
    for (int i = 0; i < 9; ++i) EXPECT_FLOAT_EQ(cp[i], ap[i]);
}

TEST(Matmul, RejectsInnerDimMismatch) {
    Tensor a = Tensor::empty({2, 3}, DType::Fp32);
    Tensor b = Tensor::empty({4, 2}, DType::Fp32);
    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(Matmul, RejectsDtypeMismatch) {
    Tensor a = Tensor::empty({2, 2}, DType::Fp32);
    Tensor b = Tensor::empty({2, 2}, DType::Fp16);
    EXPECT_THROW(matmul(a, b), std::invalid_argument);
}

TEST(Matmul, CudaMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP() << "Built without CUDA";
#else
    Tensor a_cpu = Tensor::empty({2, 3}, DType::Fp32);
    fill_fp32(a_cpu, {1, 2, 3, 4, 5, 6});
    Tensor b_cpu = Tensor::empty({3, 2}, DType::Fp32);
    fill_fp32(b_cpu, {7, 8, 9, 10, 11, 12});

    Tensor a_gpu = a_cpu.to(Device::CUDA);
    Tensor b_gpu = b_cpu.to(Device::CUDA);
    Tensor c_gpu = matmul(a_gpu, b_gpu);
    Tensor c_cpu = c_gpu.to(Device::CPU);

    ASSERT_EQ(c_cpu.shape, (std::vector<int64_t>{2, 2}));
    auto* p = static_cast<const float*>(c_cpu.data());
    EXPECT_FLOAT_EQ(p[0], 58.0f);
    EXPECT_FLOAT_EQ(p[1], 64.0f);
    EXPECT_FLOAT_EQ(p[2], 139.0f);
    EXPECT_FLOAT_EQ(p[3], 154.0f);
#endif
}

TEST(Matmul, CudaLargeMatchesCpu) {
#ifndef HAS_CUDA
    GTEST_SKIP() << "Built without CUDA";
#else
    constexpr int M = 64, K = 96, N = 48;
    Tensor a = Tensor::empty({M, K}, DType::Fp32);
    Tensor b = Tensor::empty({K, N}, DType::Fp32);
    auto* ap = static_cast<float*>(a.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < M * K; ++i) ap[i] = (i * 0.013f) - 0.4f;
    for (int i = 0; i < K * N; ++i) bp[i] = (i * 0.007f) + 0.1f;

    Tensor c_cpu = matmul(a, b);
    Tensor c_gpu = matmul(a.to(Device::CUDA), b.to(Device::CUDA)).to(Device::CPU);

    auto* cp = static_cast<const float*>(c_cpu.data());
    auto* gp = static_cast<const float*>(c_gpu.data());
    for (int i = 0; i < M * N; ++i) EXPECT_NEAR(cp[i], gp[i], 1e-3f);
#endif
}
