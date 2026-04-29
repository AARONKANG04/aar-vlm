#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "core/tensor.hpp"
#include "ops/elementwise.hpp"
#include "ops/layernorm.hpp"
#include "ops/matmul.hpp"
#include "ops/softmax.hpp"

using namespace vlm;

namespace {
    void fill_fp32(Tensor& t, std::initializer_list<float> values) {
        auto* p = static_cast<float*>(t.data());
        size_t i = 0;
        for (float v : values) p[i++] = v;
    }

    float scalar(const Tensor& t) {
        return *static_cast<const float*>(t.data());
    }

    void verify_finite_diff(Tensor& input, const float* autograd_grad,
                            std::function<float()> forward_loss,
                            float eps = 1e-3f, float tol = 1e-2f) {
        auto* p = static_cast<float*>(input.data());
        const size_t n = input.numel();
        for (size_t i = 0; i < n; ++i) {
            const float orig = p[i];
            p[i] = orig + eps;
            const float lp = forward_loss();
            p[i] = orig - eps;
            const float lm = forward_loss();
            p[i] = orig;
            const float numerical = (lp - lm) / (2.0f * eps);
            EXPECT_NEAR(numerical, autograd_grad[i], tol) << "element " << i;
        }
    }
}

TEST(AutogradOps, AddBackwardOnesGrad) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(a, {1, 2, 3, 4});
    fill_fp32(b, {5, 6, 7, 8});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = add(a, b);
        Tensor loss = sum_all(c);
        loss.backward();
    }

    auto* ag = static_cast<const float*>(a.grad()->data());
    auto* bg = static_cast<const float*>(b.grad()->data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(ag[i], 1.0f);
        EXPECT_FLOAT_EQ(bg[i], 1.0f);
    }
}

TEST(AutogradOps, AddBackwardFiniteDiff) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(a, {0.5f, -1.0f, 2.5f, 3.0f});
    fill_fp32(b, {1.5f, 4.0f, -2.0f, 0.25f});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = add(a, b);
        Tensor loss = sum_all(c);
        loss.backward();
    }

    std::vector<float> a_grad(4), b_grad(4);
    std::memcpy(a_grad.data(), a.grad()->data(), 16);
    std::memcpy(b_grad.data(), b.grad()->data(), 16);

    a.set_requires_grad(false);
    b.set_requires_grad(false);
    a.zero_grad();
    b.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(add(a, b))); };
    verify_finite_diff(a, a_grad.data(), loss_fn);
    verify_finite_diff(b, b_grad.data(), loss_fn);
}

TEST(AutogradOps, MulBackwardSwapInputs) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(a, {2, 3, 5, 7});
    fill_fp32(b, {-1.0f, 4.0f, 0.5f, 2.0f});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = mul(a, b);
        Tensor loss = sum_all(c);
        loss.backward();
    }

    auto* ag = static_cast<const float*>(a.grad()->data());
    auto* bg = static_cast<const float*>(b.grad()->data());
    EXPECT_FLOAT_EQ(ag[0], -1.0f);
    EXPECT_FLOAT_EQ(ag[1], 4.0f);
    EXPECT_FLOAT_EQ(ag[2], 0.5f);
    EXPECT_FLOAT_EQ(ag[3], 2.0f);
    EXPECT_FLOAT_EQ(bg[0], 2.0f);
    EXPECT_FLOAT_EQ(bg[1], 3.0f);
    EXPECT_FLOAT_EQ(bg[2], 5.0f);
    EXPECT_FLOAT_EQ(bg[3], 7.0f);
}

TEST(AutogradOps, MulBackwardFiniteDiff) {
    Tensor a = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(a, {0.5f, -1.2f, 2.5f, 3.0f});
    fill_fp32(b, {1.5f, 4.0f, -2.0f, 0.25f});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = mul(a, b);
        Tensor loss = sum_all(c);
        loss.backward();
    }

    std::vector<float> a_grad(4), b_grad(4);
    std::memcpy(a_grad.data(), a.grad()->data(), 16);
    std::memcpy(b_grad.data(), b.grad()->data(), 16);

    a.set_requires_grad(false);
    b.set_requires_grad(false);
    a.zero_grad();
    b.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(mul(a, b))); };
    verify_finite_diff(a, a_grad.data(), loss_fn);
    verify_finite_diff(b, b_grad.data(), loss_fn);
}

TEST(AutogradOps, MixedRequiresGrad) {
    Tensor a = Tensor::empty({3}, DType::Fp32);
    Tensor b = Tensor::empty({3}, DType::Fp32);
    fill_fp32(a, {2, 3, 4});
    fill_fp32(b, {5, 6, 7});
    a.set_requires_grad(true);

    Tensor c = mul(a, b);
    sum_all(c).backward();

    auto* ag = static_cast<const float*>(a.grad()->data());
    EXPECT_FLOAT_EQ(ag[0], 5.0f);
    EXPECT_FLOAT_EQ(ag[1], 6.0f);
    EXPECT_FLOAT_EQ(ag[2], 7.0f);
    EXPECT_EQ(b.grad(), nullptr);
}

TEST(AutogradOps, ChainedAddMul) {
    Tensor a = Tensor::empty({2}, DType::Fp32);
    Tensor b = Tensor::empty({2}, DType::Fp32);
    Tensor c = Tensor::empty({2}, DType::Fp32);
    fill_fp32(a, {1, 2});
    fill_fp32(b, {3, 5});
    fill_fp32(c, {7, 11});
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    c.set_requires_grad(true);

    {
        Tensor y = mul(add(a, b), c);
        sum_all(y).backward();
    }

    auto* ag = static_cast<const float*>(a.grad()->data());
    auto* bg = static_cast<const float*>(b.grad()->data());
    auto* cg = static_cast<const float*>(c.grad()->data());
    EXPECT_FLOAT_EQ(ag[0], 7.0f);
    EXPECT_FLOAT_EQ(ag[1], 11.0f);
    EXPECT_FLOAT_EQ(bg[0], 7.0f);
    EXPECT_FLOAT_EQ(bg[1], 11.0f);
    EXPECT_FLOAT_EQ(cg[0], 1.0f + 3.0f);
    EXPECT_FLOAT_EQ(cg[1], 2.0f + 5.0f);
}

TEST(AutogradOps, ReluBackwardKillsNegativeGrads) {
    Tensor a = Tensor::empty({5}, DType::Fp32);
    fill_fp32(a, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    a.set_requires_grad(true);

    {
        Tensor c = relu(a);
        sum_all(c).backward();
    }

    auto* ag = static_cast<const float*>(a.grad()->data());
    EXPECT_FLOAT_EQ(ag[0], 0.0f);
    EXPECT_FLOAT_EQ(ag[1], 0.0f);
    EXPECT_FLOAT_EQ(ag[2], 0.0f);
    EXPECT_FLOAT_EQ(ag[3], 1.0f);
    EXPECT_FLOAT_EQ(ag[4], 1.0f);
}

TEST(AutogradOps, ReluBackwardFiniteDiff) {
    Tensor a = Tensor::empty({6}, DType::Fp32);
    fill_fp32(a, {-1.5f, -0.3f, 0.7f, 1.2f, -2.0f, 0.4f});
    a.set_requires_grad(true);

    {
        Tensor b = Tensor::empty({6}, DType::Fp32);
        fill_fp32(b, {0.5f, -0.2f, 1.0f, 0.3f, 0.1f, 0.4f});
        Tensor y = relu(mul(a, mul(a, a)));
        sum_all(y).backward();
    }

    std::vector<float> a_grad(6);
    std::memcpy(a_grad.data(), a.grad()->data(), 24);

    a.set_requires_grad(false);
    a.zero_grad();

    auto loss_fn = [&]() {
        return scalar(sum_all(relu(mul(a, mul(a, a)))));
    };
    verify_finite_diff(a, a_grad.data(), loss_fn, 1e-3f, 5e-2f);
}

TEST(AutogradOps, MatmulBackwardKnownCase) {
    Tensor a = Tensor::empty({2, 3}, DType::Fp32);
    fill_fp32(a, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::empty({3, 2}, DType::Fp32);
    fill_fp32(b, {7, 8, 9, 10, 11, 12});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = matmul(a, b);
        sum_all(c).backward();
    }

    auto* ag = static_cast<const float*>(a.grad()->data());
    auto* bg = static_cast<const float*>(b.grad()->data());

    EXPECT_FLOAT_EQ(ag[0], 7.0f + 8.0f);
    EXPECT_FLOAT_EQ(ag[1], 9.0f + 10.0f);
    EXPECT_FLOAT_EQ(ag[2], 11.0f + 12.0f);
    EXPECT_FLOAT_EQ(ag[3], 7.0f + 8.0f);
    EXPECT_FLOAT_EQ(ag[4], 9.0f + 10.0f);
    EXPECT_FLOAT_EQ(ag[5], 11.0f + 12.0f);

    EXPECT_FLOAT_EQ(bg[0], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(bg[1], 1.0f + 4.0f);
    EXPECT_FLOAT_EQ(bg[2], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(bg[3], 2.0f + 5.0f);
    EXPECT_FLOAT_EQ(bg[4], 3.0f + 6.0f);
    EXPECT_FLOAT_EQ(bg[5], 3.0f + 6.0f);
}

TEST(AutogradOps, MatmulBackwardFiniteDiff) {
    Tensor a = Tensor::empty({3, 4}, DType::Fp32);
    Tensor b = Tensor::empty({4, 2}, DType::Fp32);
    auto* ap = static_cast<float*>(a.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < 12; ++i) ap[i] = (i * 0.13f) - 0.5f;
    for (int i = 0; i < 8; ++i)  bp[i] = (i * 0.07f) + 0.2f;
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = matmul(a, b);
        sum_all(c).backward();
    }

    std::vector<float> a_grad(12), b_grad(8);
    std::memcpy(a_grad.data(), a.grad()->data(), 48);
    std::memcpy(b_grad.data(), b.grad()->data(), 32);

    a.set_requires_grad(false);
    b.set_requires_grad(false);
    a.zero_grad();
    b.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(matmul(a, b))); };
    verify_finite_diff(a, a_grad.data(), loss_fn, 1e-3f, 5e-2f);
    verify_finite_diff(b, b_grad.data(), loss_fn, 1e-3f, 5e-2f);
}

TEST(AutogradOps, MatmulReluChain) {
    Tensor a = Tensor::empty({2, 3}, DType::Fp32);
    Tensor b = Tensor::empty({3, 2}, DType::Fp32);
    fill_fp32(a, {0.5f, -0.3f, 0.4f, 0.1f, 0.2f, -0.1f});
    fill_fp32(b, {1.0f, -2.0f, 0.5f, 1.0f, -1.0f, 0.5f});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor c = relu(matmul(a, b));
        sum_all(c).backward();
    }

    std::vector<float> a_grad(6), b_grad(6);
    std::memcpy(a_grad.data(), a.grad()->data(), 24);
    std::memcpy(b_grad.data(), b.grad()->data(), 24);

    a.set_requires_grad(false);
    b.set_requires_grad(false);
    a.zero_grad();
    b.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(relu(matmul(a, b)))); };
    verify_finite_diff(a, a_grad.data(), loss_fn, 1e-3f, 5e-2f);
    verify_finite_diff(b, b_grad.data(), loss_fn, 1e-3f, 5e-2f);
}

TEST(AutogradOps, SoftmaxBackwardZerosOutWhenSummed) {
    Tensor x = Tensor::empty({1, 5}, DType::Fp32);
    fill_fp32(x, {0.1f, -0.5f, 1.2f, -0.3f, 0.4f});
    x.set_requires_grad(true);

    {
        Tensor y = softmax(x);
        sum_all(y).backward();
    }

    auto* xg = static_cast<const float*>(x.grad()->data());
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(xg[i], 0.0f, 1e-5f);
    }
}

TEST(AutogradOps, SoftmaxBackwardFiniteDiff) {
    Tensor x = Tensor::empty({3, 4}, DType::Fp32);
    auto* xp = static_cast<float*>(x.data());
    for (int i = 0; i < 12; ++i) xp[i] = std::sin(i * 0.31f) * 0.7f;
    x.set_requires_grad(true);

    Tensor weight = Tensor::empty({3, 4}, DType::Fp32);
    auto* wp = static_cast<float*>(weight.data());
    for (int i = 0; i < 12; ++i) wp[i] = (i * 0.13f) - 0.5f;

    {
        Tensor y = mul(softmax(x), weight);
        sum_all(y).backward();
    }

    std::vector<float> x_grad(12);
    std::memcpy(x_grad.data(), x.grad()->data(), 48);

    x.set_requires_grad(false);
    x.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(mul(softmax(x), weight))); };
    verify_finite_diff(x, x_grad.data(), loss_fn, 1e-3f, 5e-2f);
}

TEST(AutogradOps, LayerNormBackwardFiniteDiff) {
    Tensor x = Tensor::empty({4, 6}, DType::Fp32);
    Tensor w = Tensor::empty({6}, DType::Fp32);
    Tensor b = Tensor::empty({6}, DType::Fp32);
    auto* xp = static_cast<float*>(x.data());
    auto* wp = static_cast<float*>(w.data());
    auto* bp = static_cast<float*>(b.data());
    for (int i = 0; i < 24; ++i) xp[i] = std::sin(i * 0.21f) * 1.5f + 0.1f;
    for (int j = 0; j < 6; ++j) wp[j] = 1.0f + 0.2f * std::cos(j * 0.4f);
    for (int j = 0; j < 6; ++j) bp[j] = 0.05f * j;

    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    {
        Tensor y = layernorm(x, w, b);
        sum_all(y).backward();
    }

    std::vector<float> x_grad(24), w_grad(6), b_grad(6);
    std::memcpy(x_grad.data(), x.grad()->data(), 96);
    std::memcpy(w_grad.data(), w.grad()->data(), 24);
    std::memcpy(b_grad.data(), b.grad()->data(), 24);

    x.set_requires_grad(false);
    w.set_requires_grad(false);
    b.set_requires_grad(false);
    x.zero_grad();
    w.zero_grad();
    b.zero_grad();

    auto loss_fn = [&]() { return scalar(sum_all(layernorm(x, w, b))); };
    verify_finite_diff(x, x_grad.data(), loss_fn, 1e-3f, 5e-2f);
    verify_finite_diff(w, w_grad.data(), loss_fn, 1e-3f, 5e-2f);
    verify_finite_diff(b, b_grad.data(), loss_fn, 1e-3f, 5e-2f);
}

TEST(AutogradOps, LayerNormBiasGradIsRowSum) {
    Tensor x = Tensor::empty({3, 4}, DType::Fp32);
    Tensor w = Tensor::empty({4}, DType::Fp32);
    Tensor b = Tensor::empty({4}, DType::Fp32);
    fill_fp32(x, {1, 2, 3, 4, -1, -2, -3, -4, 0.5f, 1.5f, 2.5f, 3.5f});
    fill_fp32(w, {1, 1, 1, 1});
    fill_fp32(b, {0, 0, 0, 0});

    b.set_requires_grad(true);

    {
        Tensor y = layernorm(x, w, b);
        sum_all(y).backward();
    }

    auto* bg = static_cast<const float*>(b.grad()->data());
    for (int j = 0; j < 4; ++j) EXPECT_FLOAT_EQ(bg[j], 3.0f);
}

TEST(AutogradOps, AccumulatesAcrossSecondBackward) {
    Tensor a = Tensor::empty({3}, DType::Fp32);
    fill_fp32(a, {1, 2, 3});
    a.set_requires_grad(true);

    Tensor b = Tensor::empty({3}, DType::Fp32);
    fill_fp32(b, {0.1f, 0.2f, 0.3f});

    sum_all(add(a, b)).backward();
    sum_all(add(a, b)).backward();

    auto* ag = static_cast<const float*>(a.grad()->data());
    EXPECT_FLOAT_EQ(ag[0], 2.0f);
    EXPECT_FLOAT_EQ(ag[1], 2.0f);
    EXPECT_FLOAT_EQ(ag[2], 2.0f);
}
