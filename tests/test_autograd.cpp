#include <gtest/gtest.h>

#include <stdexcept>

#include "autograd/function.hpp"
#include "core/tensor.hpp"

using namespace vlm;

namespace {
    class IdentityFunction : public Function {
    public:
        std::vector<Tensor> backward(const Tensor& grad_output) override {
            return {grad_output};
        }
        const char* name() const override { return "Identity"; }
    };

    class ScaleFunction : public Function {
    public:
        float k;
        explicit ScaleFunction(float k) : k(k) {}
        std::vector<Tensor> backward(const Tensor& grad_output) override {
            Tensor g = Tensor::empty(grad_output.shape, grad_output.dtype, grad_output.device);
            const float* in = static_cast<const float*>(grad_output.data());
            float* out = static_cast<float*>(g.data());
            const size_t n = grad_output.numel();
            for (size_t i = 0; i < n; ++i) out[i] = in[i] * k;
            return {g};
        }
    };
}

TEST(Autograd, IsLeafByDefault) {
    Tensor x = Tensor::zeros({4}, DType::Fp32);
    EXPECT_TRUE(x.is_leaf());
    EXPECT_FALSE(x.requires_grad);
    EXPECT_EQ(x.grad(), nullptr);
}

TEST(Autograd, SetRequiresGradAllocatesSlot) {
    Tensor x = Tensor::zeros({4}, DType::Fp32);
    x.set_requires_grad(true);
    EXPECT_TRUE(x.requires_grad);
    EXPECT_TRUE(x.is_leaf());
    EXPECT_EQ(x.grad(), nullptr);
}

TEST(Autograd, SetRequiresGradOnNonLeafThrows) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);
    x.grad_fn = std::make_shared<IdentityFunction>();
    EXPECT_THROW(x.set_requires_grad(true), std::runtime_error);
}

TEST(Autograd, IdentityBackwardFillsLeafGrad) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);
    x.set_requires_grad(true);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({3}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    Tensor seed = Tensor::ones({3}, DType::Fp32);
    y.backward(seed);

    auto g = x.grad();
    ASSERT_NE(g, nullptr);
    auto* p = static_cast<const float*>(g->data());
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 1.0f);
    EXPECT_FLOAT_EQ(p[2], 1.0f);
}

TEST(Autograd, GradAccumulatesAcrossBackwardCalls) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);
    x.set_requires_grad(true);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({3}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    y.backward(Tensor::ones({3}, DType::Fp32));
    y.backward(Tensor::ones({3}, DType::Fp32));

    auto* p = static_cast<const float*>(x.grad()->data());
    EXPECT_FLOAT_EQ(p[0], 2.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 2.0f);
}

TEST(Autograd, ZeroGradResetsSlot) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);
    x.set_requires_grad(true);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({3}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    y.backward(Tensor::ones({3}, DType::Fp32));
    ASSERT_NE(x.grad(), nullptr);
    x.zero_grad();
    EXPECT_EQ(x.grad(), nullptr);
}

TEST(Autograd, ScalarBackwardImplicitOnes) {
    Tensor x = Tensor::zeros({1}, DType::Fp32);
    x.set_requires_grad(true);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({1}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    y.backward();

    auto* p = static_cast<const float*>(x.grad()->data());
    EXPECT_FLOAT_EQ(p[0], 1.0f);
}

TEST(Autograd, NonScalarBackwardWithoutSeedThrows) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);
    x.set_requires_grad(true);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({3}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    EXPECT_THROW(y.backward(), std::runtime_error);
}

TEST(Autograd, ChainedGraphPropagatesScaling) {
    Tensor x = Tensor::zeros({2}, DType::Fp32);
    x.set_requires_grad(true);

    auto inner = std::make_shared<ScaleFunction>(3.0f);
    inner->record_input(x);
    Tensor mid = Tensor::zeros({2}, DType::Fp32);
    mid.requires_grad = true;
    mid.grad_fn = inner;

    auto outer = std::make_shared<ScaleFunction>(5.0f);
    outer->record_input(mid);
    Tensor y = Tensor::zeros({2}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = outer;

    y.backward(Tensor::ones({2}, DType::Fp32));

    auto* p = static_cast<const float*>(x.grad()->data());
    EXPECT_FLOAT_EQ(p[0], 15.0f);
    EXPECT_FLOAT_EQ(p[1], 15.0f);
}

TEST(Autograd, NonRequiresGradLeafReceivesNothing) {
    Tensor x = Tensor::zeros({3}, DType::Fp32);

    auto fn = std::make_shared<IdentityFunction>();
    fn->record_input(x);

    Tensor y = Tensor::zeros({3}, DType::Fp32);
    y.requires_grad = true;
    y.grad_fn = fn;

    y.backward(Tensor::ones({3}, DType::Fp32));
    EXPECT_EQ(x.grad(), nullptr);
}
