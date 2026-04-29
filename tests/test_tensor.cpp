#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include "core/allocator.hpp"
#include "core/storage.hpp"
#include "core/tensor.hpp"

using namespace vlm;

TEST(Tensor, NumelAndNBytes) {
    Tensor t({2, 3}, DType::Fp32);
    EXPECT_EQ(t.numel(), 6u);
    EXPECT_EQ(t.nbytes(), 24u);
}

TEST(Tensor, NBytesFp16) {
    Tensor t({4, 4}, DType::Fp16);
    EXPECT_EQ(t.numel(), 16u);
    EXPECT_EQ(t.nbytes(), 32u);
}

TEST(Tensor, ScalarTensorIsOneElement) {
    Tensor t({}, DType::Fp32);
    EXPECT_EQ(t.numel(), 1u);
    EXPECT_EQ(t.nbytes(), 4u);
}

TEST(Tensor, ZerosAllocatesAndZeros) {
    Tensor t = Tensor::zeros({4}, DType::Fp32);
    ASSERT_NE(t.data(), nullptr);
    EXPECT_EQ(t.nbytes(), 16u);
    auto* p = static_cast<float*>(t.data());
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(p[i], 0.0f);
}

TEST(Tensor, EmptyAllocatesButDoesNotZero) {
    Tensor t = Tensor::empty({4}, DType::Fp32);
    ASSERT_NE(t.data(), nullptr);
    auto* p = static_cast<float*>(t.data());
    p[0] = 3.14f;
    p[1] = 2.71f;
    EXPECT_FLOAT_EQ(p[0], 3.14f);
    EXPECT_FLOAT_EQ(p[1], 2.71f);
}

TEST(Tensor, CopySharesStorage) {
    Tensor a = Tensor::zeros({4}, DType::Fp32);
    Tensor b = a;
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.storage.use_count(), 2);

    static_cast<float*>(b.data())[0] = 42.0f;
    EXPECT_FLOAT_EQ(static_cast<float*>(a.data())[0], 42.0f);
}

TEST(Storage, IsMoveOnly) {
    static_assert(!std::is_copy_constructible_v<Storage>);
    static_assert(!std::is_copy_assignable_v<Storage>);
    static_assert(std::is_move_constructible_v<Storage>);
    static_assert(std::is_move_assignable_v<Storage>);
}

TEST(Storage, MoveLeavesSourceEmpty) {
    Storage s1(64, cpu_allocator());
    void* original = s1.data();
    Storage s2 = std::move(s1);
    EXPECT_EQ(s2.data(), original);
    EXPECT_EQ(s1.data(), nullptr);
    EXPECT_EQ(s1.nbytes(), 0u);
}
