#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vlm {
    enum class DType : uint8_t {
        Fp32,
        Fp16,
        Bf16,
    };
    enum class Device : uint8_t {
        CPU,
        CUDA,
    };

    constexpr size_t dtype_size(DType dtype) {
        switch (dtype) {
            case DType::Fp32: return 4;
            case DType::Fp16: return 2;
            case DType::Bf16: return 2;
            default: throw std::runtime_error("Unsupported dtype");
        }
    }
}