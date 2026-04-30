#pragma once

#include <cstdint>

#include "core/tensor.hpp"

namespace vlm {
    Tensor dropout(const Tensor& x, float p);

    // For tests: reset the global RNG counter so the next dropout call(s)
    // produce a reproducible mask sequence.
    void manual_seed(uint64_t seed);
}
