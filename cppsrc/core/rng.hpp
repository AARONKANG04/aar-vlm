#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define VLM_HOSTDEV __host__ __device__
#else
#define VLM_HOSTDEV
#endif

namespace vlm {
    // Philox-4x32-10 counter-based RNG. Deterministic given (seed, idx).
    // Returns a float uniformly distributed in [0, 1).
    VLM_HOSTDEV inline float philox_uniform(uint64_t seed, uint64_t idx) {
        uint32_t counter[4] = {
            static_cast<uint32_t>(idx & 0xFFFFFFFFu),
            static_cast<uint32_t>(idx >> 32),
            0u, 0u
        };
        uint32_t key[2] = {
            static_cast<uint32_t>(seed & 0xFFFFFFFFu),
            static_cast<uint32_t>(seed >> 32)
        };
        constexpr uint32_t M0 = 0xD2511F53u;
        constexpr uint32_t M1 = 0xCD9E8D57u;
        constexpr uint32_t W0 = 0x9E3779B9u;
        constexpr uint32_t W1 = 0xBB67AE85u;
        for (int r = 0; r < 10; ++r) {
            uint64_t mul0 = static_cast<uint64_t>(counter[0]) * M0;
            uint64_t mul1 = static_cast<uint64_t>(counter[2]) * M1;
            uint32_t hi0 = static_cast<uint32_t>(mul0 >> 32);
            uint32_t lo0 = static_cast<uint32_t>(mul0);
            uint32_t hi1 = static_cast<uint32_t>(mul1 >> 32);
            uint32_t lo1 = static_cast<uint32_t>(mul1);
            uint32_t n0 = hi1 ^ counter[1] ^ key[0];
            uint32_t n1 = lo1;
            uint32_t n2 = hi0 ^ counter[3] ^ key[1];
            uint32_t n3 = lo0;
            counter[0] = n0; counter[1] = n1; counter[2] = n2; counter[3] = n3;
            if (r < 9) {
                key[0] += W0;
                key[1] += W1;
            }
        }
        return static_cast<float>(counter[0]) * (1.0f / 4294967296.0f);
    }
}
