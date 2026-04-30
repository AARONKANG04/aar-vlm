#pragma once

#include <cstddef>

#include "core/types.hpp"

namespace vlm {
    void copy_bytes(void* dst, Device dst_device,
                    const void* src, Device src_device,
                    size_t nbytes);

    void zero_bytes(void* p, Device device, size_t nbytes);
}
