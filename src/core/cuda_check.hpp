#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) 
    do {
        cudaError_t _err = (call);
        if (_err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) + " at " + __FILE__ + ":" + std::to_string(__LINE__));
        }
    } while (0)