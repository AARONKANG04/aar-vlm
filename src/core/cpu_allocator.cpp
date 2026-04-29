#include "core/cpu_allocator.hpp"

#include <cstdlib>
#include <new>

#if defined(_WIN32)
    #include <malloc.h>
#endif

namespace vlm {
    void* CPUAllocator::alloc(size_t nbytes, size_t alignment) {
        if (nbytes == 0) return nullptr;

        #if defined(_WIN32)
            void* ptr = _aligned_malloc(nbytes, alignment);
            if (!ptr) throw std::bad_alloc();
            return ptr;
        #else
            // posix_memalign requires alignment to be a power of 2 and a multiple of sizeof(void*)
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, nbytes) != 0) throw std::bad_alloc();
            return ptr;
        #endif
    }

    void CPUAllocator::free(void* ptr) noexcept {
        if (!ptr) return;

        #if defined(_WIN32)
            _aligned_free(ptr);
        #else
            std::free(ptr);
        #endif
    }

    Allocator* cpu_allocator() {
        static CPUAllocator instance;
        return &instance;
    }
}