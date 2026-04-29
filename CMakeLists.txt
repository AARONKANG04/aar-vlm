cmake_minimum_required(VERSION 3.24)
project(aar-vlm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) 
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(HAS_CUDA ON)
    add_compile_definitions(HAS_CUDA)
    message(STATUS "CUDA enabled with architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(HAS_CUDA OFF)
    message(STATUS "CUDA not found, building for CPU only")
endif()

option(BUILD_TESTS "Build tests" ON)

include(FetchContent)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(json)

if(BUILD_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # Windows MSVC only
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)
endif()

set(VLM_CORE_SOURCES
    src/core/tensor.cpp
    src/core/storage.cpp
    src/core/cpu_allocator.cpp
    src/autograd/function.cpp
    src/autograd/backward.cpp
    src/ops/matmul.cpp
    src/ops/elementwise.cpp
    src/ops/softmax.cpp
    src/ops/layernorm.cpp
)

if(HAS_CUDA)
    list(APPEND VLM_CORE_SOURCES
        src/core/cuda_allocator.cpp
        src/core/cuda_copy.cu
        src/ops/matmul.cu
        src/ops/elementwise.cu
        src/ops/softmax.cu
        src/ops/layernorm.cu
    )
else()
    list(APPEND VLM_CORE_SOURCES
        src/core/cuda_allocator_stub.cpp
        src/core/cuda_copy_stub.cpp
        src/ops/matmul_cuda_stub.cpp
        src/ops/elementwise_cuda_stub.cpp
        src/ops/softmax_cuda_stub.cpp
        src/ops/layernorm_cuda_stub.cpp
    )
endif()

add_library(vlm_core STATIC ${VLM_CORE_SOURCES})
target_include_directories(vlm_core PUBLIC src)
target_compile_features(vlm_core PUBLIC cxx_std_20)

if(HAS_CUDA)
    target_link_libraries(vlm_core PUBLIC CUDA::cudart)
    set_target_properties(vlm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTS)
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_tensor)

    add_executable(test_matmul tests/test_matmul.cpp)
    target_link_libraries(test_matmul PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_matmul)

    add_executable(test_elementwise tests/test_elementwise.cpp)
    target_link_libraries(test_elementwise PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_elementwise)

    add_executable(test_softmax tests/test_softmax.cpp)
    target_link_libraries(test_softmax PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_softmax)

    add_executable(test_layernorm tests/test_layernorm.cpp)
    target_link_libraries(test_layernorm PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_layernorm)

    add_executable(test_autograd tests/test_autograd.cpp)
    target_link_libraries(test_autograd PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_autograd)
endif()
