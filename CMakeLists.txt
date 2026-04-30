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
option(BUILD_PYTHON "Build Python bindings" OFF)

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
    cppsrc/core/tensor.cpp
    cppsrc/core/storage.cpp
    cppsrc/core/cpu_allocator.cpp
    cppsrc/autograd/function.cpp
    cppsrc/autograd/backward.cpp
    cppsrc/ops/matmul.cpp
    cppsrc/ops/elementwise.cpp
    cppsrc/ops/softmax.cpp
    cppsrc/ops/layernorm.cpp
    cppsrc/ops/attention.cpp
    cppsrc/ops/shape.cpp
    cppsrc/ops/bmm.cpp
    cppsrc/ops/embedding.cpp
    cppsrc/ops/cross_entropy.cpp
)

if(HAS_CUDA)
    list(APPEND VLM_CORE_SOURCES
        cppsrc/core/cuda_allocator.cpp
        cppsrc/core/cuda_copy.cu
        cppsrc/ops/matmul.cu
        cppsrc/ops/elementwise.cu
        cppsrc/ops/softmax.cu
        cppsrc/ops/layernorm.cu
        cppsrc/ops/attention.cu
        cppsrc/ops/shape.cu
        cppsrc/ops/bmm.cu
        cppsrc/ops/embedding.cu
        cppsrc/ops/cross_entropy.cu
    )
else()
    list(APPEND VLM_CORE_SOURCES
        cppsrc/core/cuda_allocator_stub.cpp
        cppsrc/core/cuda_copy_stub.cpp
        cppsrc/ops/matmul_cuda_stub.cpp
        cppsrc/ops/elementwise_cuda_stub.cpp
        cppsrc/ops/softmax_cuda_stub.cpp
        cppsrc/ops/layernorm_cuda_stub.cpp
        cppsrc/ops/attention_cuda_stub.cpp
        cppsrc/ops/shape_cuda_stub.cpp
        cppsrc/ops/bmm_cuda_stub.cpp
        cppsrc/ops/embedding_cuda_stub.cpp
        cppsrc/ops/cross_entropy_cuda_stub.cpp
    )
endif()

add_library(vlm_core STATIC ${VLM_CORE_SOURCES})
target_include_directories(vlm_core PUBLIC cppsrc)
target_compile_features(vlm_core PUBLIC cxx_std_20)
set_target_properties(vlm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(HAS_CUDA)
    target_link_libraries(vlm_core PUBLIC CUDA::cudart)
endif()

if(BUILD_TESTS)
    add_executable(test_tensor tests/cpp/test_tensor.cpp)
    target_link_libraries(test_tensor PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_tensor)

    add_executable(test_matmul tests/cpp/test_matmul.cpp)
    target_link_libraries(test_matmul PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_matmul)

    add_executable(test_elementwise tests/cpp/test_elementwise.cpp)
    target_link_libraries(test_elementwise PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_elementwise)

    add_executable(test_softmax tests/cpp/test_softmax.cpp)
    target_link_libraries(test_softmax PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_softmax)

    add_executable(test_layernorm tests/cpp/test_layernorm.cpp)
    target_link_libraries(test_layernorm PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_layernorm)

    add_executable(test_autograd tests/cpp/test_autograd.cpp)
    target_link_libraries(test_autograd PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_autograd)

    add_executable(test_autograd_ops tests/cpp/test_autograd_ops.cpp)
    target_link_libraries(test_autograd_ops PRIVATE vlm_core GTest::gtest_main)
    gtest_discover_tests(test_autograd_ops)
endif()

if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(_core MODULE cppsrc/python/binding.cpp)
    target_link_libraries(_core PRIVATE vlm_core)
    install(TARGETS _core LIBRARY DESTINATION aargrad)
endif()
