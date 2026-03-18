cmake_minimum_required(VERSION 3.16)
project(raw_ml LANGUAGES CXX CUDA)

# BECAUSE MY COMPILER KEPT DEFAULTING TO DEBUG MODE AND I WANT OPTIMIZATIONS
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Set standards for both C++ and CUDA
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# march-native and O3 for release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")


# Set CUDA architecture to 89, (for 4090, may need to update for newer cards)
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Eigen3_DIR /usr/share/eigen3/cmake)
find_package(Eigen3 REQUIRED NO_MODULE)

# Main Project
add_executable(raw_ml
    main.cpp
    core/tensor.cpp
    backends/eigen_backend.cpp
    backends/cuda_backend.cu
    layers/dense.cpp
    layers/activation.cpp
    layers/conv2d.cpp
    layers/maxpool2d.cpp
    layers/sequential.cpp
    loss/loss.cpp
    data/mnist_loader.cpp

)

# unit tests
add_executable(testing
    testing/unit_tests.cpp
    core/tensor.cpp
    backends/eigen_backend.cpp
    backends/cuda_backend.cu
    layers/dense.cpp
    layers/activation.cpp
    layers/conv2d.cpp
    layers/maxpool2d.cpp
    layers/sequential.cpp
    loss/loss.cpp
    data/mnist_loader.cpp
)

# matmul benchmark: eigen vs cuda
add_executable(bench
    benchmarks/matmul_bench.cpp
    core/tensor.cpp
    backends/eigen_backend.cpp
    backends/cuda_backend.cu
)
target_include_directories(bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    core
    backends
)
target_link_libraries(bench PRIVATE Eigen3::Eigen)

target_include_directories(raw_ml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    core
    backends
    layers
    loss
    data
)
target_link_libraries(raw_ml PRIVATE Eigen3::Eigen)

target_include_directories(testing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    core
    backends
    layers
    loss
    data
)
target_link_libraries(testing PRIVATE Eigen3::Eigen)
