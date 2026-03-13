cmake_minimum_required(VERSION 3.16)
project(raw_ml LANGUAGES CXX CUDA)

# Set standards for both C++ and CUDA
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


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

# first cuda test
add_executable(check_cuda check.cu)

target_include_directories(raw_ml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    core
    backends
    layers
    loss
    data
)

target_link_libraries(raw_ml PRIVATE Eigen3::Eigen)
