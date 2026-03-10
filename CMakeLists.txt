cmake_minimum_required(VERSION 3.16)
project(raw_ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # generates compile_commands.json for IDE

set(Eigen3_DIR /usr/share/eigen3/cmake)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(raw_ml
    main.cpp
    core/tensor.cpp
    backends/eigen_backend.cpp
    layers/dense.cpp
    layers/activation.cpp
    layers/conv2d.cpp
    layers/maxpool2d.cpp
    layers/sequential.cpp
    loss/loss.cpp
    data/mnist_loader.cpp
)

# expose each subdirectory as an include root so all #include "file.h" lines
target_include_directories(raw_ml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # project root — allows "core/tensor.h" style includes
    core
    backends
    layers
    loss
    data
)

target_link_libraries(raw_ml Eigen3::Eigen)
