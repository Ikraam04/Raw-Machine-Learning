cmake_minimum_required(VERSION 3.16)
project(raw_ml)

set(CMAKE_CXX_STANDARD 17)

set(Eigen3_DIR /usr/share/eigen3/cmake)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(raw_ml main.cpp eigen_backend.cpp tensor.cpp dense.cpp activation.cpp sequential.cpp)
target_link_libraries(raw_ml Eigen3::Eigen)
