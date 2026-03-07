#include "eigen_backend.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace nn {

// memory

float* EigenBackend::allocate(size_t size) {
    return new float[size];
}

void EigenBackend::deallocate(float* ptr) {
    delete[] ptr;
}

void EigenBackend::copy(float* dst, const float* src, size_t size) {
    std::copy(src, src + size, dst);
}

void EigenBackend::fill(float* data, float value, size_t size) {
    std::fill(data, data + size, value);
}

// matrix ops

void EigenBackend::matmul(float* result,
                          const float* A, size_t A_rows, size_t A_cols,
                          const float* B, size_t B_rows, size_t B_cols) {
    // wrap raw pointers as Eigen matrices (no copy), RowMajor = row-by-row storage
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_A(A, A_rows, A_cols);
    
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_B(B, B_rows, B_cols);
    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_result(result, A_rows, B_cols);
    
    mat_result = mat_A * mat_B;
}

void EigenBackend::transpose(float* result,
                             const float* A, size_t rows, size_t cols) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_A(A, rows, cols);
    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_result(result, cols, rows);
    
    mat_result = mat_A.transpose();
}

// element-wise ops

void EigenBackend::add(float* result,
                       const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] + B[i];
    }
}

void EigenBackend::multiply(float* result,
                            const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * B[i];
    }
}

void EigenBackend::scale(float* result,
                         const float* A, float scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * scalar;
    }
}

// activations

void EigenBackend::relu(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::max(0.0f, A[i]);
    }
}

void EigenBackend::relu_derivative(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void EigenBackend::sigmoid(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-A[i]));
    }
}

void EigenBackend::sigmoid_derivative(float* result, const float* A, size_t size) {
    // A[i] is already sigmoid(x), not the raw input
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * (1.0f - A[i]);
    }
}

} // namespace nn