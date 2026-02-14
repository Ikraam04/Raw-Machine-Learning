#include "eigen_backend.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace nn {

// ============================================================
// Memory Management
// ============================================================

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

// ============================================================
// Matrix Operations
// ============================================================

void EigenBackend::matmul(float* result,
                          const float* A, size_t A_rows, size_t A_cols,
                          const float* B, size_t B_rows, size_t B_cols) {
    // Map raw pointers to Eigen matrices (no copy - just wraps the pointer)
    // RowMajor means data is stored row-by-row: [row0][row1][row2]...
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_A(A, A_rows, A_cols);
    
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_B(B, B_rows, B_cols);
    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_result(result, A_rows, B_cols);
    
    // Eigen handles the matrix multiplication efficiently
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

// ============================================================
// Element-wise Operations
// ============================================================

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

// ============================================================
// Activation Functions
// ============================================================

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
    // Assumes A contains sigmoid outputs already
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * (1.0f - A[i]);
    }
}

} // namespace nn