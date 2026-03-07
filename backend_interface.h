#pragma once
#include <cstddef>

namespace nn {

// abstract backend interface - EigenBackend and CudaBackend both implement this
class Backend {
public:
    virtual ~Backend() = default;

    // memory management

    // alloc 'size' floats, returns ptr (cpu or gpu depending on backend)
    virtual float* allocate(size_t size) = 0;

    // free memory allocated by this backend
    virtual void deallocate(float* ptr) = 0;

    // copy 'size' floats from src to dst, both ptrs must be from this backend
    virtual void copy(float* dst, const float* src, size_t size) = 0;

    // fill array with a constant value
    virtual void fill(float* data, float value, size_t size) = 0;

    // matrix ops

    // result = A * B, A is (A_rows x A_cols), B is (B_rows x B_cols)
    // result pre-allocated as (A_rows x B_cols), requires A_cols == B_rows
    virtual void matmul(float* result,
                       const float* A, size_t A_rows, size_t A_cols,
                       const float* B, size_t B_rows, size_t B_cols) = 0;

    // result = A^T, A is (rows x cols), result is (cols x rows)
    virtual void transpose(float* result,
                          const float* A, size_t rows, size_t cols) = 0;

    // element-wise ops

    // result[i] = A[i] + B[i]
    virtual void add(float* result,
                    const float* A, const float* B, size_t size) = 0;

    // result[i] = A[i] * B[i]
    virtual void multiply(float* result,
                         const float* A, const float* B, size_t size) = 0;

    // result[i] = A[i] * scalar
    virtual void scale(float* result,
                      const float* A, float scalar, size_t size) = 0;

    // activations

    // result[i] = max(0, A[i])
    virtual void relu(float* result, const float* A, size_t size) = 0;

    // result[i] = A[i] > 0 ? 1 : 0
    virtual void relu_derivative(float* result, const float* A, size_t size) = 0;

    // result[i] = 1 / (1 + exp(-A[i]))
    virtual void sigmoid(float* result, const float* A, size_t size) = 0;

    // result[i] = sigmoid(A[i]) * (1 - sigmoid(A[i])), assumes A[i] is already sigmoid output
    virtual void sigmoid_derivative(float* result, const float* A, size_t size) = 0;
};

} // namespace nn