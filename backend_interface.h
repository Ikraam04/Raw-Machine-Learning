#pragma once
#include <cstddef>

namespace nn {

/**
 * Abstract interface for computation backends.
 * Both EigenBackend and CudaBackend will implement this interface.
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    // ============================================================
    // Memory Management
    // ============================================================
    
    /**
     * Allocate memory for 'size' floats
     * Returns pointer to allocated memory (CPU or GPU depending on backend)
     */
    virtual float* allocate(size_t size) = 0;
    
    /**
     * Free memory allocated by this backend
     */
    virtual void deallocate(float* ptr) = 0;
    
    /**
     * Copy 'size' floats from src to dst
     * Both pointers must be from this backend's memory space
     */
    virtual void copy(float* dst, const float* src, size_t size) = 0;
    
    /**
     * Fill array with a constant value
     */
    virtual void fill(float* data, float value, size_t size) = 0;
    
    // ============================================================
    // Matrix Operations
    // ============================================================
    
    /**
     * Matrix multiplication: result = A * B
     * A is (A_rows x A_cols), B is (B_rows x B_cols)
     * result must be pre-allocated as (A_rows x B_cols)
     * Requires: A_cols == B_rows
     */
    virtual void matmul(float* result, 
                       const float* A, size_t A_rows, size_t A_cols,
                       const float* B, size_t B_rows, size_t B_cols) = 0;
    
    /**
     * Transpose: result = A^T
     * A is (rows x cols), result is (cols x rows)
     */
    virtual void transpose(float* result, 
                          const float* A, size_t rows, size_t cols) = 0;
    
    // ============================================================
    // Element-wise Operations
    // ============================================================
    
    /**
     * Element-wise addition: result[i] = A[i] + B[i]
     */
    virtual void add(float* result, 
                    const float* A, const float* B, size_t size) = 0;
    
    /**
     * Element-wise multiplication: result[i] = A[i] * B[i]
     */
    virtual void multiply(float* result, 
                         const float* A, const float* B, size_t size) = 0;
    
    /**
     * Scalar multiplication: result[i] = A[i] * scalar
     */
    virtual void scale(float* result, 
                      const float* A, float scalar, size_t size) = 0;
    
    // ============================================================
    // Activation Functions
    // ============================================================
    
    /**
     * ReLU activation: result[i] = max(0, A[i])
     */
    virtual void relu(float* result, const float* A, size_t size) = 0;
    
    /**
     * ReLU derivative: result[i] = A[i] > 0 ? 1 : 0
     */
    virtual void relu_derivative(float* result, const float* A, size_t size) = 0;
    
    /**
     * Sigmoid activation: result[i] = 1 / (1 + exp(-A[i]))
     */
    virtual void sigmoid(float* result, const float* A, size_t size) = 0;
    
    /**
     * Sigmoid derivative: result[i] = sigmoid(A[i]) * (1 - sigmoid(A[i]))
     * Assumes A[i] is already the sigmoid output
     */
    virtual void sigmoid_derivative(float* result, const float* A, size_t size) = 0;
};

} // namespace nn