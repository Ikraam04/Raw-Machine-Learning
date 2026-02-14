#pragma once
#include "backend_interface.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace nn {

/**
 * Tensor class - multi-dimensional array with pluggable backend.
 * Can use EigenBackend (CPU) or CudaBackend (GPU) transparently.
 */
class Tensor {
public:
    // Type alias for backend pointer
    using BackendPtr = std::shared_ptr<Backend>;
    
    // ============================================================
    // Constructors and Destructor
    // ============================================================
    
    /**
     * Create tensor with given shape and backend.
     * Example: Tensor({2, 3}, backend) creates a 2x3 matrix
     */
    Tensor(const std::vector<size_t>& shape, BackendPtr backend);
    
    // Copy constructor - creates a new tensor with copied data
    Tensor(const Tensor& other);
    
    // Move constructor - transfers ownership
    Tensor(Tensor&& other) noexcept;
    
    // Copy assignment
    Tensor& operator=(const Tensor& other);
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor - automatically frees memory via backend
    ~Tensor();
    
    // ============================================================
    // Properties
    // ============================================================
    
    /** Get the shape of the tensor */
    const std::vector<size_t>& shape() const { return shape_; }
    
    /** Total number of elements */
    size_t size() const { return size_; }
    
    /** Number of rows (for 2D tensors) */
    size_t rows() const { return shape_.size() > 0 ? shape_[0] : 0; }
    
    /** Number of columns (for 2D tensors) */
    size_t cols() const { return shape_.size() > 1 ? shape_[1] : 1; }
    
    /** Raw data pointer (use with caution - backend specific) */
    float* data() { return data_; }
    const float* data() const { return data_; }
    
    /** Get the backend */
    BackendPtr backend() const { return backend_; }
    
    // ============================================================
    // Data Manipulation
    // ============================================================
    
    /** Fill entire tensor with a single value */
    void fill(float value);
    
    /** Set data from raw array (must match size) */
    void set_data(const float* src, size_t count);
    
    /** Set data from vector (must match size) */
    void set_data(const std::vector<float>& src);
    
    // ============================================================
    // Matrix Operations (return new tensors)
    // ============================================================
    
    /** Matrix multiplication: this * other */
    Tensor matmul(const Tensor& other) const;
    
    /** Element-wise addition: this + other */
    Tensor add(const Tensor& other) const;
    
    /** Element-wise multiplication: this * other */
    Tensor multiply(const Tensor& other) const;
    
    /** Transpose (for 2D tensors) */
    Tensor transpose() const;
    
    // ============================================================
    // In-place Operations (modify this tensor)
    // ============================================================
    
    /** In-place addition: this += other */
    void add_(const Tensor& other);
    
    /** In-place scalar multiplication: this *= scalar */
    void multiply_(float scalar);
    
    // ============================================================
    // Activation Functions (return new tensors)
    // ============================================================
    
    /** Apply ReLU activation */
    Tensor relu() const;
    
    /** Apply ReLU derivative (assumes input is pre-activation) */
    Tensor relu_derivative() const;
    
    /** Apply sigmoid activation */
    Tensor sigmoid() const;
    
    /** Apply sigmoid derivative (assumes input is sigmoid output) */
    Tensor sigmoid_derivative() const;
    
    // ============================================================
    // Utility
    // ============================================================
    
    /** Print tensor to console */
    void print(const char* name = "Tensor") const;
    
private:
    std::vector<size_t> shape_;  // Dimensions of the tensor
    size_t size_;                 // Total number of elements
    float* data_;                 // Raw data pointer (managed by backend)
    BackendPtr backend_;          // Backend that manages memory and operations
    
    /** Calculate total size from shape */
    size_t compute_size() const;
    
    /** Validate that other tensor has same backend */
    void check_same_backend(const Tensor& other) const;
};

} // namespace nn