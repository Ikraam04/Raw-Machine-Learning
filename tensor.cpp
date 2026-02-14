#include "tensor.h"
#include <iostream>
#include <numeric>
#include <algorithm>

namespace nn {

// ============================================================
// Constructors and Destructor
// ============================================================

Tensor::Tensor(const std::vector<size_t>& shape, BackendPtr backend)
    : shape_(shape), backend_(backend) {
    size_ = compute_size();
    data_ = backend_->allocate(size_);
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), backend_(other.backend_) {
    data_ = backend_->allocate(size_);
    backend_->copy(data_, other.data_, size_);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      size_(other.size_),
      data_(other.data_),
      backend_(std::move(other.backend_)) {
    // Nullify the moved-from object
    other.data_ = nullptr;
    other.size_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Clean up old data
        if (data_ && backend_) {
            backend_->deallocate(data_);
        }
        
        // Copy from other
        shape_ = other.shape_;
        size_ = other.size_;
        backend_ = other.backend_;
        data_ = backend_->allocate(size_);
        backend_->copy(data_, other.data_, size_);
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Clean up old data
        if (data_ && backend_) {
            backend_->deallocate(data_);
        }
        
        // Move from other
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        data_ = other.data_;
        backend_ = std::move(other.backend_);
        
        // Nullify the moved-from object
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    if (data_ && backend_) {
        backend_->deallocate(data_);
    }
}

// ============================================================
// Data Manipulation
// ============================================================

void Tensor::fill(float value) {
    backend_->fill(data_, value, size_);
}

void Tensor::set_data(const float* src, size_t count) {
    if (count != size_) {
        throw std::runtime_error("Data size mismatch: expected " + 
                                std::to_string(size_) + " but got " + 
                                std::to_string(count));
    }
    backend_->copy(data_, src, count);
}

void Tensor::set_data(const std::vector<float>& src) {
    set_data(src.data(), src.size());
}

// ============================================================
// Matrix Operations
// ============================================================

Tensor Tensor::matmul(const Tensor& other) const {
    check_same_backend(other);
    
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    
    if (cols() != other.rows()) {
        throw std::runtime_error("matmul dimension mismatch: (" + 
                                std::to_string(rows()) + "x" + std::to_string(cols()) + 
                                ") * (" + std::to_string(other.rows()) + "x" + 
                                std::to_string(other.cols()) + ")");
    }
    
    Tensor result({rows(), other.cols()}, backend_);
    backend_->matmul(result.data_, data_, rows(), cols(),
                     other.data_, other.rows(), other.cols());
    return result;
}

Tensor Tensor::add(const Tensor& other) const {
    check_same_backend(other);
    
    if (size_ != other.size_) {
        throw std::runtime_error("add requires same size tensors");
    }
    
    Tensor result(shape_, backend_);
    backend_->add(result.data_, data_, other.data_, size_);
    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    check_same_backend(other);
    
    if (size_ != other.size_) {
        throw std::runtime_error("multiply requires same size tensors");
    }
    
    Tensor result(shape_, backend_);
    backend_->multiply(result.data_, data_, other.data_, size_);
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("transpose requires 2D tensor");
    }
    
    Tensor result({cols(), rows()}, backend_);
    backend_->transpose(result.data_, data_, rows(), cols());
    return result;
}

// ============================================================
// In-place Operations
// ============================================================

void Tensor::add_(const Tensor& other) {
    check_same_backend(other);
    
    if (size_ != other.size_) {
        throw std::runtime_error("add_ requires same size tensors");
    }
    
    backend_->add(data_, data_, other.data_, size_);
}

void Tensor::multiply_(float scalar) {
    backend_->scale(data_, data_, scalar, size_);
}

// ============================================================
// Activation Functions
// ============================================================

Tensor Tensor::relu() const {
    Tensor result(shape_, backend_);
    backend_->relu(result.data_, data_, size_);
    return result;
}

Tensor Tensor::relu_derivative() const {
    Tensor result(shape_, backend_);
    backend_->relu_derivative(result.data_, data_, size_);
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape_, backend_);
    backend_->sigmoid(result.data_, data_, size_);
    return result;
}

Tensor Tensor::sigmoid_derivative() const {
    Tensor result(shape_, backend_);
    backend_->sigmoid_derivative(result.data_, data_, size_);
    return result;
}

// ============================================================
// Utility
// ============================================================

void Tensor::print(const char* name) const {
    std::cout << name << " (shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]):\n";
    
    if (shape_.size() == 2) {
        // Print as matrix
        for (size_t i = 0; i < rows(); ++i) {
            std::cout << "  ";
            for (size_t j = 0; j < cols(); ++j) {
                std::cout << data_[i * cols() + j] << " ";
            }
            std::cout << "\n";
        }
    } else if (shape_.size() == 1) {
        // Print as vector
        std::cout << "  [";
        for (size_t i = 0; i < std::min(size_, size_t(10)); ++i) {
            std::cout << data_[i];
            if (i < std::min(size_, size_t(10)) - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << ", ...";
        std::cout << "]\n";
    } else {
        // Print first few elements
        std::cout << "  [";
        for (size_t i = 0; i < std::min(size_, size_t(10)); ++i) {
            std::cout << data_[i];
            if (i < std::min(size_, size_t(10)) - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << ", ...";
        std::cout << "]\n";
    }
    std::cout << "\n";
}

// ============================================================
// Private Helper Methods
// ============================================================

size_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 
                          size_t(1), std::multiplies<size_t>());
}

void Tensor::check_same_backend(const Tensor& other) const {
    if (backend_ != other.backend_) {
        throw std::runtime_error("Tensors must use the same backend");
    }
}

} // namespace nn