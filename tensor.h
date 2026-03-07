#pragma once
#include "backend_interface.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace nn {

// n-dimensional array with a swappable backend (cpu or gpu)
class Tensor {
public:
    using BackendPtr = std::shared_ptr<Backend>;

    // Tensor({2, 3}, backend) makes a 2x3 matrix
    Tensor(const std::vector<size_t>& shape, BackendPtr backend);

    Tensor(const Tensor& other);           // deep copy
    Tensor(Tensor&& other) noexcept;       // move (steals data ptr)
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();  // frees memory via backend

    // properties
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t rows() const { return shape_.size() > 0 ? shape_[0] : 0; }  // first dim
    size_t cols() const { return shape_.size() > 1 ? shape_[1] : 1; }  // second dim
    float* data() { return data_; }
    const float* data() const { return data_; }
    BackendPtr backend() const { return backend_; }

    // data ops
    void fill(float value);
    void set_data(const float* src, size_t count);
    void set_data(const std::vector<float>& src);

    // matrix ops (return new tensors)
    Tensor matmul(const Tensor& other) const;   // this * other
    Tensor add(const Tensor& other) const;       // this + other (element-wise)
    Tensor multiply(const Tensor& other) const;  // this * other (element-wise)
    Tensor transpose() const;                    // 2D only

    // in-place ops
    void add_(const Tensor& other);     // this += other
    void multiply_(float scalar);       // this *= scalar

    // activations (return new tensors)
    Tensor relu() const;
    Tensor relu_derivative() const;  // assumes input is pre-activation
    Tensor sigmoid() const;
    Tensor sigmoid_derivative() const;  // assumes input is sigmoid output

    void print(const char* name = "Tensor") const;

private:
    std::vector<size_t> shape_;  // dimensions
    size_t size_;                // total elements
    float* data_;                // raw ptr, managed by backend
    BackendPtr backend_;

    size_t compute_size() const;
    void check_same_backend(const Tensor& other) const;
};

} // namespace nn