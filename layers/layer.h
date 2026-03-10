#pragma once
#include "core/tensor.h"
#include <memory>

namespace nn {

// base class for all layers - takes input tensor, applies transform, returns output
// during training it also computes gradients for backprop
class Layer {
public:
    using BackendPtr = std::shared_ptr<Backend>;

    Layer(BackendPtr backend) : backend_(backend) {}
    virtual ~Layer() = default;

    // forward pass: input (batch_size x input_dim) -> output (batch_size x output_dim)
    // should cache anything needed for backward
    virtual Tensor forward(const Tensor& input) = 0;

    // backward pass: takes grad from next layer, returns grad w.r.t. this layer's input
    // also accumulates grads for any learnable params
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // apply gradient descent to learnable params
    virtual void update_parameters(float learning_rate) = 0;

    BackendPtr backend() const { return backend_; }

protected:
    BackendPtr backend_;
};

} // namespace nn