#pragma once
#include "tensor.h"
#include <memory>

namespace nn {

/**
 * Abstract base class for all neural network layers.
 * 
 * A layer takes an input tensor, applies a transformation,
 * and produces an output tensor. During training, it also
 * computes gradients via backpropagation.
 */
class Layer {
public:
    using BackendPtr = std::shared_ptr<Backend>;
    
    Layer(BackendPtr backend) : backend_(backend) {}
    virtual ~Layer() = default;
    
    /**
     * Forward pass: compute output from input.
     * The layer should cache any values needed for backward pass.
     * 
     * @param input: Input tensor (batch_size x input_dim)
     * @return: Output tensor (batch_size x output_dim)
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * Backward pass: compute gradient with respect to input.
     * Also updates internal gradients for learnable parameters.
     * 
     * @param grad_output: Gradient flowing back from next layer
     * @return: Gradient with respect to this layer's input
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    /**
     * Update learnable parameters using computed gradients.
     * 
     * @param learning_rate: Step size for gradient descent
     */
    virtual void update_parameters(float learning_rate) = 0;
    
    /**
     * Get the backend used by this layer.
     */
    BackendPtr backend() const { return backend_; }
    
protected:
    BackendPtr backend_;
};

} // namespace nn