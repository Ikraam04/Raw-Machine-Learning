#pragma once
#include "layer.h"
#include <random>

namespace nn {

/**
 * Dense (fully connected) layer.
 * 
 * Performs: output = input * weights + bias
 * 
 * Parameters:
 * - weights: (input_dim x output_dim) matrix
 * - bias: (output_dim) vector
 * 
 * During backprop:
 * - Computes gradient w.r.t. input
 * - Accumulates gradients for weights and bias
 */
class Dense : public Layer {
public:
    /**
     * Create a dense layer.
     * 
     * @param input_dim: Number of input features
     * @param output_dim: Number of output features
     * @param backend: Backend for computation
     */
    Dense(size_t input_dim, size_t output_dim, BackendPtr backend);
    
    /**
     * Forward pass: output = input * W + b
     * 
     * @param input: (batch_size x input_dim)
     * @return: (batch_size x output_dim)
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * Backward pass: compute gradients
     * 
     * Given grad_output (gradient of loss w.r.t. output):
     * - grad_input = grad_output * W^T
     * - grad_weights = input^T * grad_output
     * - grad_bias = sum(grad_output, axis=0)
     * 
     * @param grad_output: (batch_size x output_dim)
     * @return grad_input: (batch_size x input_dim)
     */
    Tensor backward(const Tensor& grad_output) override;
    
    /**
     * Update parameters using gradient descent.
     * 
     * W = W - learning_rate * grad_W
     * b = b - learning_rate * grad_b
     */
    void update_parameters(float learning_rate) override;
    
    /**
     * Get the weights and bias (for inspection/testing)
     */
    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }

    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }
    
    /**
     * Get the gradients (for inspection/testing)
     */
    const Tensor& grad_weights() const { return grad_weights_; }
    const Tensor& grad_bias() const { return grad_bias_; }
    
private:
    size_t input_dim_;
    size_t output_dim_;
    
    // Learnable parameters
    Tensor weights_;      // (input_dim x output_dim)
    Tensor bias_;         // (1 x output_dim)
    
    // Gradients of parameters
    Tensor grad_weights_; // Same shape as weights_
    Tensor grad_bias_;    // Same shape as bias_
    
    // Cached for backward pass
    Tensor input_cache_;  // Store input from forward pass
    
    /**
     * Initialize weights with Xavier/Glorot initialization.
     * Good default for sigmoid/tanh activations.
     */
    void initialize_weights();
};

} // namespace nn