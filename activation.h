#pragma once
#include "layer.h"

namespace nn {

/**
 * ReLU Activation Layer
 * 
 * Forward: f(x) = max(0, x)
 * Backward: f'(x) = 1 if x > 0, else 0
 * 
 * No learnable parameters, so update_parameters does nothing.
 */
class ReLU : public Layer {
public:
    ReLU(BackendPtr backend) : Layer(backend), input_cache_({1,1}, backend)  {}
    
    /**
     * Forward pass: apply ReLU element-wise
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * Backward pass: multiply gradient by ReLU derivative
     */
    Tensor backward(const Tensor& grad_output) override;
    
    /**
     * No parameters to update
     */
    void update_parameters(float learning_rate) override {
        // ReLU has no learnable parameters
        (void)learning_rate;  // Suppress unused parameter warning
    }
    
private:
    Tensor input_cache_;  // Cache input for backward pass
};

/**
 * Sigmoid Activation Layer
 * 
 * Forward: f(x) = 1 / (1 + exp(-x))
 * Backward: f'(x) = f(x) * (1 - f(x))
 * 
 * No learnable parameters.
 */
class Sigmoid : public Layer {
public:
    Sigmoid(BackendPtr backend) : Layer(backend), output_cache_({1,1}, backend) {}
    
    /**
     * Forward pass: apply sigmoid element-wise
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * Backward pass: multiply gradient by sigmoid derivative
     */
    Tensor backward(const Tensor& grad_output) override;
    
    /**
     * No parameters to update
     */
    void update_parameters(float learning_rate) override {
        (void)learning_rate;
    }
    
private:
    Tensor output_cache_;  // Cache output for backward pass
};

} // namespace nn