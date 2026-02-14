#include "dense.h"
#include <cmath>
#include <random>

namespace nn {

Dense::Dense(size_t input_dim, size_t output_dim, BackendPtr backend)
    : Layer(backend),
      input_dim_(input_dim),
      output_dim_(output_dim),
      weights_({input_dim, output_dim}, backend),
      bias_({1, output_dim}, backend),
      grad_weights_({input_dim, output_dim}, backend),
      grad_bias_({1, output_dim}, backend),
      input_cache_({1, 1}, backend) {  // Will be resized in forward
    
    initialize_weights();
    
    // Initialize gradients to zero
    grad_weights_.fill(0.0f);
    grad_bias_.fill(0.0f);
}

Tensor Dense::forward(const Tensor& input) {
    // Check input shape
    if (input.cols() != input_dim_) {
        throw std::runtime_error("Dense::forward: input dimension mismatch. Expected " +
                                std::to_string(input_dim_) + " but got " +
                                std::to_string(input.cols()));
    }
    
    // Cache input for backward pass
    input_cache_ = input;
    
    size_t batch_size = input.rows();
    
    // output = input * weights
    Tensor output = input.matmul(weights_);  // (batch_size x output_dim)
    
    // Add bias to each row
    // We need to broadcast bias across all batch samples
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            output.data()[i * output_dim_ + j] += bias_.data()[j];
        }
    }
    
    return output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    size_t batch_size = grad_output.rows();
    
    // 1. Compute gradient w.r.t. input: grad_input = grad_output * W^T
    Tensor weights_T = weights_.transpose();
    Tensor grad_input = grad_output.matmul(weights_T);
    
    // 2. Compute gradient w.r.t. weights: grad_W = input^T * grad_output
    Tensor input_T = input_cache_.transpose();
    Tensor grad_W = input_T.matmul(grad_output);
    
    // Accumulate gradient (for mini-batch training, we average later)
    grad_weights_ = grad_W;
    
    // 3. Compute gradient w.r.t. bias: grad_b = sum(grad_output, axis=0)
    // Sum gradients across all batch samples
    grad_bias_.fill(0.0f);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            grad_bias_.data()[j] += grad_output.data()[i * output_dim_ + j];
        }
    }
    
    return grad_input;
}

void Dense::update_parameters(float learning_rate) {
    // W = W - lr * grad_W
    // b = b - lr * grad_b
    
    // Update weights
    for (size_t i = 0; i < input_dim_ * output_dim_; ++i) {
        weights_.data()[i] -= learning_rate * grad_weights_.data()[i];
    }
    
    // Update bias
    for (size_t i = 0; i < output_dim_; ++i) {
        bias_.data()[i] -= learning_rate * grad_bias_.data()[i];
    }
}

void Dense::initialize_weights() {
    // Xavier/Glorot initialization: sample from U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
    // This helps with gradient flow in deep networks
    
    float limit = std::sqrt(6.0f / (input_dim_ + output_dim_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    // Initialize weights
    for (size_t i = 0; i < input_dim_ * output_dim_; ++i) {
        weights_.data()[i] = dist(gen);
    }
    
    // Initialize bias to small positive values
    for (size_t i = 0; i < output_dim_; ++i) {
        bias_.data()[i] = 0.01f;
    }
}

} // namespace nn