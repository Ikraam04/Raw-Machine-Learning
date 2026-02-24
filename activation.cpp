#include "activation.h"
#include <cmath>  // For expf
namespace nn {

// ============================================================
// ReLU Layer
// ============================================================

Tensor ReLU::forward(const Tensor& input) {
    // Cache input for backward pass
    input_cache_ = input;
    
    // Apply ReLU: max(0, x)
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    // ReLU derivative: 1 if input > 0, else 0
    Tensor relu_grad = input_cache_.relu_derivative();
    
    // Element-wise multiply: grad_input = grad_output * relu'(input)
    return grad_output.multiply(relu_grad);
}

// ============================================================
// Sigmoid Layer
// ============================================================

Tensor Sigmoid::forward(const Tensor& input) {
    // Apply sigmoid
    Tensor output = input.sigmoid();
    
    // Cache output (not input!) because sigmoid derivative uses output
    output_cache_ = output;
    
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    // We cached sigmoid(x) as output_cache_
    Tensor sigmoid_grad = output_cache_.sigmoid_derivative();
    
    // Element-wise multiply: grad_input = grad_output * sigmoid'(input)
    return grad_output.multiply(sigmoid_grad);
}

// ============================================================
// Softmax Layer
// ============================================================

Tensor Softmax::forward(const Tensor& input) {
    size_t batch_size = input.rows();
    size_t num_classes = input.cols();
    
    Tensor output({batch_size, num_classes}, backend_);
    
    // Apply softmax to each row (sample) independently
    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input.data() + b * num_classes;
        float* output_row = output.data() + b * num_classes;
        
        // Find max for numerical stability (prevent overflow)
        float max_val = input_row[0];
        for (size_t i = 1; i < num_classes; ++i) {
            if (input_row[i] > max_val) {
                max_val = input_row[i];
            }
        }
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            output_row[i] = std::exp(input_row[i] - max_val);
            sum += output_row[i];
        }
        
        // Normalize to get probabilities
        for (size_t i = 0; i < num_classes; ++i) {
            output_row[i] /= sum;
        }
    }
    
    // Cache output for backward pass
    output_cache_ = output;
    
    return output;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    size_t batch_size = grad_output.rows();
    size_t num_classes = grad_output.cols();
    
    Tensor grad_input({batch_size, num_classes}, backend_);
    
    // Compute gradient for each sample
    for (size_t b = 0; b < batch_size; ++b) {
        const float* softmax = output_cache_.data() + b * num_classes;
        const float* grad_out = grad_output.data() + b * num_classes;
        float* grad_in = grad_input.data() + b * num_classes;
        
        // Jacobian matrix computation
        // grad_input[i] = sum_j (softmax[i] * (δ_ij - softmax[j]) * grad_output[j])
        for (size_t i = 0; i < num_classes; ++i) {
            grad_in[i] = 0.0f;
            for (size_t j = 0; j < num_classes; ++j) {
                float delta = (i == j) ? 1.0f : 0.0f;
                grad_in[i] += softmax[i] * (delta - softmax[j]) * grad_out[j];
            }
        }
    }
    
    return grad_input;
}

} // namespace nn