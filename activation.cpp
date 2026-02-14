#include "activation.h"

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

} // namespace nn