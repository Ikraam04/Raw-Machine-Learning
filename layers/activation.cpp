#include "activation.h"
#include <cmath>
namespace nn {

Tensor ReLU::forward(const Tensor& input) {
    input_cache_ = input;
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor relu_grad = input_cache_.relu_derivative();
    return grad_output.multiply(relu_grad);
}

Tensor Sigmoid::forward(const Tensor& input) {
    Tensor output = input.sigmoid();
    output_cache_ = output;  // cache output (not input) since derivative uses sigmoid(x)
    return output;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)), we have sigmoid(x) cached
    Tensor sigmoid_grad = output_cache_.sigmoid_derivative();
    return grad_output.multiply(sigmoid_grad);
}

Tensor Softmax::forward(const Tensor& input) {
    size_t batch_size = input.rows();
    size_t num_classes = input.cols();

    Tensor output({batch_size, num_classes}, backend_);

    backend_->softmax_forward(input.data(), output.data(), batch_size, num_classes);

    output_cache_ = output;
    return output;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    size_t batch_size = grad_output.rows();
    size_t num_classes = grad_output.cols();

    Tensor grad_input({batch_size, num_classes}, backend_);

    backend_->softmax_backward(output_cache_.data(), grad_output.data(),
                                grad_input.data(), batch_size, num_classes);

    return grad_input;
}

} // namespace nn
