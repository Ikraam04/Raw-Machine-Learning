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

    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input.data() + b * num_classes;
        float* output_row = output.data() + b * num_classes;

        // subtract max before exp to avoid overflow (doesn't change the result)
        float max_val = input_row[0];
        for (size_t i = 1; i < num_classes; ++i) {
            if (input_row[i] > max_val) {
                max_val = input_row[i];
            }
        }

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; ++i) {
            output_row[i] = std::exp(input_row[i] - max_val);
            sum += output_row[i];
        }

        for (size_t i = 0; i < num_classes; ++i) {
            output_row[i] /= sum;
        }
    }

    output_cache_ = output;
    return output;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    size_t batch_size = grad_output.rows();
    size_t num_classes = grad_output.cols();

    Tensor grad_input({batch_size, num_classes}, backend_);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* softmax = output_cache_.data() + b * num_classes;
        const float* grad_out = grad_output.data() + b * num_classes;
        float* grad_in = grad_input.data() + b * num_classes;

        // Jacobian: grad_input[i] = sum_j(softmax[i] * (d_ij - softmax[j]) * grad_output[j])
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