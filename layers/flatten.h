#pragma once
#include "layer.h"

namespace nn {

// Reshapes a 4D tensor {batch, C, H, W} into 2D {batch, C*H*W}
// No data is moved — just reinterprets the shape for the next Dense layer
// backward restores the original shape
class Flatten : public Layer {
public:
    Flatten(BackendPtr backend) : Layer(backend) {}

    Tensor forward(const Tensor& input) override {
        input_shape_ = input.shape();
        size_t batch = input.shape()[0];
        size_t flat  = input.size() / batch;

        Tensor output({batch, flat}, backend_);
        backend_->copy(output.data(), input.data(), input.size());
        return output;
    }

    // restore original shape so Conv2D layers earlier in the net get the right grad
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_shape_, backend_);
        backend_->copy(grad_input.data(), grad_output.data(), grad_output.size());
        return grad_input;
    }

    void update_parameters(float /*learning_rate*/) override {}

private:
    std::vector<size_t> input_shape_;
};

} // namespace nn
