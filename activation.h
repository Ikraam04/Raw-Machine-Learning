#pragma once
#include "layer.h"

namespace nn {

// f(x) = max(0, x), f'(x) = 1 if x > 0 else 0
// no learnable params
class ReLU : public Layer {
public:
    ReLU(BackendPtr backend) : Layer(backend), input_cache_({1,1}, backend) {}

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;  // grad * relu'(input)

    void update_parameters(float learning_rate) override {
        (void)learning_rate;  // nothing to update
    }

private:
    Tensor input_cache_;  // needed in backward to recompute the derivative
};

// f(x) = 1 / (1 + exp(-x)), f'(x) = f(x) * (1 - f(x))
// no learnable params
class Sigmoid : public Layer {
public:
    Sigmoid(BackendPtr backend) : Layer(backend), output_cache_({1,1}, backend) {}

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;  // grad * sigmoid'(input)

    void update_parameters(float learning_rate) override {
        (void)learning_rate;
    }

private:
    Tensor output_cache_;  // cache output (not input) since derivative uses sigmoid(x)
};

// softmax(x_i) = exp(x_i) / sum(exp(x_j)) per row
// turns raw logits into a probability distribution (rows sum to 1)
// normally you'd fuse this with cross-entropy for stability, but standalone works too
class Softmax : public Layer {
public:
    Softmax(BackendPtr backend)
        : Layer(backend),
          output_cache_({1, 1}, backend) {}

    // (N, C) -> (N, C), each row sums to 1
    Tensor forward(const Tensor& input) override;

    // full Jacobian: grad_input[i] = sum_j(softmax[i] * (d_ij - softmax[j]) * grad_output[j])
    // d_ij = 1 if i==j else 0 (Kronecker delta)
    Tensor backward(const Tensor& grad_output) override;

    void update_parameters(float learning_rate) override {
        (void)learning_rate;
    }

private:
    Tensor output_cache_;  // softmax output saved for backward
};

} //namespace nn