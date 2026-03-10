#pragma once
#include "layer.h"
#include <random>

namespace nn {

// fully connected layer
// forward: output = input * weights + bias
// weights is (input_dim x output_dim), bias is (output_dim,)
class Dense : public Layer {
public:
    Dense(size_t input_dim, size_t output_dim, BackendPtr backend);

    // forward: (batch_size x input_dim) -> (batch_size x output_dim)
    Tensor forward(const Tensor& input) override;

    // backward: given grad_output (batch_size x output_dim), computes:
    //   grad_input   = grad_output * W^T
    //   grad_weights = input^T * grad_output
    //   grad_bias    = sum(grad_output, axis=0)
    Tensor backward(const Tensor& grad_output) override;

    // Adam update: adapts lr per-parameter using gradient moment estimates
    void update_parameters(float learning_rate) override;

    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }
    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }

    const Tensor& grad_weights() const { return grad_weights_; }
    const Tensor& grad_bias() const { return grad_bias_; }

private:
    size_t input_dim_;
    size_t output_dim_;

    Tensor weights_;      // (input_dim x output_dim)
    Tensor bias_;         // (1 x output_dim)
    Tensor grad_weights_; // same shape as weights_
    Tensor grad_bias_;    // same shape as bias_
    Tensor input_cache_;  // saved from forward for use in backward

    // Adam moment tensors (same shapes as weights_ / bias_)
    Tensor m_weights_;  // first moment (mean) of weight gradients
    Tensor v_weights_;  // second moment (variance) of weight gradients
    Tensor m_bias_;     // first moment of bias gradients
    Tensor v_bias_;     // second moment of bias gradients
    int t_ = 0;         // step counter for bias correction

    // Xavier/Glorot init: uniform(-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out)))
    void initialize_weights();
};

} // namespace nn