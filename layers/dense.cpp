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
      input_cache_({1, 1}, backend),
      m_weights_({input_dim, output_dim}, backend),
      v_weights_({input_dim, output_dim}, backend),
      m_bias_({1, output_dim}, backend),
      v_bias_({1, output_dim}, backend) {

    initialize_weights();

    grad_weights_.fill(0.0f);
    grad_bias_.fill(0.0f);
    m_weights_.fill(0.0f);
    v_weights_.fill(0.0f);
    m_bias_.fill(0.0f);
    v_bias_.fill(0.0f);
}

Tensor Dense::forward(const Tensor& input) {
    if (input.cols() != input_dim_) {
        throw std::runtime_error("Dense::forward: input dimension mismatch. Expected " +
                                std::to_string(input_dim_) + " but got " +
                                std::to_string(input.cols()));
    }
    
    input_cache_ = input;

    size_t batch_size = input.rows();

    Tensor output = input.matmul(weights_);  // (batch_size x output_dim)

    // broadcast bias across the batch
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            output.data()[i * output_dim_ + j] += bias_.data()[j];
        }
    }
    
    return output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    size_t batch_size = grad_output.rows();
    
    // grad_input = grad_output * W^T
    Tensor weights_T = weights_.transpose();
    Tensor grad_input = grad_output.matmul(weights_T);

    // grad_W = input^T * grad_output
    Tensor input_T = input_cache_.transpose();
    Tensor grad_W = input_T.matmul(grad_output);
    grad_weights_ = grad_W;

    // grad_b = sum(grad_output, axis=0) - just add up across the batch
    grad_bias_.fill(0.0f);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_dim_; ++j) {
            grad_bias_.data()[j] += grad_output.data()[i * output_dim_ + j];
        }
    }
    
    return grad_input;
}

void Dense::update_parameters(float lr) {
    ++t_;

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps   = 1e-8f;

    // bias-corrected effective step size
    float bc1   = 1.0f - std::pow(beta1, (float)t_);
    float bc2   = 1.0f - std::pow(beta2, (float)t_);

    // weights
    for (size_t i = 0; i < input_dim_ * output_dim_; ++i) {
        float g = grad_weights_.data()[i];
        m_weights_.data()[i] = beta1 * m_weights_.data()[i] + (1.0f - beta1) * g;
        v_weights_.data()[i] = beta2 * v_weights_.data()[i] + (1.0f - beta2) * g * g;
        float m_hat = m_weights_.data()[i] / bc1;
        float v_hat = v_weights_.data()[i] / bc2;
        weights_.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }

    // bias
    for (size_t i = 0; i < output_dim_; ++i) {
        float g = grad_bias_.data()[i];
        m_bias_.data()[i] = beta1 * m_bias_.data()[i] + (1.0f - beta1) * g;
        v_bias_.data()[i] = beta2 * v_bias_.data()[i] + (1.0f - beta2) * g * g;
        float m_hat = m_bias_.data()[i] / bc1;
        float v_hat = v_bias_.data()[i] / bc2;
        bias_.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

void Dense::initialize_weights() {
    // Xavier/Glorot: uniform(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
    // keeps variance stable through layers
    float limit = std::sqrt(6.0f / (input_dim_ + output_dim_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < input_dim_ * output_dim_; ++i) {
        weights_.data()[i] = dist(gen);
    }

    // small positive bias to start
    for (size_t i = 0; i < output_dim_; ++i) {
        bias_.data()[i] = 0.01f;
    }
}

} // namespace nn