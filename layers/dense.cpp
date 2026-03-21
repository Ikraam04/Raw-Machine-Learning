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
    backend_->bias_add(output.data(), bias_.data(), batch_size, output_dim_);
    
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
    backend_->sum_rows(grad_bias_.data(), grad_output.data(), batch_size, output_dim_);
    
    return grad_input;
}

void Dense::update_parameters(float lr) {
    ++t_;

    // Adam (Adaptive Moment Estimation)
    // vanilla SGD just does: param -= lr * grad
    // problem: same lr for every param regardless of how noisy/consistent the gradient is
    //
    // Adam fixes this by tracking two "moments" per param:
    //   m = running avg of the gradient itself       (where is it going?)
    //   v = running avg of the squared gradient      (how noisy is it?)
    //
    // params with large/noisy gradients get a smaller effective lr (v is big → divides harder)
    // params with small/consistent gradients get a larger effective lr (v is small)
    // this makes training way more stable and less sensitive to lr choice
    //
    // beta1/beta2 control how much history to keep (0.9 = 90% old, 10% new)
    // bc1/bc2 correct for the fact that m and v start at 0 — without this
    // the first few steps would be massively underestimated
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps   = 1e-8f;

    // bias correction terms for step t
    float bc1 = 1.0f - std::pow(beta1, (float)t_);
    float bc2 = 1.0f - std::pow(beta2, (float)t_);

    // weights
    backend_->adam_update(weights_.data(), grad_weights_.data(),
                          m_weights_.data(), v_weights_.data(),
                          lr, beta1, beta2, bc1, bc2, eps,
                          input_dim_ * output_dim_);

    // bias
    backend_->adam_update(bias_.data(), grad_bias_.data(),
                          m_bias_.data(), v_bias_.data(),
                          lr, beta1, beta2, bc1, bc2, eps,
                          output_dim_);
}

void Dense::initialize_weights() {
    // Xavier/Glorot: uniform(-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
    // keeps variance stable through layers
    float limit = std::sqrt(6.0f / (input_dim_ + output_dim_));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    std::vector<float> host_weights(input_dim_ * output_dim_);
    for (size_t i = 0; i < host_weights.size(); ++i) {
        host_weights[i] = dist(gen);
    }
    weights_.set_data(host_weights);

    // small positive bias to start
    std::vector<float> host_bias(output_dim_, 0.01f);
    bias_.set_data(host_bias);
}

} // namespace nn