#pragma once
#include "core/tensor.h"
#include <vector>

namespace nn {

// MSE loss: (1/N) * sum((prediction - target)^2)
// good for regression, can use for classification too
float mse_loss(const Tensor& predictions,
               const std::vector<float>& targets,
               size_t batch_size,
               size_t output_dim);

// MSE gradient: 2 * (prediction - target) / batch_size
void mse_gradient(const Tensor& predictions,
                  const std::vector<float>& targets,
                  Tensor& grad_output,
                  size_t batch_size,
                  size_t output_dim);

// cross-entropy loss: -sum(target * log(prediction + epsilon))
// use with softmax output, epsilon stops log(0) = -inf
float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes);

// cross-entropy gradient with softmax simplifies nicely to:
// grad = (prediction - target) / batch_size
void cross_entropy_gradient(const Tensor& predictions,
                           const std::vector<float>& targets,
                           Tensor& grad_output,
                           size_t batch_size,
                           size_t num_classes);

// fused softmax + cross-entropy - more numerically stable than doing them separately
// loss = -sum(y * log(softmax(z)))
// gradient = softmax(z) - y  (really clean formula)
float softmax_cross_entropy_loss_and_gradient(const Tensor& logits,
                                              const std::vector<float>& targets,
                                              Tensor& grad_output,
                                              size_t batch_size,
                                              size_t num_classes);

} // namespace nn