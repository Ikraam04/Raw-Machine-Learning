#pragma once
#include "tensor.h"
#include <vector>

namespace nn {

// ============================================================
// Mean Squared Error (MSE) Loss
// ============================================================

/**
 * Compute Mean Squared Error Loss.
 * 
 * Loss = (1/N) * sum((prediction - target)^2)
 * 
 * Used for regression tasks or as alternative for classification.
 * 
 * @param predictions: Network output (batch_size x output_dim)
 * @param targets: Target values (flat vector)
 * @param batch_size: Number of samples in batch
 * @param output_dim: Dimension of output (e.g., 10 for MNIST)
 * @return: Average MSE loss across batch
 */
float mse_loss(const Tensor& predictions,
               const std::vector<float>& targets,
               size_t batch_size,
               size_t output_dim);

/**
 * Compute gradient of MSE loss w.r.t. predictions.
 * 
 * Gradient = 2 * (prediction - target) / batch_size
 * 
 * @param predictions: Network output
 * @param targets: Target values
 * @param grad_output: Output tensor to fill with gradient
 * @param batch_size: Number of samples
 * @param output_dim: Output dimension
 */
void mse_gradient(const Tensor& predictions,
                  const std::vector<float>& targets,
                  Tensor& grad_output,
                  size_t batch_size,
                  size_t output_dim);

// ============================================================
// Cross-Entropy Loss
// ============================================================

/**
 * Compute Cross-Entropy Loss for classification.
 * 
 * Loss = -sum(target * log(prediction + epsilon))
 * 
 * Used with Softmax output layer.
 * Epsilon prevents log(0) = -inf.
 * 
 * @param predictions: Softmax output (batch_size x num_classes)
 * @param targets: One-hot encoded targets (flat vector: batch_size * num_classes)
 * @param batch_size: Number of samples in batch
 * @param num_classes: Number of classes (usually 10 for MNIST)
 * @return: Average loss across batch
 */
float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes);

/**
 * Compute gradient of Cross-Entropy loss w.r.t. predictions.
 * 
 * When combined with Softmax, gradient simplifies to:
 * grad = (prediction - target) / batch_size
 * 
 * This is numerically stable and efficient.
 * 
 * @param predictions: Softmax output (batch_size x num_classes)
 * @param targets: One-hot encoded targets
 * @param grad_output: Output tensor to fill with gradient (batch_size x num_classes)
 * @param batch_size: Number of samples
 * @param num_classes: Number of classes
 */
void cross_entropy_gradient(const Tensor& predictions,
                           const std::vector<float>& targets,
                           Tensor& grad_output,
                           size_t batch_size,
                           size_t num_classes);

/**
 * Combined Softmax + Cross-Entropy loss and gradient.
 * 
 * This is more numerically stable than computing them separately.
 * For logits z and target y:
 * 
 * Loss = -sum(y * log(softmax(z)))
 * Gradient = softmax(z) - y
 * 
 * The gradient is remarkably simple!
 * 
 * @param logits: Raw network outputs before softmax (batch_size x num_classes)
 * @param targets: One-hot encoded targets
 * @param grad_output: Output gradient (batch_size x num_classes)
 * @param batch_size: Number of samples
 * @param num_classes: Number of classes
 * @return: Average loss
 */
float softmax_cross_entropy_loss_and_gradient(const Tensor& logits,
                                              const std::vector<float>& targets,
                                              Tensor& grad_output,
                                              size_t batch_size,
                                              size_t num_classes);

} // namespace nn