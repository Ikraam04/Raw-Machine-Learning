#include "loss.h"
#include <cmath>
#include <algorithm>

namespace nn {

float mse_loss(const Tensor& predictions,
               const std::vector<float>& targets,
               size_t batch_size,
               size_t output_dim) 
               {
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < output_dim; ++c) {
            size_t idx = b * output_dim + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            total_loss += (pred - target) * (pred - target);  // (pred - target)^2
        }
    }

    return total_loss / (batch_size * output_dim);  // average over batch and dims
}

void mse_gradient(const Tensor& predictions,
                  const std::vector<float>& targets,
                  Tensor& grad_output,
                  size_t batch_size,
                  size_t output_dim) 
                  {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < output_dim; ++c) {
            size_t idx = b * output_dim + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            grad_output.data()[idx] = 2.0f * (pred - target) / (batch_size * output_dim);
        }
    }
}

float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes) {
    const float epsilon = 1e-7f;  // clamp to avoid log(0)
    float total_loss = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];

            if (target > 0.5f) {  // only the true class contributes to loss
                pred = std::max(epsilon, std::min(1.0f - epsilon, pred));
                total_loss += -std::log(pred);
            }
        }
    }

    return total_loss / batch_size;
}

void cross_entropy_gradient(const Tensor& predictions,
                           const std::vector<float>& targets,
                           Tensor& grad_output,
                           size_t batch_size,
                           size_t num_classes) {
    // grad = (prediction - target) / batch_size
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            grad_output.data()[idx] = (pred - target) / batch_size;
        }
    }
}

float softmax_cross_entropy_loss_and_gradient(const Tensor& logits,
                                              const std::vector<float>& targets,
                                              Tensor& grad_output,
                                              size_t batch_size,
                                              size_t num_classes) {
    const float epsilon = 1e-7f;
    float total_loss = 0.0f;

    for (size_t b = 0; b < batch_size; ++b) {
        const float* logit_row = logits.data() + b * num_classes;
        float* grad_row = grad_output.data() + b * num_classes;

        // softmax with stability trick (subtract max before exp)
        float max_logit = logit_row[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (logit_row[c] > max_logit) {
                max_logit = logit_row[c];
            }
        }

        float sum_exp = 0.0f;
        float softmax[10];  // hardcoded to 10 for MNIST - would need dynamic alloc for other sizes
        for (size_t c = 0; c < num_classes; ++c) {
            softmax[c] = std::exp(logit_row[c] - max_logit);
            sum_exp += softmax[c];
        }

        for (size_t c = 0; c < num_classes; ++c) {
            softmax[c] /= sum_exp;
        }

        // loss
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float target = targets[idx];

            if (target > 0.5f) {  // only the true class contributes
                float pred = std::max(epsilon, std::min(1.0f - epsilon, softmax[c]));
                total_loss += -std::log(pred);
            }
        }

        // gradient = (softmax - target) / batch_size
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            grad_row[c] = (softmax[c] - targets[idx]) / batch_size;
        }
    }
    
    return total_loss / batch_size;
}

} // namespace nn