#include "loss.h"
#include <cmath>
#include <algorithm>

namespace nn {

float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes) {
    const float epsilon = 1e-7f;  // Prevent log(0)
    float total_loss = 0.0f;
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            
            // Only compute loss for true class (target = 1)
            if (target > 0.5f) {
                // Clamp prediction to [epsilon, 1 - epsilon]
                pred = std::max(epsilon, std::min(1.0f - epsilon, pred));
                total_loss += -std::log(pred);
            }
        }
    }
    
    return total_loss / batch_size;  // Average over batch
}

void cross_entropy_gradient(const Tensor& predictions,
                           const std::vector<float>& targets,
                           Tensor& grad_output,
                           size_t batch_size,
                           size_t num_classes) {
    // Gradient of cross-entropy w.r.t. softmax output:
    // grad = (prediction - target) / batch_size
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float pred = predictions.data()[idx];
            float target = targets[idx];
            
            // Simple and numerically stable
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
    
    // Process each sample
    for (size_t b = 0; b < batch_size; ++b) {
        const float* logit_row = logits.data() + b * num_classes;
        float* grad_row = grad_output.data() + b * num_classes;
        
        // ============================================================
        // Compute Softmax (with numerical stability)
        // ============================================================
        
        // Find max for stability
        float max_logit = logit_row[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (logit_row[c] > max_logit) {
                max_logit = logit_row[c];
            }
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        float softmax[10];  // Assuming num_classes = 10 for MNIST
        for (size_t c = 0; c < num_classes; ++c) {
            softmax[c] = std::exp(logit_row[c] - max_logit);
            sum_exp += softmax[c];
        }
        
        // Normalize
        for (size_t c = 0; c < num_classes; ++c) {
            softmax[c] /= sum_exp;
        }
        
        // ============================================================
        // Compute Loss
        // ============================================================
        
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            float target = targets[idx];
            
            if (target > 0.5f) {  // This is the true class
                float pred = std::max(epsilon, std::min(1.0f - epsilon, softmax[c]));
                total_loss += -std::log(pred);
            }
        }
        
        // ============================================================
        // Compute Gradient
        // ============================================================
        
        // Beautiful simplification: gradient = (softmax - target) / batch_size
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            grad_row[c] = (softmax[c] - targets[idx]) / batch_size;
        }
    }
    
    return total_loss / batch_size;
}

} // namespace nn