#include "loss.h"
#include <cmath>
#include <algorithm>

namespace nn {

float mse_loss(const Tensor& predictions,
               const std::vector<float>& targets,
               size_t batch_size,
               size_t output_dim)
               {
    size_t total = batch_size * output_dim;
    std::vector<float> preds(total);
    predictions.backend()->download(preds.data(), predictions.data(), total);

    float total_loss = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        float diff = preds[i] - targets[i];
        total_loss += diff * diff;
    }

    return total_loss / total;
}

void mse_gradient(const Tensor& predictions,
                  const std::vector<float>& targets,
                  Tensor& grad_output,
                  size_t batch_size,
                  size_t output_dim)
                  {
    size_t total = batch_size * output_dim;
    std::vector<float> preds(total);
    predictions.backend()->download(preds.data(), predictions.data(), total);

    std::vector<float> grad(total);
    for (size_t i = 0; i < total; ++i) {
        grad[i] = 2.0f * (preds[i] - targets[i]) / total;
    }

    grad_output.set_data(grad);
}

float cross_entropy_loss(const Tensor& predictions,
                        const std::vector<float>& targets,
                        size_t batch_size,
                        size_t num_classes) {
    const float epsilon = 1e-7f;
    size_t total = batch_size * num_classes;
    std::vector<float> preds(total);
    predictions.backend()->download(preds.data(), predictions.data(), total);

    float total_loss = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        if (targets[i] > 0.5f) {
            float pred = std::max(epsilon, std::min(1.0f - epsilon, preds[i]));
            total_loss += -std::log(pred);
        }
    }

    return total_loss / batch_size;
}

void cross_entropy_gradient(const Tensor& predictions,
                           const std::vector<float>& targets,
                           Tensor& grad_output,
                           size_t batch_size,
                           size_t num_classes) {
    size_t total = batch_size * num_classes;
    std::vector<float> preds(total);
    predictions.backend()->download(preds.data(), predictions.data(), total);

    std::vector<float> grad(total);
    for (size_t i = 0; i < total; ++i) {
        grad[i] = (preds[i] - targets[i]) / batch_size;
    }

    grad_output.set_data(grad);
}

float softmax_cross_entropy_loss_and_gradient(const Tensor& logits,
                                              const std::vector<float>& targets,
                                              Tensor& grad_output,
                                              size_t batch_size,
                                              size_t num_classes) {
    size_t total = batch_size * num_classes;
    std::vector<float> logit_host(total);
    logits.backend()->download(logit_host.data(), logits.data(), total);

    const float epsilon = 1e-7f;
    float total_loss = 0.0f;
    std::vector<float> grad(total);

    for (size_t b = 0; b < batch_size; ++b) {
        const float* logit_row = logit_host.data() + b * num_classes;

        // softmax with stability trick (subtract max before exp)
        float max_logit = logit_row[0];
        for (size_t c = 1; c < num_classes; ++c) {
            if (logit_row[c] > max_logit) {
                max_logit = logit_row[c];
            }
        }

        float sum_exp = 0.0f;
        std::vector<float> softmax(num_classes);
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

            if (target > 0.5f) {
                float pred = std::max(epsilon, std::min(1.0f - epsilon, softmax[c]));
                total_loss += -std::log(pred);
            }
        }

        // gradient = (softmax - target) / batch_size
        for (size_t c = 0; c < num_classes; ++c) {
            size_t idx = b * num_classes + c;
            grad[idx] = (softmax[c] - targets[idx]) / batch_size;
        }
    }

    grad_output.set_data(grad);
    return total_loss / batch_size;
}

} // namespace nn
