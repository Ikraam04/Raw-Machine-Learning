#include "loss.h"
#include "core/backend_interface.h"
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
    auto backend = logits.backend();  // shared_ptr<Backend>
    size_t total = batch_size * num_classes;

    // upload targets to device (one-hot labels, batch x num_classes)
    Tensor targets_dev({batch_size, num_classes}, backend);
    targets_dev.set_data(targets);

    // scalar loss accumulator on device — must be zeroed before kernel
    Tensor loss_dev({1, 1}, backend);
    loss_dev.fill(0.0f);

    // everything stays on device: logits in, gradient out, loss accumulated
    backend->softmax_cross_entropy(
        logits.data(), targets_dev.data(),
        grad_output.data(), loss_dev.data(),
        (int)batch_size, (int)num_classes);

    // download only 1 float
    float loss = 0.0f;
    backend->download(&loss, loss_dev.data(), 1);
    return loss;
}

} // namespace nn
