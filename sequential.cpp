#include "sequential.h"

namespace nn {

void Sequential::add(std::shared_ptr<Layer> layer) {
    layers_.push_back(layer);
}

Tensor Sequential::forward(const Tensor& input) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential network has no layers");
    }
    
    // Pass input through each layer sequentially
    Tensor output = input;
    for (auto& layer : layers_) {
        output = layer->forward(output);
    }
    
    return output;
}

void Sequential::backward(const Tensor& grad_output) {
    if (layers_.empty()) {
        throw std::runtime_error("Sequential network has no layers");
    }
    
    // Backpropagate through layers in reverse order
    Tensor grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void Sequential::update_parameters(float learning_rate) {
    // Update parameters in all layers
    for (auto& layer : layers_) {
        layer->update_parameters(learning_rate);
    }
}

} // namespace nn