#pragma once
#include "layer.h"
#include <vector>
#include <memory>

namespace nn {

/**
 * Sequential network - chains layers together.
 * 
 * Forward pass: input → layer1 → layer2 → ... → output
 * Backward pass: grad flows backwards through all layers
 */
class Sequential {
public:
    Sequential() = default;
    
    /**
     * Add a layer to the network.
     * Layers are executed in the order they're added.
     */
    void add(std::shared_ptr<Layer> layer);
    
    /**
     * Forward pass through all layers.
     */
    Tensor forward(const Tensor& input);
    
    /**
     * Backward pass through all layers (in reverse order).
     * 
     * @param grad_output: Gradient of loss w.r.t. network output
     */
    void backward(const Tensor& grad_output);
    
    /**
     * Update all layer parameters.
     */
    void update_parameters(float learning_rate);
    
    /**
     * Get number of layers.
     */
    size_t size() const { return layers_.size(); }
    
    /**
     * Get a specific layer (for inspection).
     */
    std::shared_ptr<Layer> get_layer(size_t index) {
        return layers_[index];
    }
    
private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

} // namespace nn