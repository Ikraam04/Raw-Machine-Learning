#pragma once
#include "layer.h"
#include <vector>
#include <memory>

namespace nn {

// chains layers: input -> layer1 -> layer2 -> ... -> output
// backward runs in reverse
class Sequential {
public:
    Sequential() = default;

    void add(std::shared_ptr<Layer> layer);  // layers run in order added

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);  // grad of loss w.r.t. network output
    void update_parameters(float learning_rate);

    size_t size() const { return layers_.size(); }

    std::shared_ptr<Layer> get_layer(size_t index) {
        return layers_[index];
    }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

} // namespace nn