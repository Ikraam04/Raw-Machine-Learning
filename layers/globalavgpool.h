#pragma once
#include "layer.h"

namespace nn {

// Global Average Pooling
//
// after conv layers you have a tensor shaped {batch, C, H, W}
// C feature maps, each H×W pixels. you need to get this into a Dense layer
// which wants 2D input {batch, features}
//
// option 1: Flatten → gives {batch, C*H*W} — huge, lots of params
// option 2: GAP → takes the *average* of each H×W feature map → {batch, C}
// much smaller, and it forces the network to care about the whole feature map
// rather than just memorising where stuff appears spatially
//
// forward:  output[n,c] = mean of all input[n,c,h,w] over h and w
// backward: grad just gets distributed evenly back to every spatial position
//           grad_input[n,c,h,w] = grad_output[n,c] / (H*W)
//
// no learnable params
class GlobalAvgPool : public Layer {
public:
    GlobalAvgPool(BackendPtr backend) : Layer(backend) {}

    Tensor forward(const Tensor& input) override {
        if (input.shape().size() != 4) {
            throw std::runtime_error("GlobalAvgPool expects a 4D tensor {batch, C, H, W}");
        }

        int batch    = (int)input.shape()[0];
        int channels = (int)input.shape()[1];
        int h        = (int)input.shape()[2];
        int w        = (int)input.shape()[3];

        cached_channels_ = channels;
        cached_h_ = h;
        cached_w_ = w;

        Tensor output({(size_t)batch, (size_t)channels}, backend_);

        backend_->global_avg_pool_forward(input.data(), output.data(),
                                           batch, channels, h, w);

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        int batch    = (int)grad_output.shape()[0];
        int channels = cached_channels_;
        int h        = cached_h_;
        int w        = cached_w_;

        Tensor grad_input({(size_t)batch, (size_t)channels,
                           (size_t)h,     (size_t)w}, backend_);

        backend_->global_avg_pool_backward(grad_output.data(), grad_input.data(),
                                            batch, channels, h, w);

        return grad_input;
    }

    void update_parameters(float /*lr*/) override {}

private:
    int cached_channels_ = 0;
    int cached_h_ = 0, cached_w_ = 0;
};

} // namespace nn
