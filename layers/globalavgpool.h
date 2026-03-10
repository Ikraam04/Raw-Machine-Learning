#pragma once
#include "layer.h"

namespace nn {

// Global Average Pooling: collapses each spatial feature map to a single value
//
// forward:  input {batch, C, H, W} -> output {batch, C}
//           output[n,c] = mean over all (h,w) of input[n,c,h,w]
//
// backward: gradient distributes uniformly back across all spatial positions
//           grad_input[n,c,h,w] = grad_output[n,c] / (H * W)
//
// no learnable parameters
// output plugs directly into a Dense layer (2D tensor, shape {batch, C})
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

        float inv_hw = 1.0f / (float)(h * w);
        const float* in = input.data();
        float*      out = output.data();

        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                const float* fm = in + n * channels * h * w + c * h * w;
                for (int i = 0; i < h * w; ++i) {
                    sum += fm[i];
                }
                out[n * channels + c] = sum * inv_hw;
            }
        }

        return output;
    }

    Tensor backward(const Tensor& grad_output) override {
        // grad_output: {batch, channels}
        int batch    = (int)grad_output.shape()[0];
        int channels = cached_channels_;
        int h        = cached_h_;
        int w        = cached_w_;

        Tensor grad_input({(size_t)batch, (size_t)channels,
                           (size_t)h,     (size_t)w}, backend_);

        float inv_hw = 1.0f / (float)(h * w);
        const float* grad_out = grad_output.data();
        float*       grad_in  = grad_input.data();

        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channels; ++c) {
                float g = grad_out[n * channels + c] * inv_hw;
                float* fm = grad_in + n * channels * h * w + c * h * w;
                for (int i = 0; i < h * w; ++i) {
                    fm[i] = g;
                }
            }
        }

        return grad_input;
    }

    void update_parameters(float /*lr*/) override {}

private:
    int cached_channels_ = 0;
    int cached_h_ = 0, cached_w_ = 0;
};

} // namespace nn
