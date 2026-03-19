#pragma once
#include "layer.h"

namespace nn {

// 2D convolutional layer
//
// forward:  output = conv(input, weights) + bias
// backward: computes grad_input, grad_weights, grad_bias via im2col + matmul
//
// tensor layout: NCHW  (batch, channels, height, width)
//   input:  {batch, in_channels,  H,     W    }
//   output: {batch, out_channels, out_h, out_w}
//
// out_h = (H + 2*pad_h - kernel_h) / stride_h + 1
// out_w = (W + 2*pad_w - kernel_w) / stride_w + 1
//
// weights shape: {in_channels * kernel_h * kernel_w, out_channels}  (matches Dense convention)
// bias shape:    {1, out_channels}
class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels,
           int kernel_h, int kernel_w,
           int stride_h, int stride_w,
           int pad_h, int pad_w,
           BackendPtr backend);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float learning_rate) override;

    const Tensor& weights() const { return weights_; }
    const Tensor& bias()    const { return bias_; }

private:
    int in_channels_, out_channels_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;

    Tensor weights_;      // {in_channels*kernel_h*kernel_w, out_channels}
    Tensor bias_;         // {1, out_channels}
    Tensor grad_weights_; // same shape as weights_
    Tensor grad_bias_;    // same shape as bias_

    // Adam moment tensors (same shapes as weights_ / bias_)
    Tensor m_weights_;
    Tensor v_weights_;
    Tensor m_bias_;
    Tensor v_bias_;
    int t_ = 0;

    // cached from forward, needed for backward
    Tensor col_cache_;    // im2col output: {batch*out_h*out_w, in_channels*kernel_h*kernel_w}
    int cached_batch_ = 0;
    int cached_in_h_  = 0, cached_in_w_  = 0;
    int cached_out_h_ = 0, cached_out_w_ = 0;

    // Xavier/Glorot init: uniform(-sqrt(6 / (fan_in + fan_out)), ...)
    // fan_in  = in_channels  * kernel_h * kernel_w
    // fan_out = out_channels * kernel_h * kernel_w
    void initialize_weights();
};

} // namespace nn
