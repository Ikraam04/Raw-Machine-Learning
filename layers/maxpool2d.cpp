#include "maxpool2d.h"
#include <stdexcept>
#include <limits>

namespace nn {

Tensor MaxPool2D::forward(const Tensor& input) {
    if (input.shape().size() != 4) {
        throw std::runtime_error("MaxPool2D::forward expects a 4D tensor {batch, channels, H, W}");
    }

    int batch    = (int)input.shape()[0];
    int channels = (int)input.shape()[1];
    int in_h     = (int)input.shape()[2];
    int in_w     = (int)input.shape()[3];

    int out_h = (in_h - pool_h_) / stride_h_ + 1;
    int out_w = (in_w - pool_w_) / stride_w_ + 1;

    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("MaxPool2D: pool size larger than input spatial dims");
    }

    cached_batch_    = batch;
    cached_channels_ = channels;
    cached_in_h_     = in_h;  cached_in_w_  = in_w;
    cached_out_h_    = out_h; cached_out_w_ = out_w;

    // one index per output element — records which input position was the max
    max_indices_.resize((size_t)(batch * channels * out_h * out_w));

    Tensor output({(size_t)batch, (size_t)channels,
                   (size_t)out_h, (size_t)out_w}, backend_);

    const float* in_data  = input.data();
    float*       out_data = output.data();

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    // scan the pool_h × pool_w window
                    float max_val = -std::numeric_limits<float>::infinity();
                    int   max_idx = -1;

                    for (int ph = 0; ph < pool_h_; ++ph) {
                        for (int pw = 0; pw < pool_w_; ++pw) {
                            int ih = oh * stride_h_ + ph;
                            int iw = ow * stride_w_ + pw;
                            int flat_in = n * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + ih * in_w + iw;

                            if (in_data[flat_in] > max_val) {
                                max_val = in_data[flat_in];
                                max_idx = flat_in;
                            }
                        }
                    }

                    int flat_out = n * channels * out_h * out_w
                                 + c * out_h * out_w
                                 + oh * out_w + ow;

                    out_data[flat_out]    = max_val;
                    max_indices_[flat_out] = max_idx;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    // grad_output: {batch, channels, out_h, out_w}
    Tensor grad_input({(size_t)cached_batch_,    (size_t)cached_channels_,
                       (size_t)cached_in_h_,     (size_t)cached_in_w_}, backend_);
    grad_input.fill(0.0f);

    const float* grad_out  = grad_output.data();
    float*       grad_in   = grad_input.data();

    int total_out = cached_batch_ * cached_channels_ * cached_out_h_ * cached_out_w_;
    for (int i = 0; i < total_out; ++i) {
        // route gradient to whichever input position was the max
        grad_in[max_indices_[i]] += grad_out[i];
    }

    return grad_input;
}

} // namespace nn
