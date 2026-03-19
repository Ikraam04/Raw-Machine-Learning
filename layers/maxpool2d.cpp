#include "maxpool2d.h"
#include <stdexcept>

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

    // (re)allocate index buffer if needed
    size_t out_size = (size_t)(batch * channels * out_h * out_w);
    if (out_size != indices_size_) {
        if (max_indices_) backend_->deallocate_int(max_indices_);
        max_indices_ = backend_->allocate_int(out_size);
        indices_size_ = out_size;
    }

    Tensor output({(size_t)batch, (size_t)channels,
                   (size_t)out_h, (size_t)out_w}, backend_);

    backend_->maxpool_forward(input.data(), output.data(), max_indices_,
                               batch, channels, in_h, in_w, out_h, out_w,
                               pool_h_, pool_w_, stride_h_, stride_w_);

    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    int input_size = cached_batch_ * cached_channels_ * cached_in_h_ * cached_in_w_;
    int output_size = cached_batch_ * cached_channels_ * cached_out_h_ * cached_out_w_;

    Tensor grad_input({(size_t)cached_batch_,    (size_t)cached_channels_,
                       (size_t)cached_in_h_,     (size_t)cached_in_w_}, backend_);
    grad_input.fill(0.0f);

    backend_->maxpool_backward(grad_output.data(), grad_input.data(),
                                max_indices_, output_size, input_size);

    return grad_input;
}

} // namespace nn
