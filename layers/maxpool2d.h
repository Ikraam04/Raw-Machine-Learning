#pragma once
#include "layer.h"
#include <vector>
#include <limits>

namespace nn {

// 2D max pooling — no learnable parameters
//
// slides a pool_h×pool_w window over each feature map and keeps the max value
// standard use: MaxPool2D(2, 2) halves both spatial dims
//
// tensor layout: NCHW  (batch, channels, height, width)
//   input:  {batch, channels, H,     W    }
//   output: {batch, channels, out_h, out_w}
//
// out_h = (H - pool_h) / stride_h + 1
// out_w = (W - pool_w) / stride_w + 1
//
// backward: gradient is routed only to the position that held the max
//           (all other positions in the window get 0)
class MaxPool2D : public Layer {
public:
    MaxPool2D(int pool_h, int pool_w,
              int stride_h, int stride_w,
              BackendPtr backend)
        : Layer(backend),
          pool_h_(pool_h), pool_w_(pool_w),
          stride_h_(stride_h), stride_w_(stride_w) {}

    // convenience: square pool + matching stride (the common case)
    MaxPool2D(int pool_size, BackendPtr backend)
        : MaxPool2D(pool_size, pool_size, pool_size, pool_size, backend) {}

    ~MaxPool2D() override {
        if (max_indices_) backend_->deallocate_int(max_indices_);
    }

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_parameters(float /*learning_rate*/) override {}

private:
    int pool_h_, pool_w_;
    int stride_h_, stride_w_;

    // cached from forward for backward pass
    // max_indices_[i] = flat index into the input tensor for the i-th output element
    int* max_indices_ = nullptr;
    size_t indices_size_ = 0;
    int cached_batch_ = 0;
    int cached_channels_ = 0;
    int cached_in_h_ = 0, cached_in_w_ = 0;
    int cached_out_h_ = 0, cached_out_w_ = 0;
};

} // namespace nn
