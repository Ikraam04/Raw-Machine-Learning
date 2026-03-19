#include "conv2d.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace nn {

Conv2D::Conv2D(int in_channels, int out_channels,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               BackendPtr backend)
    : Layer(backend),
      in_channels_(in_channels), out_channels_(out_channels),
      kernel_h_(kernel_h), kernel_w_(kernel_w),
      stride_h_(stride_h), stride_w_(stride_w),
      pad_h_(pad_h), pad_w_(pad_w),
      weights_({(size_t)(in_channels * kernel_h * kernel_w), (size_t)out_channels}, backend),
      bias_({1, (size_t)out_channels}, backend),
      grad_weights_({(size_t)(in_channels * kernel_h * kernel_w), (size_t)out_channels}, backend),
      grad_bias_({1, (size_t)out_channels}, backend),
      m_weights_({(size_t)(in_channels * kernel_h * kernel_w), (size_t)out_channels}, backend),
      v_weights_({(size_t)(in_channels * kernel_h * kernel_w), (size_t)out_channels}, backend),
      m_bias_({1, (size_t)out_channels}, backend),
      v_bias_({1, (size_t)out_channels}, backend),
      col_cache_({1, 1}, backend)  // resized on first forward
{
    initialize_weights(); // initialize weights and bias
    // zero out gradients and Adam moments
    grad_weights_.fill(0.0f);
    grad_bias_.fill(0.0f);
    m_weights_.fill(0.0f);
    v_weights_.fill(0.0f);
    m_bias_.fill(0.0f);
    v_bias_.fill(0.0f);
}

// ── forward ──────────────────────────────────────────────────────────────────

Tensor Conv2D::forward(const Tensor& input) {
    // input: {batch, in_channels, H, W}
    if (input.shape().size() != 4) {
        throw std::runtime_error("Conv2D::forward expects a 4D tensor {batch, channels, H, W}");
    }
    if ((int)input.shape()[1] != in_channels_) {
        throw std::runtime_error("Conv2D::forward: input channel mismatch");
    }

    int batch = (int)input.shape()[0];
    int in_h  = (int)input.shape()[2];
    int in_w  = (int)input.shape()[3];

    int out_h = (in_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    int out_w = (in_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;

    // cache spatial dims for backward
    cached_batch_ = batch;
    cached_in_h_  = in_h;  cached_in_w_  = in_w;
    cached_out_h_ = out_h; cached_out_w_ = out_w;

    // im2col → col: {batch*out_h*out_w, in_channels*kernel_h*kernel_w}
    size_t col_rows = (size_t)(batch * out_h * out_w);
    size_t col_cols = (size_t)(in_channels_ * kernel_h_ * kernel_w_);
    col_cache_ = Tensor({col_rows, col_cols}, backend_);

    backend_->im2col(input.data(), col_cache_.data(),
                     batch, in_channels_, in_h, in_w,
                     kernel_h_, kernel_w_, out_h, out_w,
                     pad_h_, pad_w_, stride_h_, stride_w_);

    // matmul: col × weights_ → output_nhwc {batch*out_h*out_w, out_channels}
    // col:      {col_rows, col_cols}
    // weights_: {col_cols, out_channels}
    // result:   {col_rows, out_channels}  — data order is NHWC
    size_t out_ch = (size_t)out_channels_;
    Tensor output_nhwc({col_rows, out_ch}, backend_);

    backend_->matmul(output_nhwc.data(),
                     col_cache_.data(), col_rows, col_cols,
                     weights_.data(),   col_cols, out_ch);

    // add bias to every output position
    backend_->bias_add(output_nhwc.data(), bias_.data(), col_rows, out_ch);

    // permute NHWC → NCHW for the output tensor
    Tensor output({(size_t)batch, out_ch, (size_t)out_h, (size_t)out_w}, backend_);
    backend_->nhwc_to_nchw(output_nhwc.data(), output.data(), batch, out_channels_, out_h, out_w);

    return output;
}

// ── backward ─────────────────────────────────────────────────────────────────

Tensor Conv2D::backward(const Tensor& grad_output) {
    // grad_output: {batch, out_channels, out_h, out_w}  NCHW
    int batch = cached_batch_;
    int in_h  = cached_in_h_,  in_w  = cached_in_w_;
    int out_h = cached_out_h_, out_w = cached_out_w_;

    size_t col_rows = (size_t)(batch * out_h * out_w);
    size_t col_cols = (size_t)(in_channels_ * kernel_h_ * kernel_w_);
    size_t out_ch   = (size_t)out_channels_;

    // permute grad NCHW → NHWC so shapes align with col_cache_
    Tensor grad_nhwc({col_rows, out_ch}, backend_);
    backend_->nchw_to_nhwc(grad_output.data(), grad_nhwc.data(), batch, out_channels_, out_h, out_w);

    // grad_weights = col^T × grad_nhwc  →  {col_cols, out_channels}
    // col^T:      {col_cols, col_rows}
    // grad_nhwc:  {col_rows, out_ch}
    // result:     {col_cols, out_ch}
    Tensor col_T({col_cols, col_rows}, backend_);
    backend_->transpose(col_T.data(), col_cache_.data(), col_rows, col_cols);
    backend_->matmul(grad_weights_.data(),
                     col_T.data(),     col_cols, col_rows,
                     grad_nhwc.data(), col_rows, out_ch);

    // grad_bias = sum of grad_nhwc over the spatial/batch dimension
    grad_bias_.fill(0.0f);
    backend_->sum_rows(grad_bias_.data(), grad_nhwc.data(), col_rows, out_ch);

    // grad_col = grad_nhwc × weights^T  →  {col_rows, col_cols}
    // grad_nhwc:  {col_rows, out_ch}
    // weights^T:  {out_ch, col_cols}
    Tensor weights_T({out_ch, col_cols}, backend_);
    backend_->transpose(weights_T.data(), weights_.data(), col_cols, out_ch);

    Tensor grad_col({col_rows, col_cols}, backend_);
    backend_->matmul(grad_col.data(),
                     grad_nhwc.data(), col_rows, out_ch,
                     weights_T.data(), out_ch,   col_cols);

    // grad_input = col2im(grad_col)  →  {batch, in_channels, in_h, in_w}
    Tensor grad_input({(size_t)batch, (size_t)in_channels_,
                       (size_t)in_h,  (size_t)in_w}, backend_);
    grad_input.fill(0.0f);  // col2im accumulates, so must start at 0

    backend_->col2im(grad_col.data(), grad_input.data(),
                     batch, in_channels_, in_h, in_w,
                     kernel_h_, kernel_w_, out_h, out_w,
                     pad_h_, pad_w_, stride_h_, stride_w_);

    return grad_input;
}

// ── update ────────────────────────────────────────────────────────────────────

void Conv2D::update_parameters(float lr) {
    ++t_;

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps   = 1e-8f;

    float bc1 = 1.0f - std::pow(beta1, (float)t_);
    float bc2 = 1.0f - std::pow(beta2, (float)t_);

    size_t w_size = (size_t)(in_channels_ * kernel_h_ * kernel_w_ * out_channels_);

    // weights
    backend_->adam_update(weights_.data(), grad_weights_.data(),
                          m_weights_.data(), v_weights_.data(),
                          lr, beta1, beta2, bc1, bc2, eps, w_size);

    // bias
    backend_->adam_update(bias_.data(), grad_bias_.data(),
                          m_bias_.data(), v_bias_.data(),
                          lr, beta1, beta2, bc1, bc2, eps,
                          (size_t)out_channels_);
}

// ── helpers ───────────────────────────────────────────────────────────────────

void Conv2D::initialize_weights() {
    // Xavier/Glorot: fan_in = in_channels*kh*kw, fan_out = out_channels*kh*kw
    float fan_in  = (float)(in_channels_  * kernel_h_ * kernel_w_);
    float fan_out = (float)(out_channels_ * kernel_h_ * kernel_w_);
    float limit = std::sqrt(6.0f / (fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    size_t w_size = (size_t)(in_channels_ * kernel_h_ * kernel_w_ * out_channels_);
    std::vector<float> host_weights(w_size);
    for (size_t i = 0; i < w_size; ++i) {
        host_weights[i] = dist(gen);
    }
    weights_.set_data(host_weights);

    std::vector<float> host_bias(out_channels_, 0.01f);
    bias_.set_data(host_bias);
}

} // namespace nn
