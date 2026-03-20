#pragma once
#include "core/backend_interface.h"

namespace nn {

// cpu backend using Eigen - all memory in ram, everything on cpu
class EigenBackend : public Backend {
public:
    EigenBackend() = default;
    ~EigenBackend() override = default;

    // memory
    float* allocate(size_t size) override;
    void deallocate(float* ptr) override;
    void copy(float* dst, const float* src, size_t size) override;
    void upload(float* dst, const float* src_host, size_t size) override;
    void download(float* dst_host, const float* src, size_t size) override;
    void fill(float* data, float value, size_t size) override;

    // matrix ops
    void matmul(float* result,
               const float* A, size_t A_rows, size_t A_cols,
               const float* B, size_t B_rows, size_t B_cols) override;

    void transpose(float* result,
                  const float* A, size_t rows, size_t cols) override;

    void add(float* result,
            const float* A, const float* B, size_t size) override;
 
    // element-wise
    void multiply(float* result,
                 const float* A, const float* B, size_t size) override;

    void scale(float* result,
              const float* A, float scalar, size_t size) override;

    // activations
    void relu(float* result, const float* A, size_t size) override;
    void relu_derivative(float* result, const float* A, size_t size) override;
    void sigmoid(float* result, const float* A, size_t size) override;
    void sigmoid_derivative(float* result, const float* A, size_t size) override;

    // conv ops
    void im2col(const float* input, float* col,
                int batch, int in_channels, int height, int width,
                int kernel_h, int kernel_w, int out_h, int out_w,
                int pad_h, int pad_w, int stride_h, int stride_w) override;

    void col2im(const float* col, float* input,
                int batch, int in_channels, int height, int width,
                int kernel_h, int kernel_w, int out_h, int out_w,
                int pad_h, int pad_w, int stride_h, int stride_w) override;

    // broadcast ops
    void bias_add(float* data, const float* bias,
                  size_t rows, size_t cols) override;
    void sum_rows(float* output, const float* input,
                  size_t rows, size_t cols) override;

    // optimizer ops
    void adam_update(float* param, const float* grad,
                     float* m, float* v,
                     float lr, float beta1, float beta2,
                     float bc1, float bc2, float eps,
                     size_t size) override;

    // layout permutation
    void nhwc_to_nchw(const float* src, float* dst,
                       int batch, int channels, int h, int w) override;
    void nchw_to_nhwc(const float* src, float* dst,
                       int batch, int channels, int h, int w) override;

    // pooling ops
    void maxpool_forward(const float* input, float* output, int* indices,
                          int batch, int channels,
                          int in_h, int in_w,
                          int out_h, int out_w,
                          int pool_h, int pool_w,
                          int stride_h, int stride_w) override;
    void maxpool_backward(const float* grad_output, float* grad_input,
                           const int* indices,
                           int output_size, int input_size) override;

    // global average pooling
    void global_avg_pool_forward(const float* input, float* output,
                                  int batch, int channels, int h, int w) override;
    void global_avg_pool_backward(const float* grad_output, float* grad_input,
                                   int batch, int channels, int h, int w) override;

    // softmax
    void softmax_forward(const float* input, float* output,
                          size_t rows, size_t cols) override;
    void softmax_backward(const float* softmax_output,
                           const float* grad_output,
                           float* grad_input,
                           size_t rows, size_t cols) override;

    // integer memory
    int* allocate_int(size_t size) override;
    void deallocate_int(int* ptr) override;

    // fused loss
    void softmax_cross_entropy(const float* logits, const float* targets,
                               float* grad, float* loss_out,
                               int batch, int num_classes) override;
};

} // namespace nn