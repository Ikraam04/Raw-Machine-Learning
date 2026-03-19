#pragma once
#include <cstddef>

namespace nn {

// abstract backend interface - EigenBackend and CudaBackend both implement this
class Backend {
public:
    virtual ~Backend() = default;

    // memory management

    // alloc 'size' floats, returns ptr (cpu or gpu depending on backend)
    virtual float* allocate(size_t size) = 0;

    // free memory allocated by this backend
    virtual void deallocate(float* ptr) = 0;

    // copy 'size' floats from src to dst, both ptrs must be from this backend
    virtual void copy(float* dst, const float* src, size_t size) = 0;

    // transfer between cpu and backend memory
    // for Eigen these are just copies; for CUDA these cross the host/device boundary
    virtual void upload(float* dst, const float* src_host, size_t size) = 0;
    virtual void download(float* dst_host, const float* src, size_t size) = 0;

    // fill array with a constant value
    virtual void fill(float* data, float value, size_t size) = 0;

    // matrix ops

    // result = A * B, A is (A_rows x A_cols), B is (B_rows x B_cols)
    // result pre-allocated as (A_rows x B_cols), requires A_cols == B_rows
    virtual void matmul(float* result,
                       const float* A, size_t A_rows, size_t A_cols,
                       const float* B, size_t B_rows, size_t B_cols) = 0;

    // result = A^T, A is (rows x cols), result is (cols x rows)
    virtual void transpose(float* result,
                          const float* A, size_t rows, size_t cols) = 0;

    // element-wise ops

    // result[i] = A[i] + B[i]
    virtual void add(float* result,
                    const float* A, const float* B, size_t size) = 0;

    // result[i] = A[i] * B[i]
    virtual void multiply(float* result,
                         const float* A, const float* B, size_t size) = 0;

    // result[i] = A[i] * scalar
    virtual void scale(float* result,
                      const float* A, float scalar, size_t size) = 0;

    // activations

    // result[i] = max(0, A[i])
    virtual void relu(float* result, const float* A, size_t size) = 0;

    // result[i] = A[i] > 0 ? 1 : 0
    virtual void relu_derivative(float* result, const float* A, size_t size) = 0;

    // result[i] = 1 / (1 + exp(-A[i]))
    virtual void sigmoid(float* result, const float* A, size_t size) = 0;

    // result[i] = sigmoid(A[i]) * (1 - sigmoid(A[i])), assumes A[i] is already sigmoid output
    virtual void sigmoid_derivative(float* result, const float* A, size_t size) = 0;

    // conv ops (used by Conv2D layer)

    // im2col: unrolls input receptive fields into a matrix so conv becomes matmul
    // input:  (batch, in_channels, height, width) flat NCHW
    // col:    (batch*out_h*out_w, in_channels*kernel_h*kernel_w) — caller pre-allocates
    // out-of-bounds positions (from padding) are filled with 0
    virtual void im2col(const float* input, float* col,
                        int batch, int in_channels, int height, int width,
                        int kernel_h, int kernel_w, int out_h, int out_w,
                        int pad_h, int pad_w, int stride_h, int stride_w) = 0;

    // col2im: accumulates col back into input (reverse of im2col, used in backward)
    // input must be zeroed before calling — this function *adds* into it
    virtual void col2im(const float* col, float* input,
                        int batch, int in_channels, int height, int width,
                        int kernel_h, int kernel_w, int out_h, int out_w,
                        int pad_h, int pad_w, int stride_h, int stride_w) = 0;

    // broadcast ops

    // data[i * cols + j] += bias[j] for all i in [0, rows)
    virtual void bias_add(float* data, const float* bias,
                          size_t rows, size_t cols) = 0;

    // output[j] = sum over i of input[i * cols + j]  (reduce rows, keep cols)
    // output must be pre-zeroed by the caller
    virtual void sum_rows(float* output, const float* input,
                          size_t rows, size_t cols) = 0;

    // optimizer ops

    // in-place Adam update for a single parameter array
    // param, grad, m, v are all the same 'size'
    // bc1 = 1 - beta1^t, bc2 = 1 - beta2^t  (precomputed by the caller)
    virtual void adam_update(float* param, const float* grad,
                             float* m, float* v,
                             float lr, float beta1, float beta2,
                             float bc1, float bc2, float eps,
                             size_t size) = 0;

    // layout permutation ops

    // NHWC → NCHW:  src[n, h, w, c] → dst[n, c, h, w]
    virtual void nhwc_to_nchw(const float* src, float* dst,
                               int batch, int channels, int h, int w) = 0;

    // NCHW → NHWC:  src[n, c, h, w] → dst[n, h, w, c]
    virtual void nchw_to_nhwc(const float* src, float* dst,
                               int batch, int channels, int h, int w) = 0;

    // pooling ops

    // max pool forward: input {batch, channels, in_h, in_w} → output {batch, channels, out_h, out_w}
    // indices[i] = flat index into input for each output element (for backward routing)
    virtual void maxpool_forward(const float* input, float* output, int* indices,
                                  int batch, int channels,
                                  int in_h, int in_w,
                                  int out_h, int out_w,
                                  int pool_h, int pool_w,
                                  int stride_h, int stride_w) = 0;

    // max pool backward: routes grad_output to the input positions recorded in indices
    // grad_input must be pre-zeroed
    virtual void maxpool_backward(const float* grad_output, float* grad_input,
                                   const int* indices,
                                   int output_size, int input_size) = 0;

    // global average pooling: input {batch, channels, h, w} → output {batch, channels}
    virtual void global_avg_pool_forward(const float* input, float* output,
                                          int batch, int channels, int h, int w) = 0;

    // global average pooling backward: output {batch, channels} → input {batch, channels, h, w}
    virtual void global_avg_pool_backward(const float* grad_output, float* grad_input,
                                           int batch, int channels, int h, int w) = 0;

    // softmax forward: row-wise softmax with numerical stability (subtract max)
    // input/output: {rows, cols}, each row is independently softmaxed
    virtual void softmax_forward(const float* input, float* output,
                                  size_t rows, size_t cols) = 0;

    // softmax backward: Jacobian-vector product for softmax
    // softmax_output: cached forward output, grad_output: incoming gradient
    virtual void softmax_backward(const float* softmax_output,
                                   const float* grad_output,
                                   float* grad_input,
                                   size_t rows, size_t cols) = 0;

    // integer memory (used for pooling indices — needs device alloc for CUDA)
    virtual int* allocate_int(size_t size) = 0;
    virtual void deallocate_int(int* ptr) = 0;
};

} // namespace nn