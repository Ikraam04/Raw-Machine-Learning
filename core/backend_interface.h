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
};

} // namespace nn