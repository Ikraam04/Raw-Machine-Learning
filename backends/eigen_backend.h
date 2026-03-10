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
    void fill(float* data, float value, size_t size) override;

    // matrix ops
    void matmul(float* result,
               const float* A, size_t A_rows, size_t A_cols,
               const float* B, size_t B_rows, size_t B_cols) override;

    void transpose(float* result,
                  const float* A, size_t rows, size_t cols) override;

    // element-wise
    void add(float* result,
            const float* A, const float* B, size_t size) override;

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
};

} // namespace nn