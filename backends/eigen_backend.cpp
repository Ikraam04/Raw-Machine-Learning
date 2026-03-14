#include "eigen_backend.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace nn {

// memory

float* EigenBackend::allocate(size_t size) {
    return new float[size];
}

void EigenBackend::deallocate(float* ptr) {
    delete[] ptr;
}

void EigenBackend::copy(float* dst, const float* src, size_t size) {
    std::copy(src, src + size, dst);
}

void EigenBackend::upload(float* dst, const float* src_host, size_t size) {
    std::copy(src_host, src_host + size, dst);
}

void EigenBackend::download(float* dst_host, const float* src, size_t size) {
    std::copy(src, src + size, dst_host);
}

void EigenBackend::fill(float* data, float value, size_t size) {
    std::fill(data, data + size, value);
}

// matrix ops

void EigenBackend::matmul(float* result,
                          const float* A, size_t A_rows, size_t A_cols,
                          const float* B, size_t B_rows, size_t B_cols) {
    // wrap raw pointers as Eigen matrices (no copy), RowMajor = row-by-row storage
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_A(A, A_rows, A_cols);
    
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_B(B, B_rows, B_cols);
    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_result(result, A_rows, B_cols);
    
    mat_result = mat_A * mat_B;
}

void EigenBackend::transpose(float* result,
                             const float* A, size_t rows, size_t cols) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_A(A, rows, cols);
    
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat_result(result, cols, rows);
    
    mat_result = mat_A.transpose();
}

// element-wise ops

void EigenBackend::add(float* result,
                       const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] + B[i];
    }
}

// element-wise multiplication
void EigenBackend::multiply(float* result,
                            const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * B[i];
    }
}

void EigenBackend::scale(float* result,
                         const float* A, float scalar, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * scalar;
    }
}

// activations

void EigenBackend::relu(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::max(0.0f, A[i]);
    }
}

void EigenBackend::relu_derivative(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] > 0.0f ? 1.0f : 0.0f;
    }
}

void EigenBackend::sigmoid(float* result, const float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-A[i]));
    }
}

void EigenBackend::sigmoid_derivative(float* result, const float* A, size_t size) {
    // A[i] is already sigmoid(x), not the raw input
    for (size_t i = 0; i < size; ++i) {
        result[i] = A[i] * (1.0f - A[i]);
    }
}

// conv ops
// im2col and col2im are used by the Conv2D layer to transform convolution into matrix multiplication
// does this by unrolling each input patch into a row of the "col" matrix, so that the convolution
// can be done as a single matmul with the kernel weights
void EigenBackend::im2col(const float* input, float* col,
                          int batch, int in_channels, int height, int width,
                          int kernel_h, int kernel_w, int out_h, int out_w,
                          int pad_h, int pad_w, int stride_h, int stride_w) {
    //most times they are square, but we allow rectangular kernels and strides for generality
    // col layout: (batch*out_h*out_w, in_channels*kernel_h*kernel_w)
    // each row is one output position's receptive field, flattened across channels
    int col_cols = in_channels * kernel_h * kernel_w;

    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int col_row = b * out_h * out_w + oh * out_w + ow;

                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;

                            // corresponding input position
                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;

                            float val = 0.0f;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                val = input[b * in_channels * height * width
                                            + c * height * width
                                            + ih * width + iw];
                            }
                            col[col_row * col_cols + col_col] = val;
                        }
                    }
                }
            }
        }
    }
}

void EigenBackend::col2im(const float* col, float* input,
                          int batch, int in_channels, int height, int width,
                          int kernel_h, int kernel_w, int out_h, int out_w,
                          int pad_h, int pad_w, int stride_h, int stride_w) {
    // reverse of im2col: scatter-add col values back into input
    // input must be zeroed before calling
    int col_cols = in_channels * kernel_h * kernel_w;

    for (int b = 0; b < batch; ++b) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int col_row = b * out_h * out_w + oh * out_w + ow;

                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;

                            int ih = oh * stride_h + kh - pad_h;
                            int iw = ow * stride_w + kw - pad_w;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                input[b * in_channels * height * width
                                      + c * height * width
                                      + ih * width + iw] += col[col_row * col_cols + col_col];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace nn