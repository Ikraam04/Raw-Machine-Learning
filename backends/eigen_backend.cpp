#include "eigen_backend.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

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

// broadcast ops

void EigenBackend::bias_add(float* data, const float* bias,
                             size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i * cols + j] += bias[j];
        }
    }
}

void EigenBackend::sum_rows(float* output, const float* input,
                             size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            output[j] += input[i * cols + j];
        }
    }
}

// optimizer ops

void EigenBackend::adam_update(float* param, const float* grad,
                                float* m, float* v,
                                float lr, float beta1, float beta2,
                                float bc1, float bc2, float eps,
                                size_t size) {
    for (size_t i = 0; i < size; ++i) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

// layout permutation

void EigenBackend::nhwc_to_nchw(const float* src, float* dst,
                                  int batch, int channels, int h, int w) {
    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < h; ++oh) {
            for (int ow = 0; ow < w; ++ow) {
                for (int c = 0; c < channels; ++c) {
                    int src_idx = (n * h * w + oh * w + ow) * channels + c;
                    int dst_idx = n * channels * h * w + c * h * w + oh * w + ow;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void EigenBackend::nchw_to_nhwc(const float* src, float* dst,
                                  int batch, int channels, int h, int w) {
    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < h; ++oh) {
            for (int ow = 0; ow < w; ++ow) {
                for (int c = 0; c < channels; ++c) {
                    int src_idx = n * channels * h * w + c * h * w + oh * w + ow;
                    int dst_idx = (n * h * w + oh * w + ow) * channels + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// pooling ops

void EigenBackend::maxpool_forward(const float* input, float* output, int* indices,
                                    int batch, int channels,
                                    int in_h, int in_w,
                                    int out_h, int out_w,
                                    int pool_h, int pool_w,
                                    int stride_h, int stride_w) {
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;

                    for (int ph = 0; ph < pool_h; ++ph) {
                        for (int pw = 0; pw < pool_w; ++pw) {
                            int ih = oh * stride_h + ph;
                            int iw = ow * stride_w + pw;
                            int flat_in = n * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + ih * in_w + iw;

                            if (input[flat_in] > max_val) {
                                max_val = input[flat_in];
                                max_idx = flat_in;
                            }
                        }
                    }

                    int flat_out = n * channels * out_h * out_w
                                 + c * out_h * out_w
                                 + oh * out_w + ow;
                    output[flat_out] = max_val;
                    indices[flat_out] = max_idx;
                }
            }
        }
    }
}

void EigenBackend::maxpool_backward(const float* grad_output, float* grad_input,
                                     const int* indices,
                                     int output_size, int /*input_size*/) {
    for (int i = 0; i < output_size; ++i) {
        grad_input[indices[i]] += grad_output[i];
    }
}

// global average pooling

void EigenBackend::global_avg_pool_forward(const float* input, float* output,
                                            int batch, int channels, int h, int w) {
    float inv_hw = 1.0f / (float)(h * w);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            const float* fm = input + n * channels * h * w + c * h * w;
            for (int i = 0; i < h * w; ++i) {
                sum += fm[i];
            }
            output[n * channels + c] = sum * inv_hw;
        }
    }
}

void EigenBackend::global_avg_pool_backward(const float* grad_output, float* grad_input,
                                             int batch, int channels, int h, int w) {
    float inv_hw = 1.0f / (float)(h * w);
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            float g = grad_output[n * channels + c] * inv_hw;
            float* fm = grad_input + n * channels * h * w + c * h * w;
            for (int i = 0; i < h * w; ++i) {
                fm[i] = g;
            }
        }
    }
}

// softmax

void EigenBackend::softmax_forward(const float* input, float* output,
                                    size_t rows, size_t cols) {
    for (size_t b = 0; b < rows; ++b) {
        const float* in_row = input + b * cols;
        float* out_row = output + b * cols;

        float max_val = in_row[0];
        for (size_t i = 1; i < cols; ++i) {
            if (in_row[i] > max_val) max_val = in_row[i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < cols; ++i) {
            out_row[i] = std::exp(in_row[i] - max_val);
            sum += out_row[i];
        }

        for (size_t i = 0; i < cols; ++i) {
            out_row[i] /= sum;
        }
    }
}

void EigenBackend::softmax_backward(const float* softmax_output,
                                     const float* grad_output,
                                     float* grad_input,
                                     size_t rows, size_t cols) {
    for (size_t b = 0; b < rows; ++b) {
        const float* s = softmax_output + b * cols;
        const float* g_out = grad_output + b * cols;
        float* g_in = grad_input + b * cols;

        for (size_t i = 0; i < cols; ++i) {
            g_in[i] = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                float delta = (i == j) ? 1.0f : 0.0f;
                g_in[i] += s[i] * (delta - s[j]) * g_out[j];
            }
        }
    }
}

// integer memory

int* EigenBackend::allocate_int(size_t size) {
    return new int[size];
}

void EigenBackend::deallocate_int(int* ptr) {
    delete[] ptr;
}

} // namespace nn