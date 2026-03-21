#include "cuda_backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstdio>  // For fprintf and stderr
#include <cstdlib> // For exit

//  macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0) //prevent dangling else

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d — %d\n", \
                    __FILE__, __LINE__, (int)err); \
            exit(1); \
        } \
    } while(0)



__global__ void fill_kernel(float* dst, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {  
        dst[idx] = value;
    }
}

__global__ void add_kernel(float* result, const float* A, const float* B, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = A[idx] + B[idx];
    }
}

__global__ void multiply_kernel(float* result, const float* A, const float* B, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = A[idx] * B[idx];
    }
}

__global__ void scale_kernel(float* result, const float* A, float scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = A[idx] * scalar;
    }
}

__global__ void relu_kernel(float* result, const float* A, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = fmaxf(0.0f, A[idx]);
    }
}

// A[i] is the pre-activation value (not the relu output)
__global__ void relu_derivative_kernel(float* result, const float* A, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = A[idx] > 0.0f ? 1.0f : 0.0f;
    }
}

__global__ void sigmoid_kernel(float* result, const float* A, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = 1.0f / (1.0f + expf(-A[idx]));
    }
}

// A[i] is already the sigmoid output s, derivative is s*(1-s)
__global__ void sigmoid_derivative_kernel(float* result, const float* A, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = A[idx] * (1.0f - A[idx]);
    }
}

// CUDA kernel for matrix multiplication: C = A * B
// A: M x K
// B: K x N
// C: M x N
//
// Key difference vs CPU version:
//
// CPU version usually looks like:
// for (i = 0..M)
//   for (j = 0..N)
//     for (k = 0..K)
//       C[i,j] += A[i,k] * B[k,j]
//
// In CUDA we REMOVE the outer two loops (i and j).
// Instead, the GPU launches many threads and each thread
// is responsible for computing ONE element of C.
//
// So conceptually:
// thread(row, col) -> compute C[row, col]
//
__global__ void matmul_kernel(float* result,
                               const float* A, const float* B,
                               int M, int K, int N)
{
    // Determine which output element this thread is responsible for.
    // blockIdx + threadIdx together give the thread's global position
    // in the 2D grid of threads.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Some launched threads may fall outside the matrix bounds
    // (because we usually launch nice round block sizes).
    // Those threads simply do nothing.
    if (row < M && col < N) {

        float sum = 0.0f;

        // This is the ONLY loop left from the CPU version.
        // Each thread walks across K to compute the dot product
        // between row 'row' of A and column 'col' of B.
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Store the computed value into C[row, col]
        // dont assign in the for loop, avoids redundant global memory reads/writes
        result[row * N + col] = sum;
    }
}

// cuda kernel for transpose: result = A^T
// A is (rows x cols), result is (cols x rows)
//
// cpu version would look like:
// for (i = 0..rows)
//   for (j = 0..cols)
//     result[j][i] = A[i][j]
//
// same deal as matmul - we kill both loops by assigning one thread per element.
// thread(row, col) reads from A[row][col] and writes to result[col][row].
// no loop needed at all - each thread does exactly one read and one write.
//
// conceptually:
// thread(row, col) -> result[col][row] = A[row][col]
//
__global__ void transpose_kernel(float* result, const float* A, int rows, int cols)
{
    // figure out which element this thread owns, same 2d indexing as matmul
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // threads outside the matrix just chill and do nothing
    if (row < rows && col < cols) {
        // read from A[row][col] in row-major: index = row * cols + col
        // write to result[col][row] in row-major: index = col * rows + row
        // (result has shape cols x rows, so its "stride" is rows, not cols)
        result[col * rows + row] = A[row * cols + col];
    }
}

// cuda kernel for im2col
//
// cpu version has 6 nested loops:
// for b ... for oh ... for ow ...  -> outer 3: which output position
//   for c ... for kh ... for kw ...  -> inner 3: fill one row of col matrix
//
// we kill the outer 3 loops by assigning one thread per output position.
// each thread is responsible for one (b, oh, ow) triple, and fills
// the entire corresponding row of the col matrix itself (inner 3 stay as loops).
//
// col matrix layout:
//   rows: batch * out_h * out_w   (one row per output position)
//   cols: in_channels * kernel_h * kernel_w  (one col per input value in the receptive field)
//
// conceptually:
// thread(idx) -> owns row idx = (b, oh, ow), fills all kernel_h*kernel_w*in_channels cols
//
__global__ void im2col_kernel(const float* input, float* col,
                               int batch, int in_channels, int height, int width,
                               int kernel_h, int kernel_w, int out_h, int out_w,
                               int pad_h, int pad_w, int stride_h, int stride_w)
{
    // flat thread index - each thread owns one output position (b, oh, ow)
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //1d because we flattened the outer 3 loops into one dimension of threads - easier to manage
    int total = batch * out_h * out_w;

    // outside the output bounds do nothing
    if (idx >= total) return;

    // unpack flat idx back into (b, oh, ow) - same as dividing back out place values
    int b  = idx / (out_h * out_w);
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;

    // how many columns does one row of col have?
    int col_cols = in_channels * kernel_h * kernel_w;

    // this thread fills every column of its row - one value per (c, kh, kw) combo
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {

                // which column of col does this (c, kh, kw) map to?
                int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;

                // which input pixel does this kernel position land on?
                // stride moves the receptive field, padding shifts it inward
                int ih = oh * stride_h + kh - pad_h;
                int iw = ow * stride_w + kw - pad_w;

                // if we're outside the input (padding region), write 0
                float val = 0.0f;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    val = input[b * in_channels * height * width
                                + c * height * width
                                + ih * width + iw];
                }

                col[idx * col_cols + col_col] = val;
            }
        }
    }
}

// cuda kernel for col2im: reverse of im2col, scatters col values back into input
//
// cpu version just does:
// for b, oh, ow, c, kh, kw:
//   input[b][c][ih][iw] += col[row][col_col]
//
// same 1d pattern as im2col - one thread per output position (b, oh, ow).
// each thread reads its row of col and scatters values back to input.
// NOTE: because multiple threads can write to the same input pixel (when receptive fields overlap),
// we have to use atomicAdd to avoid race conditions.
//
// input must be zeroed before calling - this kernel only adds into it
//
__global__ void col2im_kernel(const float* col, float* input,
                               int batch, int in_channels, int height, int width,
                               int kernel_h, int kernel_w, int out_h, int out_w,
                               int pad_h, int pad_w, int stride_h, int stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //same 1d logic here
    int total = batch * out_h * out_w;

    if (idx >= total) return; 

    // unpack flat idx into (b, oh, ow) - same as im2col
    int b  = idx / (out_h * out_w);
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;

    int col_cols = in_channels * kernel_h * kernel_w;

    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {

                int col_col = c * kernel_h * kernel_w + kh * kernel_w + kw;

                int ih = oh * stride_h + kh - pad_h;
                int iw = ow * stride_w + kw - pad_w;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    float val = col[idx * col_cols + col_col];
                    // atomicAdd because multiple threads can map to the same input pixel
                    atomicAdd(&input[b * in_channels * height * width
                                     + c * height * width
                                     + ih * width + iw], val);
                }
            }
        }
    }
}


// bias_add: data[i * cols + j] += bias[j]
// one thread per element in the (rows x cols) matrix
__global__ void bias_add_kernel(float* data, const float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        data[idx] += bias[col];
    }
}

// sum_rows: output[j] = sum_i input[i * cols + j]
// one thread per column — walks down the column summing
__global__ void sum_rows_kernel(float* output, const float* input, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum += input[i * cols + col];
        }
        // output assumed pre-zeroed, but we write directly here
        output[col] += sum;
    }
}

// adam_update: in-place Adam step for each parameter
// one thread per parameter element
__global__ void adam_update_kernel(float* param, const float* grad,
                                    float* m, float* v,
                                    float lr, float beta1, float beta2,
                                    float bc1, float bc2, float eps,
                                    size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bc1;
        float v_hat = v[idx] / bc2;
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// nhwc_to_nchw: src[n, h, w, c] → dst[n, c, h, w]
// one thread per element — figure out (n, h, w, c) from flat NHWC index,
// then write to the NCHW position
__global__ void nhwc_to_nchw_kernel(const float* src, float* dst,
                                      int batch, int channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    if (idx < total) {
        // idx is a flat NHWC index: idx = n*(h*w*c) + row*(w*c) + col*c + c_idx
        int c = idx % channels;
        int tmp = idx / channels;
        int ow = tmp % w;
        tmp /= w;
        int oh = tmp % h;
        int n = tmp / h;

        int dst_idx = n * channels * h * w + c * h * w + oh * w + ow;
        dst[dst_idx] = src[idx];
    }
}

// nchw_to_nhwc: src[n, c, h, w] → dst[n, h, w, c]
// one thread per element — figure out (n, c, h, w) from flat NCHW index,
// then write to the NHWC position
__global__ void nchw_to_nhwc_kernel(const float* src, float* dst,
                                      int batch, int channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    if (idx < total) {
        // idx is a flat NCHW index: idx = n*(c*h*w) + c_idx*(h*w) + row*w + col
        int ow = idx % w;
        int tmp = idx / w;
        int oh = tmp % h;
        tmp /= h;
        int c = tmp % channels;
        int n = tmp / channels;

        int dst_idx = (n * h * w + oh * w + ow) * channels + c;
        dst[dst_idx] = src[idx];
    }
}

// maxpool_forward: one thread per output element
// scans the pool_h × pool_w window to find the max and its flat input index
// drops the first 4 loops of the CPU version and replaces with flat indexing to assign one thread per output element
__global__ void maxpool_forward_kernel(const float* input, float* output, int* indices,
                                        int batch, int channels,
                                        int in_h, int in_w,
                                        int out_h, int out_w,
                                        int pool_h, int pool_w,
                                        int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx < total) {
        // unpack flat output index into (n, c, oh, ow)
        int ow = idx % out_w;
        int tmp = idx / out_w; //
        int oh = tmp % out_h;
        tmp /= out_h;
        int c = tmp % channels;
        int n = tmp / channels;

        float max_val = -1e30f;
        int max_idx = -1;

        for (int ph = 0; ph < pool_h; ++ph) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int ih = oh * stride_h + ph;
                int iw = ow * stride_w + pw;
                int flat_in = n * channels * in_h * in_w
                            + c * in_h * in_w
                            + ih * in_w + iw;
                float val = input[flat_in];
                if (val > max_val) {
                    max_val = val;
                    max_idx = flat_in;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// maxpool_backward: one thread per output element
// routes gradient to the input position recorded in indices
// uses atomicAdd because multiple output positions could (in theory) map to same input
__global__ void maxpool_backward_kernel(const float* grad_output, float* grad_input,
                                         const int* indices, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        atomicAdd(&grad_input[indices[idx]], grad_output[idx]);
    }
}

// global_avg_pool_forward: one thread per (n, c) pair
// averages all h*w spatial values for that feature map
__global__ void global_avg_pool_forward_kernel(const float* input, float* output,
                                                int batch, int channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx < total) {
        int hw = h * w;
        const float* fm = input + idx * hw;
        float sum = 0.0f;
        for (int i = 0; i < hw; ++i) {
            sum += fm[i];
        }
        output[idx] = sum / (float)hw;
    }
}

// global_avg_pool_backward: one thread per element in grad_input
// each spatial position gets grad_output[n,c] / (h*w)
__global__ void global_avg_pool_backward_kernel(const float* grad_output, float* grad_input,
                                                 int batch, int channels, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * h * w;
    if (idx < total) {
        int hw = h * w;
        // which (n, c) does this element belong to?
        int nc = idx / hw;
        float g = grad_output[nc] / (float)hw;
        grad_input[idx] = g;
    }
}

// softmax_cross_entropy_kernel: one thread per batch item
// computes softmax, accumulates cross-entropy loss via atomicAdd, writes gradient
// drop the outer loop over batch and assign one thread per batch item, so each thread walks across its row to do the computation
__global__ void softmax_cross_entropy_kernel(const float* logits, const float* targets,
                                              float* grad, float* loss_out,
                                              int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    const float* logit_row  = logits  + b * num_classes;
    const float* target_row = targets + b * num_classes;
    float*       grad_row   = grad    + b * num_classes;

    // softmax with stability: subtract max before exp
    float max_val = logit_row[0];
    for (int c = 1; c < num_classes; ++c)
        if (logit_row[c] > max_val) max_val = logit_row[c];

    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        grad_row[c] = expf(logit_row[c] - max_val);  // reuse grad as temp buffer
        sum_exp += grad_row[c];
    }

    float loss = 0.0f;
    const float epsilon = 1e-7f;
    for (int c = 0; c < num_classes; ++c) {
        float s = grad_row[c] / sum_exp;
        if (target_row[c] > 0.5f) {
            float pred = fmaxf(epsilon, fminf(1.0f - epsilon, s));
            loss -= logf(pred);
        }
        grad_row[c] = (s - target_row[c]) / batch;
    }

    // accumulate scalar loss — atomicAdd because all threads write to same location
    atomicAdd(loss_out, loss / batch);
}

// softmax_forward: one thread per row
// each thread computes softmax for its entire row (subtract max, exp, normalize)
// this is fine for small num_classes (e.g. 10 for MNIST) but not for large ones.
// works by killing the outer loop over rows and assigning one thread per row, so each thread walks across its row to do the computation
__global__ void softmax_forward_kernel(const float* input, float* output,
                                        int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const float* in_row = input + row * cols;
        float* out_row = output + row * cols;

        // find max for numerical stability
        float max_val = in_row[0];
        for (int i = 1; i < cols; ++i) {
            if (in_row[i] > max_val) max_val = in_row[i];
        }

        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            out_row[i] = expf(in_row[i] - max_val);
            sum += out_row[i];
        }

        // normalize
        for (int i = 0; i < cols; ++i) {
            out_row[i] /= sum;
        }
    }
}

// softmax_backward: one thread per row
// computes the Jacobian-vector product: grad_in[i] = sum_j s[i]*(d_ij - s[j])*g[j]
__global__ void softmax_backward_kernel(const float* softmax_output,
                                         const float* grad_output,
                                         float* grad_input,
                                         int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        const float* s = softmax_output + row * cols;
        const float* g_out = grad_output + row * cols;
        float* g_in = grad_input + row * cols;

        for (int i = 0; i < cols; ++i) {
            float val = 0.0f;
            for (int j = 0; j < cols; ++j) {
                float delta = (i == j) ? 1.0f : 0.0f;
                val += s[i] * (delta - s[j]) * g_out[j];
            }
            g_in[i] = val;
        }
    }
}

namespace nn
{

CudaBackend::CudaBackend() : cublas_handle_(nullptr) {}


CudaBackend::~CudaBackend() {
    cublasDestroy(cublas_handle_);
}

float* CudaBackend::allocate(size_t size) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
    return ptr;
}

void CudaBackend::deallocate(float* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}
void CudaBackend::copy(float* dst, const float* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToDevice));
}

// upload from host to device
void CudaBackend::upload(float* dst, const float* src_host, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src_host, size * sizeof(float), cudaMemcpyHostToDevice));
}
// download from device to host
void CudaBackend::download(float* dst_host, const float* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst_host, src, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void CudaBackend::fill(float* dst, float value, size_t size) {
    fill_kernel<<<(size + 255) / 256, 256>>>(dst, value, size);
}

void CudaBackend::add(float* result, const float* A, const float* B, size_t size) {
    add_kernel<<<(size + 255) / 256, 256>>>(result, A, B, size);
}

void CudaBackend::multiply(float* result, const float* A, const float* B, size_t size) {
    multiply_kernel<<<(size + 255) / 256, 256>>>(result, A, B, size);
}

void CudaBackend::scale(float* result, const float* A, float scalar, size_t size) {
    scale_kernel<<<(size + 255) / 256, 256>>>(result, A, scalar, size);
}

void CudaBackend::relu(float* result, const float* A, size_t size) {
    relu_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
}

void CudaBackend::relu_derivative(float* result, const float* A, size_t size) {
    relu_derivative_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
}

void CudaBackend::sigmoid(float* result, const float* A, size_t size) {
    sigmoid_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
}

void CudaBackend::sigmoid_derivative(float* result, const float* A, size_t size) {
    sigmoid_derivative_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
}

void CudaBackend::matmul(float* result,
                         const float* A, size_t A_rows, size_t A_cols,
                         const float* B, size_t B_rows, size_t B_cols)
{
    if (!cublas_handle_) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    }
    // cuBLAS is column-major but our data is row-major.
    // C = A * B  (row-major)  ≡  C^T = B^T * A^T  (col-major)
    // cuBLAS sees our row-major A as col-major A^T automatically,
    // so we just swap A and B in the call and flip M/N.
    //
    // cublasSgemm computes: result = alpha*(B * A) + beta*result
    // with dimensions: result(M x N) = B(M x K) * A(K x N)  in col-major
    // which gives us:  result(A_rows x B_cols) = A(A_rows x A_cols) * B(A_cols x B_cols)
    int M = (int)A_rows;
    int K = (int)A_cols;
    int N = (int)B_cols;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cublasSgemm(handle, transB, transA, N, M, K, alpha, B, N, A, K, beta, result, N)
    CUBLAS_CHECK(cublasSgemm(cublas_handle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             result, N));
}

void CudaBackend::transpose(float* result, const float* A, size_t rows, size_t cols) {
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(result, A, (int)rows, (int)cols);
}

// im2col and col2im use 1d blocks since we use a flat indexing scheme in the kernel

void CudaBackend::im2col(const float* input, float* col,
                         int batch, int in_channels, int height, int width,
                         int kernel_h, int kernel_w, int out_h, int out_w,
                         int pad_h, int pad_w, int stride_h, int stride_w) {
    int total = batch * out_h * out_w;
    im2col_kernel<<<(total + 255) / 256, 256>>>(
        input, col,
        batch, in_channels, height, width,
        kernel_h, kernel_w, out_h, out_w,
        pad_h, pad_w, stride_h, stride_w);
}

void CudaBackend::col2im(const float* col, float* input,
                         int batch, int in_channels, int height, int width,
                         int kernel_h, int kernel_w, int out_h, int out_w,
                         int pad_h, int pad_w, int stride_h, int stride_w) {
    int total = batch * out_h * out_w;
    col2im_kernel<<<(total + 255) / 256, 256>>>(
        col, input,
        batch, in_channels, height, width,
        kernel_h, kernel_w, out_h, out_w,
        pad_h, pad_w, stride_h, stride_w);
}

void CudaBackend::bias_add(float* data, const float* bias,
                            size_t rows, size_t cols) {
    size_t total = rows * cols;
    bias_add_kernel<<<(total + 255) / 256, 256>>>(data, bias, (int)rows, (int)cols);
}

void CudaBackend::sum_rows(float* output, const float* input,
                            size_t rows, size_t cols) {
    sum_rows_kernel<<<(cols + 255) / 256, 256>>>(output, input, (int)rows, (int)cols);
}

void CudaBackend::adam_update(float* param, const float* grad,
                               float* m, float* v,
                               float lr, float beta1, float beta2,
                               float bc1, float bc2, float eps,
                               size_t size) {
    adam_update_kernel<<<(size + 255) / 256, 256>>>(
        param, grad, m, v, lr, beta1, beta2, bc1, bc2, eps, size);
}

void CudaBackend::nhwc_to_nchw(const float* src, float* dst,
                                 int batch, int channels, int h, int w) {
    int total = batch * channels * h * w;
    nhwc_to_nchw_kernel<<<(total + 255) / 256, 256>>>(src, dst, batch, channels, h, w);
}

void CudaBackend::nchw_to_nhwc(const float* src, float* dst,
                                 int batch, int channels, int h, int w) {
    int total = batch * channels * h * w;
    nchw_to_nhwc_kernel<<<(total + 255) / 256, 256>>>(src, dst, batch, channels, h, w);
}

void CudaBackend::maxpool_forward(const float* input, float* output, int* indices,
                                   int batch, int channels,
                                   int in_h, int in_w,
                                   int out_h, int out_w,
                                   int pool_h, int pool_w,
                                   int stride_h, int stride_w) {
    int total = batch * channels * out_h * out_w;
    maxpool_forward_kernel<<<(total + 255) / 256, 256>>>(
        input, output, indices,
        batch, channels, in_h, in_w, out_h, out_w,
        pool_h, pool_w, stride_h, stride_w);
}

void CudaBackend::maxpool_backward(const float* grad_output, float* grad_input,
                                    const int* indices,
                                    int output_size, int /*input_size*/) {
    maxpool_backward_kernel<<<(output_size + 255) / 256, 256>>>(
        grad_output, grad_input, indices, output_size);
}

void CudaBackend::global_avg_pool_forward(const float* input, float* output,
                                           int batch, int channels, int h, int w) {
    int total = batch * channels;
    global_avg_pool_forward_kernel<<<(total + 255) / 256, 256>>>(
        input, output, batch, channels, h, w);
}

void CudaBackend::global_avg_pool_backward(const float* grad_output, float* grad_input,
                                            int batch, int channels, int h, int w) {
    int total = batch * channels * h * w;
    global_avg_pool_backward_kernel<<<(total + 255) / 256, 256>>>(
        grad_output, grad_input, batch, channels, h, w);
}

void CudaBackend::softmax_forward(const float* input, float* output,
                                   size_t rows, size_t cols) {
    softmax_forward_kernel<<<(rows + 255) / 256, 256>>>(input, output, (int)rows, (int)cols);
}

void CudaBackend::softmax_backward(const float* softmax_output,
                                    const float* grad_output,
                                    float* grad_input,
                                    size_t rows, size_t cols) {
    softmax_backward_kernel<<<(rows + 255) / 256, 256>>>(
        softmax_output, grad_output, grad_input, (int)rows, (int)cols);
}

void CudaBackend::softmax_cross_entropy(const float* logits, const float* targets,
                                         float* grad, float* loss_out,
                                         int batch, int num_classes) {
    softmax_cross_entropy_kernel<<<(batch + 255) / 256, 256>>>(
        logits, targets, grad, loss_out, batch, num_classes);
}

int* CudaBackend::allocate_int(size_t size) {
    int* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(int)));
    return ptr;
}

void CudaBackend::deallocate_int(int* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

} // namespace nn
