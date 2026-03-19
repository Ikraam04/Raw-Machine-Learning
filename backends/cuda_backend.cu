#include "cuda_backend.h"
#include <cuda_runtime.h>
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

namespace nn
{

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
    // launch kernel to fill the array
    // syntax: kernel<<<numBlocks, blockSize>>>(args...)
    fill_kernel<<<(size + 255) / 256, 256>>>(dst, value, size);
    CUDA_CHECK(cudaDeviceSynchronize()); //synchronize to ensure kernel has finished before we return
}


void CudaBackend::add(float* result, const float* A, const float* B, size_t size) {
    add_kernel<<<(size + 255) / 256, 256>>>(result, A, B, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}
//element-wise multiplication
void CudaBackend::multiply(float* result, const float* A, const float* B, size_t size) {
    multiply_kernel<<<(size + 255) / 256, 256>>>(result, A, B, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::scale(float* result, const float* A, float scalar, size_t size) {
    scale_kernel<<<(size + 255) / 256, 256>>>(result, A, scalar, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void CudaBackend::relu(float* result, const float* A, size_t size) {
    relu_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::relu_derivative(float* result, const float* A, size_t size) {
    relu_derivative_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::sigmoid(float* result, const float* A, size_t size) {
    sigmoid_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::sigmoid_derivative(float* result, const float* A, size_t size) {
    sigmoid_derivative_kernel<<<(size + 255) / 256, 256>>>(result, A, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::matmul(float* result,
                         const float* A, size_t A_rows, size_t A_cols,
                         const float* B, size_t B_rows, size_t B_cols)
{
    int M = (int)A_rows;
    int K = (int)A_cols; 
    int N = (int)B_cols;

    // 16x16 block = 256 threads, arranged in 2D to map onto the output matrix
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<blocks, threads>>>(result, A, B, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaBackend::transpose(float* result, const float* A, size_t rows, size_t cols) {
    // 2d block just like matmul - one thread per output element
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(result, A, (int)rows, (int)cols);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// im2col and col2im can use 1d blocks since we use a flat indexing scheme in the kernel

void CudaBackend::im2col(const float* input, float* col,
                         int batch, int in_channels, int height, int width,
                         int kernel_h, int kernel_w, int out_h, int out_w,
                         int pad_h, int pad_w, int stride_h, int stride_w) {

    int total = batch * out_h * out_w;
    im2col_kernel<<<(total + 255) / 256, 256>>>(
        input, col,
        batch, in_channels, height, width,
        kernel_h, kernel_w, out_h, out_w,
        pad_h, pad_w, stride_h, stride_w
    );
    CUDA_CHECK(cudaDeviceSynchronize());
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
        pad_h, pad_w, stride_h, stride_w
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── stubs for new primitives (TODO: replace with real CUDA kernels) ──────────

void CudaBackend::bias_add(float* /*data*/, const float* /*bias*/,
                            size_t /*rows*/, size_t /*cols*/) {
    throw std::runtime_error("CudaBackend::bias_add not yet implemented");
}

void CudaBackend::sum_rows(float* /*output*/, const float* /*input*/,
                            size_t /*rows*/, size_t /*cols*/) {
    throw std::runtime_error("CudaBackend::sum_rows not yet implemented");
}

void CudaBackend::adam_update(float* /*param*/, const float* /*grad*/,
                               float* /*m*/, float* /*v*/,
                               float /*lr*/, float /*beta1*/, float /*beta2*/,
                               float /*bc1*/, float /*bc2*/, float /*eps*/,
                               size_t /*size*/) {
    throw std::runtime_error("CudaBackend::adam_update not yet implemented");
}

void CudaBackend::nhwc_to_nchw(const float* /*src*/, float* /*dst*/,
                                 int /*batch*/, int /*channels*/, int /*h*/, int /*w*/) {
    throw std::runtime_error("CudaBackend::nhwc_to_nchw not yet implemented");
}

void CudaBackend::nchw_to_nhwc(const float* /*src*/, float* /*dst*/,
                                 int /*batch*/, int /*channels*/, int /*h*/, int /*w*/) {
    throw std::runtime_error("CudaBackend::nchw_to_nhwc not yet implemented");
}

void CudaBackend::maxpool_forward(const float* /*input*/, float* /*output*/, int* /*indices*/,
                                   int /*batch*/, int /*channels*/,
                                   int /*in_h*/, int /*in_w*/,
                                   int /*out_h*/, int /*out_w*/,
                                   int /*pool_h*/, int /*pool_w*/,
                                   int /*stride_h*/, int /*stride_w*/) {
    throw std::runtime_error("CudaBackend::maxpool_forward not yet implemented");
}

void CudaBackend::maxpool_backward(const float* /*grad_output*/, float* /*grad_input*/,
                                    const int* /*indices*/,
                                    int /*output_size*/, int /*input_size*/) {
    throw std::runtime_error("CudaBackend::maxpool_backward not yet implemented");
}

void CudaBackend::global_avg_pool_forward(const float* /*input*/, float* /*output*/,
                                           int /*batch*/, int /*channels*/, int /*h*/, int /*w*/) {
    throw std::runtime_error("CudaBackend::global_avg_pool_forward not yet implemented");
}

void CudaBackend::global_avg_pool_backward(const float* /*grad_output*/, float* /*grad_input*/,
                                            int /*batch*/, int /*channels*/, int /*h*/, int /*w*/) {
    throw std::runtime_error("CudaBackend::global_avg_pool_backward not yet implemented");
}

void CudaBackend::softmax_forward(const float* /*input*/, float* /*output*/,
                                   size_t /*rows*/, size_t /*cols*/) {
    throw std::runtime_error("CudaBackend::softmax_forward not yet implemented");
}

void CudaBackend::softmax_backward(const float* /*softmax_output*/,
                                    const float* /*grad_output*/,
                                    float* /*grad_input*/,
                                    size_t /*rows*/, size_t /*cols*/) {
    throw std::runtime_error("CudaBackend::softmax_backward not yet implemented");
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
