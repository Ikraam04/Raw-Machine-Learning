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

        // Store the computed value into C[row, col].
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
    int K = (int)A_cols;  // == B_rows
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

//stubs, not implemented yet.

void CudaBackend::im2col(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int) {
    throw std::runtime_error("CudaBackend::im2col not implemented");
}

void CudaBackend::col2im(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int) {
    throw std::runtime_error("CudaBackend::col2im not implemented");
}

} // namespace nn
