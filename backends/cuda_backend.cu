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

//stubs, not implemented yet.

void CudaBackend::matmul(float*, const float*, size_t, size_t, const float*, size_t, size_t) {
    throw std::runtime_error("CudaBackend::matmul not implemented");
}

void CudaBackend::transpose(float*, const float*, size_t, size_t) {
    throw std::runtime_error("CudaBackend::transpose not implemented");
}


void CudaBackend::im2col(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int) {
    throw std::runtime_error("CudaBackend::im2col not implemented");
}

void CudaBackend::col2im(const float*, float*, int, int, int, int, int, int, int, int, int, int, int, int) {
    throw std::runtime_error("CudaBackend::col2im not implemented");
}

} // namespace nn
