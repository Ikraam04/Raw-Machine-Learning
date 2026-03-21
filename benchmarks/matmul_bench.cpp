// cuda backend now uses cublas (cublasSgemm) for matmul
// eigen uses its own optimised matmul via Eigen::Map

#include "core/tensor.h"
#include "backends/eigen_backend.h"
#include "backends/cuda_backend.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <memory>
//testtt
using namespace nn;
using namespace std::chrono;

static std::vector<float> random_data(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

static double bench(std::shared_ptr<Backend> backend, size_t N, int reps) {
    auto data_a = random_data(N * N);
    auto data_b = random_data(N * N);
    Tensor a({N, N}, backend); a.set_data(data_a);
    Tensor b({N, N}, backend); b.set_data(data_b);

    // first call often has driver/init overhead
    auto _ = a.matmul(b);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < reps; i++) {
         auto r = a.matmul(b);
        }
    auto end = high_resolution_clock::now();
    // return average time per matmul in milliseconds
    return duration<double, std::milli>(end - start).count() / reps;
}

int main() {
    auto eigen = std::make_shared<EigenBackend>();
    auto cuda  = std::make_shared<CudaBackend>();

    //square matrices
    std::vector<size_t> sizes = {64, 256, 512, 1024, 2048};

    std::cout << std::fixed;
    std::cout.precision(3);

    for (size_t N : sizes) {
        int reps = (N >= 1024) ? 5 : 10;
        double t_eigen = bench(eigen, N, reps);
        double t_cuda  = bench(cuda,  N, reps);
        std::cout << N << "x" << N << ":\n";
        std::cout << "  eigen: " << t_eigen << "ms\n";
        std::cout << "  cuda:  " << t_cuda  << "ms\n";
        std::cout << "  speedup: " << (t_eigen / t_cuda) << "x\n\n";
    }

    return 0;
}
