#include <iostream>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found! Check your WSL setup." << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA device(s). You are ready!" << std::endl;
    }
    return 0;
}