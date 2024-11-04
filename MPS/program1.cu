#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel1() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 100000000) { //cycles to spin
        data[idx] += value;
    }
}

int main() {
    kernel1<<<256, 256>>>();
    cudaDeviceSynchronize();
    std::cout << "Kernel 1 completed." << std::endl;
    return 0;
}
