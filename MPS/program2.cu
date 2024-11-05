#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel2(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 10000000000) { //cycles to spin
        data[idx] += value;
    }
}

int main() {
    const int arraySize = 1024;
    const int blockSize = 256;
    const int numBlocks = arraySize / blockSize;

    int *d_data1;
    cudaMalloc(&d_data1, arraySize * sizeof(int));
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);
    kernel2<<<numBlocks, blockSize>>>(d_data1, 10);
     // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "Kernel 2 completed." << std::endl;
    return 0;
}