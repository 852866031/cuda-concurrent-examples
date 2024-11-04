#include <cuda_runtime.h>
#include <iostream>

__global__ void simpleKernel(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 100000000) { //cycles to spin
        data[idx] += value;
    }
}

int main() {
    const int arraySize = 1024;
    const int blockSize = 256;
    const int numBlocks = arraySize / blockSize;

    int *d_data1, *d_data2;
    cudaMalloc(&d_data1, arraySize * sizeof(int));
    cudaMalloc(&d_data2, arraySize * sizeof(int));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch kernels in separate streams
    simpleKernel<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    simpleKernel<<<numBlocks, blockSize, 0, stream2>>>(d_data2, 20);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);

    return 0;
}
