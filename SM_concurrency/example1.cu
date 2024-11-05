#include <iostream>
__device__ volatile int s = 0;

__global__ void k1() {
    long long int startClock = clock64(); // Record the starting clock cycle
    while (s == 0) {
        // Check if the maximum running time has been exceeded
        if (clock64() - startClock > 10000000000) {
            if(threadIdx.x == 0 && blockIdx.x == 0) printf("Kernel k1: Maximum running time exceeded!\n");
            return;
        }
    }
}

__global__ void k2() {
    // Set `s` to 1, allowing `k1` to exit
    s = 1;
}

int main() {
    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Check the number of SMs and the maximum number of threads per block
    int numSMs = deviceProp.multiProcessorCount;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    // Print device properties for debugging
    std::cout << "Number of SMs: " << numSMs << std::endl;
    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;

    // Create two separate streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Launch `k1` with enough blocks to occupy nearly all SMs but leave one block available
    int numBlocks = numSMs - 2; // Leave one block available for `k2`
    int threadsPerBlock = maxThreadsPerBlock/2; // Use the maximum number of threads per block

    // Launch `k1` in stream `s1`
    k1<<<numBlocks, threadsPerBlock, 0, s1>>>();

    // Launch `k2` in stream `s2`
    k2<<<1, 1, 0, s2>>>();

    // Synchronize the device to wait for kernels to complete
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Program terminated successfully." << std::endl;
    }

    return 0;
}
