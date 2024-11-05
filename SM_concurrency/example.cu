//Note that this example only works to show that you can run two kernels in one SM if your GPU only has one SM
//In a GPU with multiple SMs, you cannot explicity distribute blocks to SM 
// so use kernel 1 to occupy all SMs and almost all GPU resouces and let kernel 2 use only 1 thread does not work
#include <iostream>
__device__ volatile int s = 0;

__global__ void k1() {
    while (s == 0) {};  // Spin until s is non-zero
}

__global__ void k2() {
    s = 1;  // Set s to 1, allowing k1 to exit
}

int main() {
    // Create two separate streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // Launch k1 in stream s1 and k2 in stream s2
    k1<<<1, 1, 0, s1>>>();
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
