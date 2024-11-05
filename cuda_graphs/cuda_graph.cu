#include <cuda_runtime.h>
#include <iostream>

// Define four simple CUDA kernels
__global__ void A(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 1000000000) { // cycles to spin
        data[idx] += value;
    }
}

__global__ void B(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 1000000000) { // cycles to spin
        data[idx] += value;
    }
}

__global__ void C(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 1000000000) { // cycles to spin
        data[idx] += value;
    }
}

__global__ void D(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long int startClock = clock64();
    // Spin until the desired duration has passed
    while (clock64() - startClock < 1000000000) { // cycles to spin
        data[idx] += value;
    }
}

int main() {
    const int arraySize = 1024;
    const int blockSize = 256;
    const int numBlocks = arraySize / blockSize;
    int *d_data1;
    cudaMalloc(&d_data1, arraySize * sizeof(int));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t e1, e2, startEvent, stopEvent, seqStartEvent, seqStopEvent;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&seqStartEvent);
    cudaEventCreate(&seqStopEvent);


    // Measure time for CUDA Graph execution
    cudaEventRecord(startEvent, stream1);
    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

    A<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    cudaEventRecord(e1, stream1);
    B<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);

    cudaStreamWaitEvent(stream2, e1);
    C<<<numBlocks, blockSize, 0, stream2>>>(d_data1, 10);
    cudaEventRecord(e2, stream2);
    cudaStreamWaitEvent(stream1, e2);
    D<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);

    cudaGraph_t graph;
    cudaStreamEndCapture(stream1, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(graphExec, stream1);
    cudaEventRecord(stopEvent, stream1);
    cudaStreamSynchronize(stream1);

    // Calculate and print elapsed time for CUDA Graph execution
    float graphMilliseconds = 0;
    cudaEventElapsedTime(&graphMilliseconds, startEvent, stopEvent);
    std::cout << "Time taken for CUDA Graph execution: " << graphMilliseconds << " ms" << std::endl;

    // Measure time for sequential execution
    cudaEventRecord(seqStartEvent, stream1);

    A<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    cudaStreamSynchronize(stream1);
    B<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    cudaStreamSynchronize(stream1);
    C<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    cudaStreamSynchronize(stream1);
    D<<<numBlocks, blockSize, 0, stream1>>>(d_data1, 10);
    cudaStreamSynchronize(stream1);

    cudaEventRecord(seqStopEvent, stream1);
    cudaStreamSynchronize(stream1);

    // Calculate and print elapsed time for sequential execution
    float seqMilliseconds = 0;
    cudaEventElapsedTime(&seqMilliseconds, seqStartEvent, seqStopEvent);
    std::cout << "Time taken for sequential execution: " << seqMilliseconds << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(seqStartEvent);
    cudaEventDestroy(seqStopEvent);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);

    std::cout << "CUDA Graph executed successfully." << std::endl;
    return 0;
}
