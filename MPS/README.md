1. **Start the MPS Control Daemon**
   - Run `nvidia-cuda-mps-control -d` to start the MPS control daemon in the background.

2. **Compile Two CUDA Programs**
   - Prepare and compile two separate CUDA programs that each run a kernel.

3. **Run Both Programs Under MPS**
   - Execute both programs concurrently from separate terminals or scripts, using the `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable to manage resource allocation.

4. **Profile Using Nsight Systems**
   - Use `nsys profile` to profile the execution of each program and generate profiling reports.
   - Analyze the `.qdrep` files in the Nsight Systems GUI to observe kernel concurrency.

5. **Stop the MPS Control Daemon**
   - Use `nvidia-cuda-mps-control` and type `quit` to stop the MPS control daemon when finished.

## Notes
- Adjust `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` to experiment with resource allocation.
- Analyze the profiling data to confirm concurrent kernel execution.
