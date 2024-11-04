MPS:
	nvcc program1.cu -o program1
	nvcc program2.cu -o program2

run:
	nvidia-cuda-mps-control -d
	CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 sudo nsys profile --force-overwrite true -o MPS_profile ./program1 & CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 ./program2 
	
clean:
	rm program1 program2