cuda_streams:
	nvcc cuda_streams.cu -o cuda_streams

run:
	sudo nsys profile --force-overwrite true -o cuda_stream_profile ./cuda_streams
	
clean:
	rm cuda_streams