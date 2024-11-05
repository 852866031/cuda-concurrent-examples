cuda_streams:
	nvcc cuda_graph.cu -o cuda_graph

run:
	sudo nsys profile --force-overwrite true -o cuda_graph_profile ./cuda_graph
	
clean:
	rm cuda_graph