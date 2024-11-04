Use 
```
nvcc cuda_streams.cu -o cuda_streams
```
to compile and 
 ```
sudo nsys profile --force-overwrite true -o cuda_stream_profile ./cuda_streams
```
to run the program with profiler \\
Or you can also use the ```Makefile`` to run. \\
The result will be generated as ```cuda_stream_profile.nsys-rep``` \\
You need to download it to a computer with GUI and nsight-host to view the result