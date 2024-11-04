Compile
```
nvcc cuda_streams.cu -o cuda_streams
```
Run with profiler
 ```
sudo nsys profile --force-overwrite true -o cuda_stream_profile ./cuda_streams
```
Or you can also use the ```Makefile`` to run. \\
The result will be generated as ```cuda_stream_profile.nsys-rep``` \\
You need to download it to a computer with GUI and nsight-host to view the result