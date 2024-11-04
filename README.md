# cuda-concurrent-examples
# cuda-concurrent-examples

You need to download NVIDIA Driver, NVIDIA ToolKit and Nsight-systems to run the program
```
//Install NVIDIA Driver
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers list 
sudo apt install nvidia-driver-550 nvidia-fabricmanager-550 libnvidia-nscq-550
```
verify the driver is installed
```
nvidia-smi
```
Install the CUDA toolkit
```
sudo apt install nvidia-cuda-toolkit
```
Download the nsight-systems deb file from [NVIDIA](https://developer.nvidia.com/nsight-systems/get-started)
Install the dependencies for nsight-systems and install the .deb file:
```
sudo apt-get install libxcb-xinput0 libxcb-cursor0
sudo dpkg -i nsight-systems-[VERSION].deb 
```
Use ```nvcc cuda_streams.cu -o cuda_streams``` to compile and ```sudo nsys profile --force-overwrite true -o cuda_stream_profile ./cuda_streams``` to run
Or you can also use the ```Makefile`` to run the examples.