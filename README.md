# cuda-concurrent-examples
Each directory under this repo contains an example to run CUDA kernels concurrently.

## Dependencies
Install NVIDIA packages
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