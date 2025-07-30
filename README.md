# CUDA_Gravity
##Setup
Untar `data.tar.xz` archive before running:
```
tar xvzf data.tar.xz
```
Adjust `Makefile` to correct Nvidia architecture:
```
ARCH=-arch sm_XX
```
Compile:
```
make all
```
