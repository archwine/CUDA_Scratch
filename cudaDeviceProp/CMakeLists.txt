cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(DeviceProp CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
add_executable(DeviceProp cudaDeviceProp.cu)
