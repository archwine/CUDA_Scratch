cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(CUDADeviceInfo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(CUDADeviceInfo queue.cu)

target_link_libraries(CUDADeviceInfo PRIVATE cudart)