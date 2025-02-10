CMAKE_MINIMUM_REQUIRED_VERSION(VERSION 3.12)

SET(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(Reduce_family LANGUAGES CXX CUDA)
add_subdirectory(reduce_v0)