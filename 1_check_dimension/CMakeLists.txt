cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(check_dimension CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
add_executable(check_dimension check_dimension.cu)
target_link_libraries(check_dimension cudart)
