cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(sum_arrays LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80)

add_executable(sum_arrays sum_arrays.cu)
target_include_directories(sum_arrays PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(sum_arrays PRIVATE cudart)
