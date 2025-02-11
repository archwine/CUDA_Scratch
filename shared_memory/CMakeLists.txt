cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(L1_and_shared_memory LANGUAGES CXX CUDA)

add_executable(L1_and_shared_memory L1_and_sm_memory.cu)

target_link_libraries(L1_and_shared_memory cudart)