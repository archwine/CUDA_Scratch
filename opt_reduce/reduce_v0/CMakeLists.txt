cmake_minimum_required(VERSION 3.18)
project(reduce_v0 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89) 
add_executable(${PROJECT_NAME} reduce_v0.cu)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})