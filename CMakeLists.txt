cmake_minimum_required(VERSION 3.28)
project(untitled CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled main.cu)

set_target_properties(untitled PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
