cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(choleskyGPU VERSION 1.0 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
link_directories(/usr/local/cuda-11.7/targets/x86_64-linux/lib/) # to include CUDA libraries

add_executable(tileDataType main.cpp)
target_link_libraries(tileDataType cublas cusolver )
