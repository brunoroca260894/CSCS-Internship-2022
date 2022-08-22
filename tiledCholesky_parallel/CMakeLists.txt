cmake_minimum_required(VERSION 3.10)
include(CMakePrintHelpers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(tileDataType VERSION 1.0)
link_directories(/usr/local/lib) #to include LAPACK and BLAS

find_library(LAPACK_PATH lapack) #this is not used but just to very LAPACK AND BLAS are installed
find_library(BLAS_PATH blas)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(tileDataType main.cpp)
target_include_directories(tileDataType PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tileDataType lapack blas gfortran)

#cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(LAPACK) #to read where LAPACK are BLAS are
cmake_print_variables(BLAS)
