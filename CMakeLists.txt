cmake_minimum_required(VERSION 3.17)
project(SYCL_ML_Lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH /home/atharva/computecpp-sdk/cmake/Modules/)
find_package(ComputeCpp)
include_directories($(COMPUTECPP_INCLUDE_DIRECTORY))
set(ComputeCpp_DIR /home/atharva/ComputeCPP/computeCPP/)

add_subdirectory(Kernels)
add_subdirectory(Linalg)
add_subdirectory(Linalg/Kernels)

add_executable(SYCL_ML_Lib main.cpp clustering.h Device.h Regression.h tests.h)
add_sycl_to_target(TARGET SYCL_ML_Lib SOURCES Kernels/RegressionKernel.h)
add_sycl_to_target(TARGET SYCL_ML_Lib SOURCES Linalg/Kernels/BLAS1.h)
