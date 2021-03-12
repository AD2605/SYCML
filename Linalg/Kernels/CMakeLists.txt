cmake_minimum_required(VERSION 3.17)
project(Linalg_Kernels)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER /home/atharva/ComputeCPP/computeCPP/bin/compute++)
set(CMAKE_CXX_FLAGS -sycl-driver)

set(ComputeCpp_DIR /home/atharva/ComputeCPP/computeCPP/)
set(CMAKE_MODULE_PATH /home/atharva/computecpp-sdk/cmake/Modules/)

find_package(ComputeCpp)
include_directories($(COMPUTECPP_INCLUDE_DIRECTORY))

add_library(Linalg_Kernels BLAS1_Kernel.h)
target_link_libraries(Linalg_Kernels PUBLIC ComputeCpp::ComputeCpp)
set_target_properties(Linalg_Kernels PROPERTIES LINKER_LANGUAGE CXX)