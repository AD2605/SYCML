set(CMAKE_CXX_STANDARD 3.17)
project(Linalg CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS -fopenmp)
set(CMAKE_CXX_FLAGS -fopenmp-simd)
set(CMAKE_CXX_FLAGS -O3)

set(ComputeCpp_DIR /home/atharva/ComputeCPP/computeCPP/)
set(CMAKE_MODULE_PATH /home/atharva/computecpp-sdk/cmake/Modules/)
find_package(ComputeCpp)
include_directories($(COMPUTECPP_INCLUDE_DIRECTORY))
set(CMAKE_CXX_COMPILER /home/atharva/ComputeCPP/computeCPP/bin/compute++)
set(CMAKE_CXX_FLAGS -sycl-driver)

add_library( Linalg BLAS1.h
        BLAS1.cpp)
link_libraries(Linalg PUBLIC ComputeCpp::ComputeCpp)