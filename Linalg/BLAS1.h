#ifndef SYCL_ML_LIB_BLAS1_H
#define SYCL_ML_LIB_BLAS1_H

#include <iostream>
#include "Kernels/BLAS1_Kernel.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

template<typename type>  type* transpose(type* input, type* output, size_t rows, size_t columns){
    //BLAS Dispatch Here
#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
#pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            *(output + j * rows + i) = *(input + i * columns + j);
        }
    }
    return output;
}

template<typename type> type* transpose(type* input, type* output, size_t rows, size_t columns, queue Queue){
    output = SYCLtranspose(input, rows, columns, output, Queue);
    return output;
}


#endif //SYCL_ML_LIB_BLAS1_H
