#ifndef SYCL_ML_LIB_BLAS1_H

#include <CL/sycl.hpp>

template<typename type>
type * SYCLtranspose(type* input, size_t rows, size_t columns, type* target_matrix, queue Queue){
    buffer<type, 2> matrix{input, {rows, columns}};
    buffer<type, 2> matrix_copy{target_matrix, {rows, columns}};

    Queue.submit([&matrix, &matrix_copy, rows, columns, second](handler& cgh){
       auto matrix_accessor = matrix.get_access<access::mode::read>(cgh);
       auto final_accessor = matrix.get_access<access::mode::write>(cgh);

       cgh.parallel_for<class transpose>(range<2>{rows, columns}, [matrix_accessor, final_accessor](id<2> idx){
          final_access[idx.get(1)][idx.get(0)] = matrix_accessor[idx.get(0)][idx.get(1)];
       });
       cgh.copy(final_accessor, target_matrix);
    });
    Queue.wait();
    return second;
}
#endif //SYCL_ML_LIB_BLAS1_H
