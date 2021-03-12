//
// Created by atharva on 05/03/21.
//

#ifndef SYCL_ML_LIB_REGRESSIONKERNEL_H
#define SYCL_ML_LIB_REGRESSIONKERNEL_H

#include <CL/sycl.hpp>
#include <string>
#include <iostream>

#include "Device.h"

class regression;
class ReductionKernel;
class calcProducts;
class weightUpdate;

using namespace cl::sycl;

float RegressionForward(float* input, float* weights, float bias, size_t num_features, queue Queue) {

    float result;
    auto device = Queue.get_device();
    auto deviceName = device.get_info<cl::sycl::info::device::name>();
    size_t local = std::min(num_features, device.get_info<info::device::max_work_group_size>());

    auto calc_array = (float *) malloc(num_features * sizeof(float));
    buffer<float, 1> weights_buffer{weights, {num_features}};
    buffer<float, 1> input_buffer{input, {num_features}};
    buffer<float, 1> calc_buffer{calc_array, {num_features}};



    Queue.submit([&weights_buffer, &input_buffer, &calc_buffer, num_features, result](handler& cgh) {
        auto weights_Accessor = weights_buffer.get_access<access::mode::read>(cgh);
        auto input_Accessor = input_buffer.get_access<access::mode::read>(cgh);
        auto calc_Accessor = calc_buffer.get_access<access::mode::write>(cgh);

        cgh.parallel_for<regression>(range<1>(num_features), [=](id<1> idx) {
            calc_Accessor[idx.get(0)] = weights_Accessor[idx.get(0)] * input_Accessor[idx.get(0)];
        });
        cgh.update_host(calc_Accessor);
    });
    Queue.wait();
    do{
        auto lamda_function = [num_features, local, &calc_buffer](handler& cgh) mutable {
            nd_range<1> indexer{range<1>{std::max(num_features, local)},
                                range<1>{std::min(num_features, local)}};
            auto calc_accessor = calc_buffer.get_access<access::mode::read_write>(cgh);
            accessor<float, 1, access::mode::read_write, access::target::local> calc_scratchPad(range < 1 > (local), cgh);

            cgh.parallel_for<ReductionKernel>(indexer, [calc_accessor, calc_scratchPad, local, num_features](nd_item<1> id) {

                size_t globalID = id.get_global_id(0);
                size_t localID = id.get_local_id(0);

                if (globalID < num_features) {
                    calc_scratchPad[localID] = calc_accessor[globalID];
                }
                id.barrier(access::fence_space::local_space);
                if (globalID < num_features) {
                    int min = (num_features < local) ? num_features : local;
                    for (size_t offset = min / 2; offset > 0; offset /= 2) {
                        if (localID < offset) {
                            calc_scratchPad[localID] += calc_scratchPad[localID + offset];
                        }
                        id.barrier(access::fence_space::local_space);
                    }
                    if (localID == 0) {
                        calc_accessor[id.get_group(0)] = calc_scratchPad[localID];
                    }
                }
            });
        };

        Queue.submit(lamda_function);
        num_features = num_features/local;
    } while (num_features > 1);
    Queue.wait();
    {
        auto result_accessor = calc_buffer.get_access<access::mode::read>();
        result = result_accessor[0];
    }
    return  result + bias;
}

float reductionKernel(cl::sycl::buffer<float> calc_buffer, size_t num_features){
    queue Queue;
    float result;
    auto device = Queue.get_device();
    auto deviceName = device.get_info<cl::sycl::info::device::name>();
    size_t local = std::min(num_features, device.get_info<info::device::max_work_group_size>());

    do{
        auto lamda_function = [num_features, local, &calc_buffer](handler& cgh) mutable {
            nd_range<1> indexer{range<1>{std::max(num_features, local)},
                                range<1>{std::min(num_features, local)}};
            auto calc_accessor = calc_buffer.get_access<access::mode::read_write>(cgh);
            accessor<float, 1, access::mode::read_write, access::target::local> calc_scratchPad(range < 1 > (local), cgh);

            cgh.parallel_for<class ReductionKernel1>(indexer, [calc_accessor, calc_scratchPad, local, num_features](nd_item<1> id) {

                size_t globalID = id.get_global_id(0);
                size_t localID = id.get_local_id(0);

                if (globalID < num_features) {
                    calc_scratchPad[localID] = calc_accessor[globalID];
                }
                id.barrier(access::fence_space::local_space);
                if (globalID < num_features) {
                    int min = (num_features < local) ? num_features : local;
                    for (size_t offset = min / 2; offset > 0; offset /= 2) {
                        if (localID < offset) {
                            calc_scratchPad[localID] += calc_scratchPad[localID + offset];
                        }
                        id.barrier(access::fence_space::local_space);
                    }
                    if (localID == 0) {
                        calc_accessor[id.get_group(0)] = calc_scratchPad[localID];
                    }
                }
            });
        };

        Queue.submit(lamda_function);
        num_features = num_features/local;
    } while (num_features > 1);

    {
        auto result_accessor = calc_buffer.get_access<access::mode::read>();
        result = result_accessor[0];
    }
    return result;
}

void updateWeights(float* weights, float* inputs, size_t num_features, float lr, float loss){
    queue Queue;

    buffer<float, 1> weight_buffer{weights, {num_features}};
    buffer<float, 1> input_buffer{inputs, {num_features}};

    Queue.submit([&weight_buffer, &input_buffer, num_features, lr, loss, weights](handler& cgh){
       auto weight_accessor = weight_buffer.get_access<access::mode::read_write>(cgh);
       auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);

       cgh.parallel_for<weightUpdate>(range<1>(num_features), [weight_accessor, input_accessor, lr, loss, weights](id<1> idx){
           weight_accessor[idx.get(0)] = weight_accessor[idx.get(0)]  - 2*lr*loss*input_accessor[idx.get(0)];
       });
       cgh.copy(weight_accessor, weights);
    });

}

template<typename Func>
void updateWeights_custom(float* weights, float* inputs, size_t num_features, float lr, float loss, float output, float target, Func loss_func_grad){
    queue Queue;
    buffer<float, 1> weight_buffer{weights, {num_features}};
    buffer<float, 1> input_buffer{inputs, {num_features}};

    Queue.submit([&weight_buffer, &input_buffer, num_features, lr, loss, loss_func_grad, output, target, weights](handler& cgh){
        auto weight_accessor = weight_buffer.get_access<access::mode::read_write>(cgh);
        auto input_accessor = input_buffer.get_access<access::mode::read>(cgh);

        cgh.parallel_for<class weight_update>(range<1>(num_features), [weight_accessor, input_accessor, lr, loss, loss_func_grad, output, target](id<1> idx){
            float grad = loss_func_grad(input_accessor[idx.get(0)], output, target);
            weight_accessor[idx.get(0)] = weight_accessor[idx.get(0)] - lr*grad;
        });
        cgh.copy(weight_accessor,weights);
    });
}


#endif //SYCL_ML_LIB_REGRESSIONKERNEL_H
