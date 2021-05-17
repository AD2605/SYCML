#ifndef SYCL_ML_LIB_SOM_H
#define SYCL_ML_LIB_SOM_H
#endif //SYCL_ML_LIB_SOM_H


#include <limits>
#include <iostream>
#include <vector>
#include <random>
#include <atomic>
#include <cassert>
#include "Device.h"


class competitiveLearning{
public:
    competitiveLearning(int feature_dim, int output_dim){
        this->input_dim = feature_dim;
        this->output_dim = output_dim;

        std::random_device device{};
        std::normal_distribution<float> distribution{0, 1};
        std::ranlux48 generator{device()};

        for(int i=0; i<output_dim; i++){
            weights.push_back((float*)malloc(feature_dim * sizeof(float)));
#pragma omp simd
                for(int j=0; j<this->input_dim; j++){
                    *(this->weights.at(i) + j) = distribution(generator);
                }
        }
    }

template<typename criterion_func>
    void forward(float* datapoint, float lr, criterion_func criterion){
        int minPos = -1;
        float minCriterion = std::numeric_limits<float>::max();

        for(int i = 0; i<this->output_dim; i++){
            auto distance = std::abs(criterion(datapoint, this->weights.at(i)));
            if(distance < minCriterion){
                minPos = i;
                minCriterion = distance;
            } 
        }
        assert(minPos != -1);

#pragma omp  simd
        for(int i=0; i<this->input_dim; i++)
            *(this->weights.at(minPos) + i) = *(this->weights.at(minPos) + i) + lr*minCriterion;
    }

private:
    int input_dim;
    int output_dim;
    std::vector<float*> weights;
};
template <typename distFunction>
class SOM{
public:
    SOM(size_t input_features, size_t output_features, distFunction function){
        this->num_features_in = input_features;
        this->num_features_out = output_features;
        this->criterion = function;

        std::random_device device{};
        std::normal_distribution<float> distribution{0, 1};
        std::ranlux48 generator{device()};

        for(int i=0; i < this->num_features_out; i++){
            this->weights.push_back((float*) malloc(this->num_features_in * sizeof(float)));
#pragma omp simd
            for (int j = 0; j < this->num_features_in; j++) {
                *(this->weights.at(i) + j) = distribution(generator);
            }
        }
    }

    template<typename damping>
    void forward(float* input, damping damping_func, float lr){
        int minPos = -1;
        float minCriterion = std::numeric_limits<float>::max();
        for(int i = 0; i<this->num_features_out; i++){
            auto distance = std::abs(this->criterion(datapoint, this->weights.at(i)));
            if(distance < minCriterion){
                minPos = i;
                minCriterion = distance;
            }
        }
        assert(minPos != -1);
        for(int i = 0; i<this->num_features_out; i++){
            if(i == minPos){
#pragma omp simd
                for(int j=0; i<this->num_features_in; j++)
                    *(this->weights.at(i) + j) = *(this->weights.at(i) + j) + lr * minCriterion;
            }
            else{
#pragma omp simd
                for (int j=0; j<this->num_features_in; j++) {
                    *(this->weights.at(i) + j) = *(this->weights.at(i) + j) + lr * damping_func(this->weights.at(i),
                                                                                                this->weights.at(minPos)) *
                                                                                                        this->criterion*(input, this->weights.at(i));
                }
            }

        }
    }

private:
    size_t num_features_in;
    size_t num_features_out;
    distFunction criterion;
    std::vector<float*> weights;

}
