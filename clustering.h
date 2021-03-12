#ifndef SYCL_ML_LIB_CLUSTERING_H
#define SYCL_ML_LIB_CLUSTERING_H

#include <tuple>
#include <vector>
#include <string>
#include <stack>
#include <random>

#include <CL/sycl.hpp>
#include "Device.h"
#include "Kernels/clusterKernel.h"


/*
 * This function would in return be called by another function exposed to the user
 * where the input can be given as vector<vector>*/

inline vector<float*> KMeans(std::vector<float*> dataPoints, int num_clusters, string Device, float eps = 1e-3, int dimension){
    /*
     * Simple K Means with random Data points as the Initial Centroids.
     * Distance with */

    auto num_Datapoints = dataPoints.size();
    int* centroid_position = (int*)malloc()
    random_device device;
    std::mt19937_64 generator(device());
    std::unordered_set<int> positions;

    for(int r = num_datapoints - num_clusters; r<num_datapoints; ++r){
        int v = std::uniform_int_distribution<>(1, r)(generator);
        if(!positions.insert(v).second){
            positions.insert(r);
        }
    }

    vector<float*> centroids;

    for(int i=0; i<num_clusters; i++){
        centroids.push_back(dataPoints.at(positions.at(i)));
    }

    auto datapoint_ptr = datapoints.data();
    auto centroids_ptr = centroids.data();

    auto device = getDeviceSelector(Device);

    while(true){
        auto centroid_and_done = KMeans_Kernel()
    }

}
