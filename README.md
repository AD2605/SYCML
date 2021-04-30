# SYCML (SYCL ML)
A Lightweight, Powerful Header only Machine Learning Library written with  SYCL support for multi-vendor target platforms which support SPIR/SPIR-V instructions. 
This library aims to stand as a substitute to `cuML`, `RAPIDS` etc and provide a wider range of accelerator devices like GPU's from any Vendor(Even Intel and AMD) and FPGA's and ASICS as well. 

Currently features the following - 
* Linear and Logistic Regression. 
* PCA, SVD
* K means Clustering
* Comptetitive Learning and SOM

## Usage 
The usage is very simple. To target an accelerator device, simply call `.sycl()` on your model and pass the target as a string in the `forward` method. 
```
#include "Regression.h"
int main(){
    size_t size = 32768;
    std::vector<float> input_vector(size, 1.0f);

    LinearRegression linearRegression(size); //Initialize the Class.
    linearRegression.sycl("AMD"); //call the .sycl() method to run the forward and Backward on SPIR supported Devices.
    float output = linearRegression.forward(input_vector);

    // One can choose Devices by Vendor Names, AMD, Intel, Arm or
    // by Device type, GPU, CPU and the HOST.
    // One can change the target Device after Initialization as well.

    float loss = -2.0124;
    float learning_rate = 1e-3;
    linearRegression.setDevice(std::string("Intel"));

    linearRegression.backward(loss, learning_rate);
}
```
If one does not chose the `.sycl()` option, the algorithms will be accelerated using `OpenMP`. Later Down the line in further Releases, I plan to use more vendor specific libraries like 'mkl' for Intel and 'ACML' for AMD. 

### What is SYCL 
SYCL (pronounced ‘sickle’) is a royalty-free, cross-platform abstraction layer that enables code for heterogeneous processors to be written using standard ISO C++ with the host and kernel code for an application contained in the same source file. This Enables us to target more accelerator devices like GPU's from AMD and even FPGA's and ASICS instead of relying only on Nvidia's devices and this is the main motivation behind this library. 

### Dependencies
This library uses `ComputeCPPs` SYCL Implementation and their Compiler. Please have it up and running. https://developer.codeplay.com/products/computecpp/ce/download
