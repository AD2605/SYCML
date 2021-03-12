#include <CL/sycl.hpp>
#include <vector>
#include <tuple>

using namespace std;
using namespace cl::sycl;

class KmeansRunClass;
class DBscan;

std::pair<float**, int> KMeans_Kernel(device_selector selector,
                                      float** datapoints, float** centroids, int num_datapoints, int num_clusters){
    queue Queue(selector);
    buffer<float**, 1> datapoints_buffer{datapoints, {num_datapoints}};
    buffer<float**, 1> centroid_buffer{centroids, {num_clusters}};

    return std::pair<nullptr, 1>;
}