
#include <quickcluster/device/common.h>
#include <cstdlib>
#include <cmath>

inline float eclidean_distance(float *v1, float *v2, size_t N) {

    float sum = 0.0;

    for (size_t i = 0; i < N; i++) {
        sum += powf(v1[i] - v2[i], 2);
    }

    return sqrtf(sum);
}

__device__ void compute_nearest_centroids(const float *data, const float *centroids, DataContext *context, unsigned long *results) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;


}