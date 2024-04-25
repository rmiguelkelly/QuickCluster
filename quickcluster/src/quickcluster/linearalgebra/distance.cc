
#include <quickcluster/linearalgebra/distance.h>

float distance_euclidean(const Array<float> &point1, const Array<float> &point2) {

    float distance = 0.0;

    for (size_t i = 0; i < point1.size(); i++) {
        distance += powf(point1[i] - point2[i], 2);
    }

    return sqrt(distance);
}