
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/benchmark/stopwatch.h>
#include <quickcluster/linearalgebra/operations.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>

using std::vector;
using std::runtime_error;

KMeans::KMeans(size_t k, size_t iterations, int random_state, float epsilon, DeviceHandle device_handle) {
    
    this->_k = k;
    this->_iterations = iterations;
    this->_fitted = false;
    this->_epsilon = epsilon;
    this->device_handle = device_handle;

    srand(random_state);
}

void KMeans::fit(const Array<float> &data) {

    size_t N = data.rows();

    // 1. Initialize k centroids with random points
    //      We need the min and max for each feature

    // Create K centroids with randomly initialized points
    Array centroids = this->create_centroids(data);

    // Used to store the index of the closest centroid at the respective index
    // Each index correlates to the feature at the same index in the data
    size_t *centroid_index_for_data = new size_t[N];
    memset(centroid_index_for_data, 0, sizeof(size_t) * N);

    // Number of rows

    // Some data for the GPU
    DataContext datacontext;
    datacontext.cols = data.cols();
    datacontext.rows = N;
    datacontext.k = this->_k;

    // Adjust the centroids here
    for (size_t iteration = 0; iteration < this->_iterations; iteration++) {

        #ifdef __APPLE__
        if (device_handle) {

            // Device handle is present so do these calculations on the GPU
            metal_compute_nearest_centroids(device_handle, data.data(), centroids.data(), &datacontext, centroid_index_for_data);
        }
        else {

            // Default back to the CPU
            this->compute_clostest_centroids_index(data, centroids, centroid_index_for_data);
        }
        #elif defined(__NVCC__)
        // TODO implement CUDA
        exit(0)
        #else
        this->compute_clostest_centroids_index(data, centroids, centroid_index_for_data);
        #endif


        // We now have an array of N rows of closest centroid indexes
        Array centroid_avg = compute_mean(data, centroid_index_for_data);

        if (this->did_converge(centroids, centroid_avg)) {
            break;
        }

        centroids = centroid_avg;
    }

    // Delete the centroid index buffer
    delete [] centroid_index_for_data;

    // use these centroids to predeict others
    this->_centroids = centroids.copy();
    this->_fitted = true;
}

Array<float> KMeans::create_centroids(const Array<float> &data) const {

    // Empty array of zeros
    auto centroids = Array<float>::values(this->_k * data.cols(), 0);

    // K rows X data.cols columns
    centroids.resize(this->_k, data.cols());

    for (size_t i = 0; i < this->_k; i++) {
        centroids.set(i, data.row(rand() % data.rows()));
    }

    return centroids;
}

size_t KMeans::closest_centroid_index(const Array<float> &feature, const Array<float> &centroids) const {

    // feature:     1 X Cols array
    // centroids:   K X Cols array
    size_t centroid_count = centroids.rows();

    float min_distance = 0.0;
    size_t closest_index = 0;

    for (size_t i = 0; i < centroid_count; i++) {

        auto centroid = centroids.row(i);

        float distance = distance_euclidean(centroid, feature);

        if (distance < min_distance || min_distance == 0.0) {
            min_distance = distance;
            closest_index = i;
        }
    }

    return closest_index;
}

void KMeans::compute_clostest_centroids_index(const Array<float> &data, const Array<float> &centroids, size_t *indexes) const {

    size_t N = data.rows();

    for (size_t i = 0; i < N; i++) {

        auto feature = data.row(i);

        // Get the clostest centroid index
        size_t centroid_index = closest_centroid_index(feature, centroids);
        indexes[i] = centroid_index;
    }
}

Array<float> KMeans::compute_mean(const Array<float> &data, size_t *indexes) const {

    size_t k = this->_k;

    size_t rows = data.rows();
    size_t cols = data.cols();

    size_t clusters_per_index[k];
    float feature_sums[cols * k];

    memset(clusters_per_index, 0, k * sizeof(size_t));
    memset(feature_sums, 0, cols * k * sizeof(float));

    for (size_t i = 0; i < rows; i++) {
        size_t centroid_index = indexes[i];
        size_t ptr_offset = cols * centroid_index;

        buffer_sum(feature_sums + ptr_offset, data.row(i).data(), cols);
        clusters_per_index[centroid_index] += 1;
    }

    // Compute the mean
    for (size_t i = 0; i < k; i++) {

        if (clusters_per_index[i] == 0) {
            continue;
        }

        float divisor = static_cast<float>(clusters_per_index[i]);

        for (size_t j = 0; j < cols; j++) {
            feature_sums[j + (i * cols)] /= divisor;
        }
    }

    return Array<float>(feature_sums, k, cols, true);
}

bool KMeans::did_converge(const Array<float> c1, const Array<float> c2) const {

    float max_distance = 0.0;

    size_t N = c1.rows();

    for (size_t i = 0; i < N; i++) {
        max_distance = std::max(max_distance, distance_euclidean(c1, c2));
    }

    return max_distance <= this->_epsilon; 
}

Array<int> KMeans::predict(const Array<float> &data) const {

    if (!this->_fitted) {
        throw runtime_error("The model must be fitted first");
    }

    // Ensure that the data we are passing has the same number of columns as the centroids
    if (data.cols() != this->_centroids.cols()) {
        throw runtime_error("Column count of data must match centroid");
    }

    size_t rows = data.rows();

    auto predict_buffer = Array<int>::values(rows, 0);

    for (size_t i = 0; i < rows; i++) {

        auto feature = data.row(i);

        size_t index = closest_centroid_index(feature, this->_centroids);

        predict_buffer.set(i, index);
    }

    return predict_buffer;
}

Array<float> KMeans::centroids() const {
    return this->_centroids;
}