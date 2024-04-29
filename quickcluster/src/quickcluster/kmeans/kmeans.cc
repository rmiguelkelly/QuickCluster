
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/benchmark/stopwatch.h>

#include <iostream>
#include <vector>
#include <math.h>

using std::vector;
using std::runtime_error;
using std::reference_wrapper;

KMeans::KMeans(size_t k, size_t iterations, int random_state, float epsilon) {
    
    this->_k = k;
    this->_iterations = iterations;
    this->_fitted = false;
    this->_epsilon = epsilon;

    srand(random_state);
}

void KMeans::fit(const Array<float> &data) {

    // 1. Initialize k centroids with random points
    //      We need the min and max for each feature

    // Create K centroids with randomly initialized points
    auto centroids = this->create_centroids(data);

    // We will group the centroids here
    vector<reference_wrapper<Array<float>>> grouped_features[this->_k];

    // Number of rows
    size_t rows = data.rows();

    // Adjust the centroids here
    for (size_t iteration = 0; iteration < this->_iterations; iteration++) {
        
        // Initialize empty centroid vectors
        for (size_t i = 0; i < this->_k; i++) {
            grouped_features[i] = { };
        }

        // Group the centroids into groups
        for (size_t i = 0; i < rows; i++) {

            auto feature = data.row(i);

            size_t centroid_index = closest_centroid_index(feature, centroids);

            // Add this centroid to the specific group
            grouped_features[centroid_index].push_back(feature);
        }

        // Find the mean

        bool converged = true;

        for (size_t i = 0; i < this->_k; i++) {
            converged &= set_centroid_mean(grouped_features[i], centroids, i);
        }

        // If the clusters converged then the fitting can terminate
        if (converged) {
            break;
        }
    }

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
    size_t rows = centroids.rows();

    float min_distance = 0.0;
    size_t closest_index = 0;

    for (size_t i = 0; i < rows; i++) {

        auto centroid = centroids.row(i);

        float distance = distance_euclidean(centroid, feature);

        if (distance < min_distance || min_distance == 0.0) {
            min_distance = distance;
            closest_index = i;
        }
    }

    return closest_index;
}

bool KMeans::set_centroid_mean(const vector<reference_wrapper<Array<float>>> &features, Array<float> &centroids, size_t index) const {

    if (features.empty()) {
        return false;
    }

    size_t rows = features.size();
    size_t cols = features[0].get().size();

    bool converged = true;
    
    for (size_t col = 0; col < cols; col++) {

        float total = 0.0;

        for (size_t i = 0; i < rows; i++) {
            total += features[i].get()[col];
        }

        size_t offset = index * cols;

        float mean = total / rows;

        // Check for convergence
        converged &= abs(mean - centroids[col + offset]) <= this->_epsilon;

        centroids.set(col + offset, mean);
    }

    return converged;
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