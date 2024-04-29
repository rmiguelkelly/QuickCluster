
#ifndef KMEANS_H
#define KMEANS_H

#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/linearalgebra/distance.h>

using std::vector;
using std::reference_wrapper;

/** @brief A KMeans implementation in C++
 * 
 */
class KMeans {

private:
    size_t _k;
    size_t _iterations;
    bool _fitted;
    float _epsilon;
    Array<float> _centroids;

    // Creates K random clusters, returns a matrix of K rows of features with N dimensions
    Array<float> create_centroids(const Array<float> &data) const;

    // Returns the index of the closest centroid for a single feature
    // feature      - 1 x Dimensions feature set
    // centroids    - K x Dimensions centroids
    size_t closest_centroid_index(const Array<float> &feature, const Array<float> &centroids) const;

    // From a group of features, set the mean of the centroid at index
    bool set_centroid_mean(const vector<reference_wrapper<Array<float>>> &features, Array<float> &centroids, size_t index) const;

public:

    /** @brief Initialize the KMeans clustering
    * 
    * @param[in]  k  number of clusters to group the data into
    * @param[in]  iterations  Maximum number of iterations before stopping
    * @param[in]  random_state  Used to seed the RNG
    * @param[in]  epsilon Arbitary number used to check if the clusters converged
    */
    KMeans(size_t k, size_t iterations, int random_state, float epsilon = 0.00001);

    /** @brief Fits the data to the model
    * @param[in]  data  The data is a 2 dimensional array where each row represents a set of features
    */
    void fit(const Array<float> &data);

    /** @brief Predicts from a dataset the cluster index of each feature
    * @param[in]  data  The data is a 2 dimensional array where each row represents a set of features
    * @return A single dimension array of cluster indices
    */
    Array<int> predict(const Array<float> &data) const;

    /** @brief Gets the centroids
    * @return A K x N matrix representing the fitted centroids
    */
    Array<float> centroids() const;
};

#endif