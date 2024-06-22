
#ifndef KMEANS_H
#define KMEANS_H

#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/linearalgebra/distance.h>
#include <quickcluster/device/common.h>

#ifdef __APPLE__
#include <quickcluster/device/metal.h>
#endif

#define EPSILON 0.001

using std::vector;
using std::reference_wrapper;

/// @brief Adds two buffers
/// @tparam T numeric type
/// @param dst adds the src buffer to this buffer of values
/// @param src this buffer is added to dst and is not modified
/// @param N number of elements in src and dst
template<typename T> inline void buffer_sum(T *dst, const T *src, size_t N) {
    for (size_t i = 0; i < N; i++) {
        dst[i] += src[i];
    }
}

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

    DeviceHandle device_handle;

    // Creates K random clusters, returns a matrix of K rows of features with N dimensions
    Array<float> create_centroids(const Array<float> &data) const;

    // Returns the index of the closest centroid for a single feature
    // feature      - 1 x Dimensions feature set
    // centroids    - K x Dimensions centroids
    size_t closest_centroid_index(const Array<float> &feature, const Array<float> &centroids) const;

    /// Computes the closest centroid index for all feature
    void compute_clostest_centroids_index(const Array<float> &data, const Array<float> &centroids, size_t *indexes) const;

    /// From a list of closest indexes per feature compute their mean
    Array<float> compute_mean(const Array<float> &data, size_t *indexes) const;

    // Checks if the new computed centroids are close enough to the previous centroids
    bool did_converge(const Array<float> c1, const Array<float> c2) const;

public:

    /** @brief Initialize the KMeans clustering
    * 
    * @param[in]  k  number of clusters to group the data into
    * @param[in]  iterations  Maximum number of iterations before stopping
    * @param[in]  random_state  Used to seed the RNG
    * @param[in]  epsilon Arbitary number used to check if the clusters converged
    * @param[in]  device_handle Pointer to a gpu device handler, null by default
    */
    KMeans(size_t k, size_t iterations, int random_state, float epsilon = EPSILON, DeviceHandle device_handle = nullptr);

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