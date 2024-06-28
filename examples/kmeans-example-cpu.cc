
#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/datasets/clusters.h>

#include <iostream>

int main() {

    // Create centers for the sample data generator
    Array<float> centers = {
        3.0f, 3.0f,
        7.0f, 7.0f,
        0.0f, 0.0f,
    };

    centers.resize(3, 2);

    // Create random sample data with 100 points per cluster
    Array<float> sample_data = create_blobs(centers, 100);

    int k = 3;
    size_t max_iterations = 100;

    time_t random_state;
    time(&random_state);

    // Initialize the KMeans class with the above params
    KMeans kmeans(k, max_iterations, random_state, EPSILON);
    kmeans.fit(sample_data);


    Array data_untrained = {
        0.1f, 0.2f,
        0.2f, 0.2f,
        -0.3f, 0.1f,

        8.0f, 7.5f,
        6.8f, 7.1f,
        9.1f, 8.0f,

        2.7f, 3.0f,
        3.1f, 3.7f,
        2.8f, 3.1f,

        7.3f, 7.5f,
        6.3f, 7.1f,
        9.7f, 6.4f,
    };

    data_untrained.resize(12, 2);


    // Ensure that the data was clustered into 4 distinct groups
    // Predictions is a single dimension array of 12 values representing the labels of each group
    Array predictions = kmeans.predict(data_untrained);

    // Resize to 4 x 3
    predictions.resize(4, 3);

    // Showcase that the rows are all equal
    predictions.__debug_print();

    return 0;
}