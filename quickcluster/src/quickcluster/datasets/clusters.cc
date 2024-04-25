

#include <quickcluster/datasets/clusters.h>

#include <math.h>
#include <vector>

using std::vector;

inline float randf() {
    return (rand() % 100) / 100.0;
}

Array<float> create_blobs(const Array<float> &centers, size_t points_per_cluster) {

    vector<float> points;

    for (size_t i = 0; i < centers.rows(); i++) { 
        
        auto row = centers.row(i);
        
        float center_x = row[0];
        float center_y = row[1];

        for (size_t j = 0; j < points_per_cluster; j++) {
            
            float angle = 2 * M_PI * randf();
            float len = randf() + 10.0;

            float x = center_x + cos(angle) * len;
            float y = center_y + sin(angle) * len;

            points.push_back(x);
            points.push_back(y);
        }
    }

    return Array<float>(points.data(), centers.rows() * points_per_cluster, 2, true);
}