
#ifndef CLUSTERS_H
#define CLUSTERS_H

#include <quickcluster/linearalgebra/array.h>

#include <stdlib.h>

// Creates 2D clusters where centers is a N x 2 array representing each center
Array<float> create_blobs(const Array<float> &centers, size_t points_per_cluster);

#endif