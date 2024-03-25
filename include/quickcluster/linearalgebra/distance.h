#ifndef DISTANCE_H
#define DISTANCE_H

#include <quickcluster/linearalgebra/array.h>

#include <math.h>

// Returns the euclidean distance between two single dimension arrays
float distance_euclidean(const Array<float> &point1, const Array<float> &point2);

#endif