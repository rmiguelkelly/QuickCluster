
#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/linearalgebra/distance.h>
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/benchmark/stopwatch.h>
#include <quickcluster/datasets/clusters.h>

#include <iostream>

int main() {

    Array centers = { 
        10.0f, 5.0f,
        6.0f, 6.0f,
        25.0f, 16.0f,
    };

    centers.resize(3, 2);

    // 12 million rows total
    Array data = create_blobs(centers, 4000000);

    time_t t;
    time(&t);

    StopWatch sw;
    sw.start();

    KMeans kmeans(3, 10, t);
    kmeans.fit(data);

    sw.stop();

    printf("Kmeans took %dms\n", (int)sw.elapsed());

    return 0;
}