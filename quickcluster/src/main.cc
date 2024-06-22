
#include <quickcluster/linearalgebra/operations.h>
#include <quickcluster/linearalgebra/distance.h>
#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/benchmark/stopwatch.h>
#include <quickcluster/datasets/clusters.h>
#include <quickcluster/kmeans/kmeans.h>
#include <quickcluster/device/metal.h>

#include <iostream>
#include <vector>
#include <random>

int main() {

    time_t t;
    time(&t);
    
    Array centers = {
        10.0f, 5.0f,
        6.0f, 6.0f,
        25.0f, 16.0f,
    };

    centers.resize(3, 2);

    // 12 million rows total
    Array data = create_blobs(centers, 3000000);
    centers.__debug_print();

    DeviceHandle handle;
    metal_init_device(&handle);

    gpu_device device;
    metal_find_device(&device);

    StopWatch sw;
    sw.start();

    KMeans kmeans(3, 10, t, EPSILON, handle);
    kmeans.fit(data);
 
    sw.stop();

    printf("Kmeans took %dms\n", (int)sw.elapsed());
    kmeans.centroids().__debug_print();


    metal_release_device(&handle);

    return 0;
}