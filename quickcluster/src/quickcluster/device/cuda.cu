
#include <quickcluster/device/cuda.h>

int cuda_find_device(struct gpu_device *device) {

    int cuda_device;
    
    // CUDA result stuff
    cudaError res;

    // Get the device
    res = cudaGetDevice(&cuda_device);

    if (res != cudaError::cudaSuccess) {
        return -1;
    }

    // Get the device properties
    cudaDeviceProp props;
    res = cudaGetDeviceProperties(&props, cuda_device);

    if (res != cudaError::cudaSuccess) {
        return -1;
    }

    device->id = cuda_device;

    int len = strlen(props.name);
    strncpy(device->name, props.name, len);

    return 0;
}


int cuda_device_init(DeviceHandle *handle) {

    // Used to signify that CUDA is to be used
    cuda_device* hndl = new cuda_device();
    *handle = hndl;

    return 0;
}


int cuda_compute_nearest_centroids(DeviceHandle *handle, const float *data, const float *centroids, const DataContext *context, unsigned long *results) {

    return 0;
}


int cuda_device_release(DeviceHandle *handle) {

    // Delete the pointer
    cuda_device *hndl = (cuda_device*)(*handle);
    delete hndl;

    return 0;
}