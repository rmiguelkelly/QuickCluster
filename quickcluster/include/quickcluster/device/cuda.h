
#ifndef CUDA_H
#define CUDA_H

#include <quickcluster/device/common.h>

#include <cstring>
#include <cstdlib>

#ifdef __cpluplus
extern "C" {
#endif

// Used to represent a device handle for CUDA
class cuda_device { };

/// @brief Attempts to find a GPU on the computer
/// @param device Device struct to write to
/// @return success or not
int cuda_find_device(struct gpu_device *device);


/// @brief Initiates the handle 
/// @param handle 
/// @return success or not
int cuda_device_init(DeviceHandle *handle);

/// @brief Method to retrieve the default GPU on a device with CUDA
/// @param handle The internal device manager handle
/// @param data The data to cluster the model
/// @param centroids The centroids to cluster against
/// @param context Dimensional information
/// @return 0 if successful
int cuda_compute_nearest_centroids(DeviceHandle *handle, const float *data, const float *centroids, const DataContext *context, unsigned long *results);

/// @brief Releases the internal CUDA device handle
/// @param handle 
int cuda_device_release(DeviceHandle *handle);

#ifdef __cpluplus
extern "C" {
#endif

#endif