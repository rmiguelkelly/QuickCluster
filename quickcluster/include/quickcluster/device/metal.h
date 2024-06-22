
#ifndef METAL_H
#define METAL_H

#include <quickcluster/device/common.h>

/// @brief Initialize the metal device manager
/// @param handle a raw pointer to keep the handle alive
/// @return 0 if successful
int metal_init_device(DeviceHandle *handle, const char *metallib_path = nullptr);

/// @brief Method to retrieve the default GPU for MacOS
/// @param handle The internal device manager handle
/// @param device Reference to a `metal_device` struct that will be written to
/// @return 0 if successful
int metal_find_device(struct gpu_device *device);

/// @brief Method to retrieve the default GPU for MacOS
/// @param handle The internal device manager handle
/// @param data The data to cluster the model
/// @param centroids The centroids to cluster against
/// @param context Dimensional information
/// @return 0 if successful
int metal_compute_nearest_centroids(DeviceHandle handle, const float *data, const float *centroids, const DataContext *context, unsigned long *results);

/// @brief Releases the internal metal device manager reference
/// @param handle 
void metal_release_device(DeviceHandle *handle);


#endif