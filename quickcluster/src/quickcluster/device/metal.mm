
#import <quickcluster/device/metal.h>
#import <quickcluster/device/MetalDeviceManager.h>

#import <CoreFoundation/CoreFoundation.h>

// C++ Headers
#include <stdio.h>

#define METAL_LIB_NAME @"cluster.metallib"

int metal_init_device(void** handle, const char *metallib_path) {

    id<MTLDevice> mtlDevice = [MetalDeviceManager defaultDevice];

    if (mtlDevice == nil) {
        return -1;
    }

    NSString* metallib = nil;

    if (metallib_path != NULL) {
        metallib = [NSString stringWithCString: metallib_path encoding: NSUTF8StringEncoding];
        metallib = [metallib stringByAppendingPathComponent:METAL_LIB_NAME];
    }

    // Initialize
    MetalDeviceManager *deviceManager = [[MetalDeviceManager alloc] initWithDevice: mtlDevice metallib: metallib];
    void *object = (__bridge void*)deviceManager;

    // Increase the reference count by 1
    CFRetain(object);

    // Set the handle
    *handle = object;

    return 0;
}

int metal_find_device(struct gpu_device *device) {

    // Attempt to get the default GPU
    id<MTLDevice> mtlDevice = [MetalDeviceManager defaultDevice];

    // Return -1 if there is no metal GPU device available
    if (mtlDevice == nil) {
        return -1;
    }

    NSString *deviceName = [mtlDevice name];

    const char *deviceNamecString = [deviceName cStringUsingEncoding:NSUTF8StringEncoding];
    size_t deviceNamecStringLen = [deviceName lengthOfBytesUsingEncoding:NSUTF8StringEncoding];

    device->device_type = DeviceType::DeviceTypeGPUMetal;
    device->id = [mtlDevice registryID];

    // Zero out and set the name
    memset(device->name, 0, deviceNamecStringLen);
    strncpy((char*)device->name, deviceNamecString, deviceNamecStringLen);
    device->name[deviceNamecStringLen] = 0;

    return 0;
}

int metal_compute_nearest_centroids(void* handle, const float *data, const float *centroids, const DataContext *context, unsigned long *results) {

    MetalDeviceManager *deviceManager = (__bridge MetalDeviceManager*)handle;

    if (deviceManager == nil) {
        return -1;
    }
    
    int result = [deviceManager computeNearestCentroids: data centroids: centroids context: context results: results];

    return result;
}

void metal_release_device(void **handle) {

    void *object = *handle;

    // Decrement the reference count to deallocate the object
    CFRelease(object);
}