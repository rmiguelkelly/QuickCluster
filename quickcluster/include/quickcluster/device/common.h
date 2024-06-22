#ifndef COMMON_H
#define COMMON_H

#define DeviceHandle void*

/// @brief Type pf GPU supported by this program
enum DeviceType {
    DeviceTypeCPU = 0x0,
    DeviceTypeGPUMetal = 0x1,
    DeviceTypeGPUCuda = 0x2,
};

// Used to define a GPU on the computer
struct gpu_device {
    DeviceType device_type;
    int id;
    char name[64];
};

/// @brief Used to pass contextual information to the GPU
struct DataContext {
    unsigned long rows;
    unsigned long cols;
    unsigned long k;
};

#endif