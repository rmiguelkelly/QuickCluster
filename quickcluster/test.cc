
#include <quickcluster/device/cuda.h>
#include <quickcluster/device/common.h>
#include <stdio.h>

int main() {

  struct gpu_device device;
  int r = cuda_find_device(&device);
  printf(">> %d\n", r);

  printf(">> %s\n", device.name);

  return 0;
}
