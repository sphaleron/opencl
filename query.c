#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "opencl_utils.h"

#define MAX_INFO_SIZE 1024

static cl_int opencl_error;

static bool query_platform(cl_platform_id platform);
static bool query_device(cl_device_id device);


int main(int argc, char* argv[])
{
  opencl_handle opencl;
  cl_platform_id platform, old_platform;
  cl_device_id device;
  memset(&opencl, 0, sizeof(opencl_handle));
  
  
  printf("== OpenCL device query exercise. ==\n\n");


  if (!opencl_discover(&opencl, CL_DEVICE_TYPE_ALL))
     return 1;

  // We happily threw away all platform information above, and here rely on the fact
  // that devices belonging to the same platform are in one block in opencl.devices.
  for (uint_fast32_t dev_loop = 0; dev_loop < opencl.n_devices; dev_loop++) {
    device = opencl.devices[dev_loop];
    opencl_error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
    OPENCL_CHECK(opencl_error);

    // If we encountered a new platform here
    if (dev_loop == 0 || platform != old_platform) {
      if (!query_platform(platform))
        return 1;
      printf("Devices:\n");
      printf("--------\n");
    }

    if (!query_device(device))
      return 1;
    printf("\n");

    old_platform = platform;
  }
  
  if (!opencl_free(&opencl))
    return 1;
    
  return 0;
}




static bool query_platform(cl_platform_id platform) {
  char info_str[MAX_INFO_SIZE];
  size_t data_size;

  opencl_error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, MAX_INFO_SIZE, info_str, &data_size);
  OPENCL_CHECK(opencl_error);
  printf("Platform name  : %s\n", info_str);
  opencl_error = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, MAX_INFO_SIZE, info_str, &data_size);
  OPENCL_CHECK(opencl_error);
  printf("OpenCL version : %s\n", info_str);
  
  return true;
}
    
  
static bool query_device(cl_device_id device) {
  char info_str[MAX_INFO_SIZE];
  cl_device_type type;
  size_t data_size;
 
  opencl_error =  clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_SIZE, info_str, &data_size);
  OPENCL_CHECK(opencl_error);
  printf("  Device name  : %s\n", info_str);
  opencl_error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, &data_size);
  OPENCL_CHECK(opencl_error);
  if (type == CL_DEVICE_TYPE_CPU)
    snprintf(info_str, MAX_INFO_SIZE, "CPU");
  else if (type == CL_DEVICE_TYPE_GPU)
    snprintf(info_str, MAX_INFO_SIZE, "GPU");
  else
    snprintf(info_str, MAX_INFO_SIZE, "other");
  printf("  Device type  : %s\n", info_str);
  
  return true;
}
