#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <CL/cl.h>

#include "opencl_utils.h"

static cl_int opencl_error;


bool opencl_discover(opencl_handle* handle, cl_device_type type) {
  cl_platform_id* platforms = NULL;
  cl_device_id* devices = NULL;
  cl_uint n_platforms = 0;
  cl_uint n_devices   = 0;
  uint32_t total_devices = 0;
  
  opencl_error = clGetPlatformIDs(0, NULL, &n_platforms);
  OPENCL_CHECK(opencl_error);
  if (n_platforms == 0) {
    printf("No platforms found!\n");
    return false;
  }

  platforms = (cl_platform_id*) malloc(n_platforms * sizeof(cl_platform_id));
  if (platforms == NULL) {
    printf("Out of memory!\n");
    return false;
  }
  
  // Count all devices and reserve enough memory for IDs
  opencl_error = clGetPlatformIDs(n_platforms, platforms, NULL);
  OPENCL_CHECK(opencl_error);
  for (uint_fast32_t pfm_loop = 0; pfm_loop < n_platforms; pfm_loop++) {
    cl_platform_id platform = platforms[pfm_loop];
    n_devices = 0;
    opencl_error = clGetDeviceIDs(platform, type, 0, NULL, &n_devices);
    OPENCL_CHECK(opencl_error);
    total_devices += n_devices;
  }
  handle->devices = (cl_device_id*) malloc(total_devices * sizeof(cl_device_id));
  if (handle->devices == NULL) {
    printf("Out of memory!\n");
    return false;
  }
  handle->n_devices = total_devices;

  // Enumerate all devices in the handle
  devices = handle->devices;
  total_devices = 0;
  for (uint_fast32_t pfm_loop = 0; pfm_loop < n_platforms; pfm_loop++) {
    cl_platform_id platform = platforms[pfm_loop];
    opencl_error = clGetDeviceIDs(platform, type, handle->n_devices - total_devices, devices, &n_devices);
    OPENCL_CHECK(opencl_error);
    devices += n_devices;
    total_devices += n_devices;
  }
  free(platforms);
  
  return true;
}



bool opencl_setup(opencl_handle* handle, bool separate_contexts) {
  if (separate_contexts) {
    handle->contexts = (cl_context*) malloc(handle->n_devices * sizeof(cl_context));
    if (handle->contexts = NULL) {
      printf("Out of memory!\n");
      return false;
    }
    handle->n_contexts = handle->n_devices;
    for (uint_fast32_t dev_loop = 0; dev_loop < handle->n_devices; dev_loop++) {
      handle->contexts[dev_loop] = clCreateContext(NULL, 1, &handle->devices[dev_loop],
                                                   NULL, NULL, &opencl_error);
      OPENCL_CHECK(opencl_error);
    }
  } else {
    handle->contexts = (cl_context*) malloc(sizeof(cl_context));
    if (handle->contexts = NULL) {
      printf("Out of memory!\n");
      return false;
    }
    handle->n_contexts = 1;
    handle->contexts[0] = clCreateContext(NULL, handle->n_devices, handle->devices,
                                                  NULL, NULL, &opencl_error);
    OPENCL_CHECK(opencl_error);
  }
  return true;
}




bool opencl_free(opencl_handle* handle) {
  for (uint_fast32_t context_loop = 0; context_loop < handle->n_contexts; context_loop++) {
    if (handle->contexts[context_loop] != NULL)
      clReleaseContext(handle->contexts[context_loop]);
  }
  free(handle->contexts);
  
  free(handle->devices);
  
  return true;
}


// TODO display a more informative error message: file and line number + error code name
void _display_opencl_error(cl_uint x)
{
  printf("OpenCL error %d!\n", x);
  return;
}
