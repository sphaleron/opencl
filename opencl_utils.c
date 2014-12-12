#include "opencl_utils.h"

#include <CL/cl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define MAX_BUILD_LOG_SIZE 2048
#define MAX_KERNEL_NAME_SIZE 256

// This is anything but thread-safe.
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


bool opencl_setup(opencl_handle* handle, int n_devices) {
  handle->context = clCreateContext(NULL, n_devices, handle->devices, NULL, NULL, &opencl_error);
  OPENCL_CHECK(opencl_error);
  handle->n_devices = n_devices;

  handle->queues = (cl_command_queue*) malloc(n_devices * sizeof(cl_command_queue));
  if (handle->queues == NULL) {
    printf("Out of memory!\n");
    return false;
  }

  for (uint_fast32_t dev_loop = 0; dev_loop < n_devices; dev_loop++) {
    handle->queues[dev_loop] = clCreateCommandQueue(handle->context, handle->devices[dev_loop], 0, &opencl_error);
    OPENCL_CHECK(opencl_error);
  }

  return true;
}




bool opencl_free(opencl_handle* handle) {
   for (uint_fast32_t kloop = 0; kloop < handle->n_kernels; kloop++) {
      opencl_error = clReleaseKernel(handle->kernels[kloop]);
      OPENCL_CHECK(opencl_error);
   }
   free(handle->kernels);

   if (handle->context != NULL) {
      opencl_error = clReleaseContext(handle->context);
      OPENCL_CHECK(opencl_error);
   }
   for (uint_fast32_t dev_loop = 0; dev_loop < handle->n_devices; dev_loop++) {
      if (handle->queues != NULL && handle->queues[dev_loop] != NULL) {
         opencl_error = clReleaseCommandQueue(handle->queues[dev_loop]);
         OPENCL_CHECK(opencl_error);
      }
   }
  free(handle->queues);
  free(handle->devices);

  return true;
}


bool opencl_load_source_file(const char* filename, cl_context context, cl_program* program) {
  FILE* source_fid;
  char* source = NULL;
  long source_length;

  source_fid = fopen(filename, "r");
  if (source_fid == NULL) {
    printf("Opening kernel source file %s failed!\n", filename);
    return false;
  }
  fseek(source_fid, 0, SEEK_END);
  source_length = ftell(source_fid);
  fseek(source_fid, 0, SEEK_SET);

  // Add a byte for null character:
  source = (char*) malloc(source_length + 1);
  if (source == NULL) {
    printf("Out of memory!\n");
    return false;
  }
  fread(source, source_length, 1, source_fid);
  source[source_length] = '\0';
  fclose(source_fid);

  // File reading is now done, let's create a program:
  *program = clCreateProgramWithSource(context, 1, (const char**) &source, NULL, &opencl_error);
  OPENCL_CHECK(opencl_error);

  free(source);

  return true;
}

cl_int opencl_build_kernels(opencl_handle* handle, cl_program program, const char* options, bool verbose) {
   size_t n_kernels;
   cl_uint n_created;

   opencl_error = clBuildProgram(program, 0, NULL, options, NULL, NULL);
   if (opencl_error != CL_SUCCESS) {
      printf("Build error! Return code %d.\n", opencl_error);
      // Should we instead  try to get also build log to diagnose the error? Some platforms provide it
      // for failed builds, some do not.
      return -1;
   }

//   if (verbose) {
//      TODO need to figure out device IDs, blaah
//      char* log = (char*) malloc(MAX_BUILD_LOG_SIZE + 1);
//      opencl_error = clGetProgramBuildInfo(program, f, CL_PROGRAM_BUILD_LOG, MAX_BUILD_LOG_SIZE, log, NULL);
//      if (opencl_error != CL_SUCCESS) {
//         printf("Failed to get the build log! Error code %d.\n", opencl_error);
//         return -1;
//      }
//      log[MAX_BUILD_LOG_SIZE] = '\0';
//      printf("Build log for device x: %s\n", log);
//      free(log);
//   }

   opencl_error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &n_kernels, NULL);
   if (opencl_error != CL_SUCCESS) {
      printf("Failed to get the number of kernels! Error code %d.\n", opencl_error);
      return -1;
   }

   handle->kernels = (cl_kernel*) malloc(n_kernels*sizeof(cl_kernel));
   if (handle->kernels == NULL) {
      printf("Out of memory!\n");
      return -1;
   }

   opencl_error = clCreateKernelsInProgram(program, n_kernels, handle->kernels, &n_created);
   if (opencl_error != CL_SUCCESS) {
      printf("Failed to create kernel objects! Error code %d.\n", opencl_error);
      return -1;
   }
   handle->n_kernels = n_created;

   return n_created;
}

cl_kernel opencl_get_named_kernel(opencl_handle* handle, const char* kname) {
   size_t name_length;
   char name[MAX_KERNEL_NAME_SIZE];
   for (uint_fast32_t kloop = 0; kloop < handle->n_kernels; kloop++) {
      opencl_error = clGetKernelInfo(handle->kernels[kloop], CL_KERNEL_FUNCTION_NAME, MAX_KERNEL_NAME_SIZE, name, &name_length);
      if (opencl_error != CL_SUCCESS) {
         printf("Failed to get kernel name! Error code %d.\n", opencl_error);
         return NULL;
      }
      if (!strcmp(kname, name))
         return handle->kernels[kloop];
   }
   return NULL;
}



// TODO display a more informative error message: file and line number + error code name
void _display_opencl_error(cl_uint x)
{
  printf("OpenCL error %d!\n", x);
  return;
}
