#ifndef OPENCL_UTILS_H
#define OPENCL_UTILS_H

#include <stdbool.h>
#include <CL/cl.h>


#define OPENCL_CHECK(x) \
if (x != CL_SUCCESS) {\
  _display_opencl_error(x);\
  return false;\
}

typedef struct {
  uint32_t      n_devices;
  cl_device_id* devices;
  cl_context    context;
  cl_command_queue* queues;
} opencl_handle;


/**
 * Discovers all OpenCL supported devices of the given type on the system.
 * Only a list of devices and their total number is stored.
 * @param handle OpenCL handle to fill with data.
 * @param type   Device types included, usually one of CL_DEVICE_TYPE_{CPU,GPU,ALL}.
 * @return True on success, false on failure.
 */
bool opencl_discover(opencl_handle* handle, cl_device_type type);

/**
 * Setup an OpenCL context and command queues for the first n_devices in the handle,
 * provided by an earlier opencl_discover.
 * @param handle OpenCL structure.
 * @param n_devices Length of the device list.
 * @return True on success, false on failure.
 */
bool opencl_setup(opencl_handle* handle, int n_devices);

/**
 * Frees all memory and release all OpenCL structures allocated inside the given handle,
 * but not the memory for the handle itself.
 * @param handle OpenCL handle to be freed.
 * @return True on success, false on failure.
 */
bool opencl_free(opencl_handle* handle);

/**
 * Internal use only, print an informative error message when an OpenCL API call
 * returns an error.
 * @param x error code
 */
void _display_opencl_error(cl_uint x);

#endif