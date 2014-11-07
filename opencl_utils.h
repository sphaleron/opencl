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
 * NOTE: redesign at some point, better idea would be to communicate via an OpenCL context.
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
 * Load OpenCL kernel source from file, and create a program from it in the given context.
 * @param filename File containing the source code.
 * @param context OpenCL context in which the program will be executed.
 * @param program Will be updated to contain the program.
 */
bool opencl_load_source_file(const char* filename, cl_context context, cl_program* program);

/**
 * Build the OpenCL program with given build options and create kernel objects
 * for all kernels contained in the program.
 * @param program OpenCL program containing the kernels.
 * @param options Build options.
 * @param verbose Print the number of kernels and build information.
 * @param kernels A newly allocated array containing all kernels in the program. Otherwise NULL.
 * @return The number of kernels, or -1 on failure.
 */
cl_int opencl_build_kernels(cl_program program, const char* options, bool verbose, cl_kernel** kernels);


/**
 * Find the kernel with the given name in the list of created kernels.
 * @param kname The kernel name.
 * @param kernels Array of kernels.
 * @param n_kernels Length of the kernels array.
 * @return Kernel object with the given name, or NULL in case of not found or errors.
 */
cl_kernel opencl_get_named_kernel(const char* kname, cl_kernel* kernels, cl_uint n_kernels);

/**
 * Internal use only, print an informative error message when an OpenCL API call
 * returns an error.
 * TODO: make this really internal to the library, let the clients worry about errors themselves?
 * @param x error code
 */
void _display_opencl_error(cl_uint x);

#endif