/*
 * These functions will setup the necessary OpenCL structures
 * for the rest of the library.
 */

#ifndef OWL_OPENCL_H
#define OWL_OPENCL_H

#include <CL/cl.h>

typedef struct {
   cl_context context;
   cl_uint dev_n;
   cl_device_id* devices;
   cl_command_queue* queues;
} owl_opencl_handle;


/**
 * Allocates memory for necessary OpenCL structures and fills it.
 * @param context Existing OpenCL context where all work should take place.
 * @param queue OpenCL command queue that should be used for running kernels.
 *              If the queue is NULL, a new one will be created.
 * @return A pointer to an allocated and filled owl_opencl_handle.
 */
owl_opencl_handle* owl_opencl_init(cl_context context, cl_command_queue queue);

/**
 * Frees the OpenCL structures allocated in owl_opencl_init.
 * @param handle owl_opencl_handle structure previously allocated.
 */
void owl_opencl_free(owl_opencl_handle* handle);

#endif
