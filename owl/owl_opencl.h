#ifndef OWL_OPENCL_H
#define OWL_OPENCL_H

#include <CL/cl.h>

typedef struct {
   cl_context context;
   cl_uint dev_n;
   cl_device_id* devices;
   cl_command_queue* queues;
} owl_opencl_handle;

owl_opencl_handle* owl_opencl_init(cl_context context, cl_command_queue queue);

void owl_opencl_free(owl_opencl_handle* handle);

#endif
