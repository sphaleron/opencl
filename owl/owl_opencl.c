#include "owl_opencl.h"

owl_opencl_handle* owl_opencl_init(cl_context context, cl_command_queue queue) {
   cl_int opencl_error;

   owl_opencl_handle* handle = calloc(sizeof(owl_opencl_handle), 1);
   if (handle == NULL)
      return NULL;

   opencl_error = clRetainContext(context);
   // TODO error checking

   handle->context = context;
   opencl_error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &handle->dev_n, NULL);
   // TODO error checking

   handle->devices = calloc(sizeof(cl_device_id), handle->dev_n);
   opencl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, handle->dev_n*sizeof(cl_device_id), handle->devices, NULL);
   // TODO error checking

   // Using only one device for now
   handle->queues = calloc(sizeof(cl_command_queue), 1);
   if (handle->queues == NULL)
      return NULL;

   if (queue == NULL)
      handle->queues[0] = clCreateCommandQueue(handle->context, handle->devices[0], 0, &opencl_error);
   else {
      opencl_error = clRetainCommandQueue(queue);
      handle->queues[0] = queue;
   }
   // TODO error checking

   return handle;
}



void owl_opencl_free(owl_opencl_handle* handle) {
   cl_int opencl_error;

   opencl_error = clReleaseCommandQueue(handle->queues[0]);
   // TODO error checking

   opencl_error = clReleaseContext(handle->context);

   free(handle->devices);
   free(handle);
}

