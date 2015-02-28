#include "owl_opencl.h"
#include "owl_errno.h"

owl_opencl_handle* owl_opencl_init(cl_context context, cl_command_queue queue) {
   cl_int opencl_error;

   owl_opencl_handle* handle = calloc(sizeof(owl_opencl_handle), 1);
   if (handle == NULL)
      OWL_ERROR_NULL("out of memory", OWL_NOMEM);

   opencl_error = clRetainContext(context);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   handle->context = context;
   opencl_error = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &handle->dev_n, NULL);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   handle->devices = calloc(sizeof(cl_device_id), handle->dev_n);
   opencl_error = clGetContextInfo(context, CL_CONTEXT_DEVICES, handle->dev_n*sizeof(cl_device_id), handle->devices, NULL);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   // Using only one device for now
   handle->queues = calloc(sizeof(cl_command_queue), 1);
   if (handle->queues == NULL)
      OWL_ERROR_NULL("out of memory", OWL_NOMEM);

   if (queue == NULL)
      handle->queues[0] = clCreateCommandQueue(handle->context, handle->devices[0], 0, &opencl_error);
   else {
      opencl_error = clRetainCommandQueue(queue);
      handle->queues[0] = queue;
   }
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   return handle;
}



void owl_opencl_free(owl_opencl_handle* handle) {
   cl_int opencl_error;

   opencl_error = clReleaseCommandQueue(handle->queues[0]);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   opencl_error = clReleaseContext(handle->context);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   free(handle->devices);
   free(handle);
}
