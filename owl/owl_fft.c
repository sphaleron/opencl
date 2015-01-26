#include "owl_fft.h"

#include "../opencl_utils.h"

#include <CL/cl.h>
#include <stdlib.h>

// Kernel sources
#include "owl_fft.cl.hex"


owl_fft_opencl_handle* owl_fft_opencl_init(cl_context context) {
   cl_int opencl_error;

   owl_fft_opencl_handle* handle = calloc(sizeof(owl_fft_opencl_handle), 1);
   if (handle == NULL)
      return NULL;

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
   handle->queues[0] = clCreateCommandQueue(handle->context, handle->devices[0], 0, &opencl_error);
   // TODO error checking

   // Passing &owl_fft_cl does not work, some fiddling required here.
   const char* source = owl_fft_cl;
   handle->program = clCreateProgramWithSource(handle->context, 1, &source, &owl_fft_cl_len, &opencl_error);
   OPENCL_CHECK(opencl_error);
   // TODO error checking


   // TODO think about the build options
   // Should the kernel building be postponed to the planning phase, in case it depends on the transfer size?
   opencl_error = clBuildProgram(handle->program, 1, handle->devices, "-cl-unsafe-math-optimizations", NULL, NULL);
   // TODO error checking
  OPENCL_CHECK(opencl_error);

   handle->fft_kernel = clCreateKernel(handle->program, "owl_fft_radix2", &opencl_error);
   // TODO error checking
  OPENCL_CHECK(opencl_error);

   return handle;
}



void owl_fft_opencl_free(owl_fft_opencl_handle* handle) {
   cl_int opencl_error;

   opencl_error = clReleaseKernel(handle->fft_kernel);
   // TODO error checking

   opencl_error = clReleaseProgram(handle->program);
   // TODO error checking

   opencl_error = clReleaseCommandQueue(handle->queues[0]);
   // TODO error checking

   free(handle->devices);
   free(handle);
}


owl_fft_complex_workspace* owl_fft_complex_workspace_alloc(owl_fft_opencl_handle* handle, size_t n) {
   cl_int opencl_error;
   const size_t buffer_size = 2*n*sizeof(cl_float);

   owl_fft_complex_workspace* workspace = calloc(sizeof(owl_fft_complex_workspace), 1);
   // TODO output some error message, or set error code
   if (workspace == NULL)
      return NULL;

   workspace->n = n;
   workspace->buffers[0] = clCreateBuffer(handle->context, CL_MEM_READ_WRITE, buffer_size, NULL, &opencl_error);
   workspace->buffers[1] = clCreateBuffer(handle->context, CL_MEM_READ_WRITE, buffer_size, NULL, &opencl_error);
   // TODO error checking

   return workspace;
}

void owl_fft_complex_workspace_free(owl_fft_complex_workspace* workspace) {
   cl_int opencl_error;

   opencl_error = clReleaseMemObject(workspace->buffers[0]);
   opencl_error = clReleaseMemObject(workspace->buffers[1]);
   // TODO error checking
   free(workspace);
}


int owl_fft_complex_forward (owl_fft_opencl_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace) {
   cl_int opencl_error;
   cl_uint param = 1;
   unsigned int k = 0;
   if (n > workspace->n)
      return 1;
   if (stride != 1)
      return 2;

   const size_t buffer_size = 2*n*sizeof(cl_float);
   // Keep it simple for now, although MapBuffer etc. might sometimes be more optimal.
   opencl_error = clEnqueueWriteBuffer(handle->queues[0], workspace->buffers[0], CL_FALSE, 0, buffer_size,
                                       data, 0, NULL, NULL);

   size_t global_work_size = n >> 1;
   // TODO fix this somehow!
   size_t local_work_size  = global_work_size;

   while (param < n) {
      opencl_error = clSetKernelArg(handle->fft_kernel, 0, sizeof(cl_mem),  (void*)&workspace->buffers[k & 1]);
      opencl_error = clSetKernelArg(handle->fft_kernel, 1, sizeof(cl_mem),  (void*)&workspace->buffers[(k + 1) & 1]);
      opencl_error = clSetKernelArg(handle->fft_kernel, 2, sizeof(cl_uint), (void*)&param);

      opencl_error = clEnqueueNDRangeKernel(handle->queues[0], handle->fft_kernel, 1, NULL,
                                          &global_work_size, &local_work_size, 0, NULL, NULL);
      k += 1;
      param = 1 << k;
   }

   opencl_error = clEnqueueReadBuffer(handle->queues[0], workspace->buffers[k & 1], CL_TRUE, 0, buffer_size,
                                       data, 0, NULL, NULL);


   return 0;
}
