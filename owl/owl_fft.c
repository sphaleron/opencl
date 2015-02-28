#include "owl_fft.h"
#include "owl_opencl.h"
#include "owl_errno.h"

#include <stdlib.h>

// Kernel sources
#include "owl_fft.cl.hex"

owl_fft_handle* owl_fft_init(owl_opencl_handle* opencl) {
   cl_int opencl_error;

   owl_fft_handle* handle = calloc(sizeof(owl_fft_handle), 1);
   if (handle == NULL)
      OWL_ERROR_NULL("out of memory", OWL_NOMEM);

   handle->opencl = opencl;

   // Passing &owl_fft_cl does not work, some fiddling required here.
   const char* source = owl_fft_cl;
   handle->program = clCreateProgramWithSource(opencl->context, 1, &source, &owl_fft_cl_len, &opencl_error);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   // TODO think about the build options
   // Should the kernel building be postponed to the planning phase, in case it depends on the transfer size?
   opencl_error = clBuildProgram(handle->program, 1, opencl->devices, "-cl-unsafe-math-optimizations", NULL, NULL);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   handle->fft_kernel = clCreateKernel(handle->program, "owl_fft_radix2", &opencl_error);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   return handle;
}


void owl_fft_free(owl_fft_handle* handle) {
   cl_int opencl_error;

   opencl_error = clReleaseKernel(handle->fft_kernel);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   opencl_error = clReleaseProgram(handle->program);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   free(handle);
}


owl_fft_complex_workspace* owl_fft_complex_workspace_alloc(owl_fft_handle* handle, size_t n) {
   cl_int opencl_error;
   const size_t buffer_size = 2*n*sizeof(cl_float);

   owl_fft_complex_workspace* workspace = calloc(sizeof(owl_fft_complex_workspace), 1);
   if (workspace == NULL)
      OWL_ERROR_NULL("out of memory", OWL_NOMEM);

   workspace->n = n;
   workspace->buffers[0] = clCreateBuffer(handle->opencl->context, CL_MEM_READ_WRITE, buffer_size, NULL, &opencl_error);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   workspace->buffers[1] = clCreateBuffer(handle->opencl->context, CL_MEM_READ_WRITE, buffer_size, NULL, &opencl_error);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_NULL(NULL, opencl_error);

   return workspace;
}

void owl_fft_complex_workspace_free(owl_fft_complex_workspace* workspace) {
   cl_int opencl_error;

   opencl_error = clReleaseMemObject(workspace->buffers[0]);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   opencl_error = clReleaseMemObject(workspace->buffers[1]);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR_VOID(NULL, opencl_error);

   free(workspace);
}


int owl_fft_complex_forward (owl_fft_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace) {
   cl_int opencl_error;
   owl_opencl_handle* opencl = handle->opencl;
   cl_uint param = 1;
   unsigned int k = 0;
   if (n > workspace->n)
      return 1;
   if (stride != 1)
      return 2;

   const size_t buffer_size = 2*n*sizeof(cl_float);
   // Keep it simple for now, although MapBuffer etc. might sometimes be more optimal.
   opencl_error = clEnqueueWriteBuffer(opencl->queues[0], workspace->buffers[0], CL_FALSE, 0, buffer_size,
                                       data, 0, NULL, NULL);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR(NULL, opencl_error);

   size_t global_work_size = n >> 1;
   // TODO fix this somehow!
   size_t local_work_size  = global_work_size;

   while (param < n) {
      opencl_error = clSetKernelArg(handle->fft_kernel, 0, sizeof(cl_mem),  (void*)&workspace->buffers[k & 1]);
      if (opencl_error != CL_SUCCESS)
         OWL_ERROR(NULL, opencl_error);
      opencl_error = clSetKernelArg(handle->fft_kernel, 1, sizeof(cl_mem),  (void*)&workspace->buffers[(k + 1) & 1]);
      if (opencl_error != CL_SUCCESS)
         OWL_ERROR(NULL, opencl_error);
      opencl_error = clSetKernelArg(handle->fft_kernel, 2, sizeof(cl_uint), (void*)&param);
      if (opencl_error != CL_SUCCESS)
         OWL_ERROR(NULL, opencl_error);

      opencl_error = clEnqueueNDRangeKernel(opencl->queues[0], handle->fft_kernel, 1, NULL,
                                          &global_work_size, &local_work_size, 0, NULL, NULL);
      if (opencl_error != CL_SUCCESS)
         OWL_ERROR(NULL, opencl_error);

      k += 1;
      param = 1 << k;
   }

   opencl_error = clEnqueueReadBuffer(opencl->queues[0], workspace->buffers[k & 1], CL_TRUE, 0, buffer_size,
                                       data, 0, NULL, NULL);
   if (opencl_error != CL_SUCCESS)
      OWL_ERROR(NULL, opencl_error);

   return 0;
}
