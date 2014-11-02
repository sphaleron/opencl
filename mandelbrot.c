#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "opencl_utils.h"

typedef struct {
  float x[2];
  float y[2];
  uint32_t dim[2];
  uint32_t max_iter;
} parameters;

static void usage() {
  printf("Usage: mandelbrot [-w width] [-h height] [-x lo,hi] [-y lo,hi] [-o outfile]\n");
  return;
}

static void parameters_init(parameters* params) {
  params->x[0] = -1.5;
  params->x[1] =  0.5;
  params->y[0] = -1.0;
  params->y[1] =  1.0;
  params->dim[0] = 256;
  params->dim[1] = 256;
  params->max_iter = 1000;
  
  return;
}

static bool write_image(const char* outfile, const parameters* params, uint32_t* data) {
  size_t data_size = params->dim[0] * params->dim[1] * sizeof(uint32_t);
  FILE* out_fid = fopen(outfile, "w");
  if (out_fid == NULL) {
    printf("Creating output file '%s' failed!\n", outfile);
    return false;
  }
  fwrite(data, data_size, 1, out_fid);
  fclose(out_fid);

  return true;
}


int main(int argc, char* argv[]) {
  opencl_handle opencl;
  cl_program program;
  cl_kernel mandelbrot_kernel;
  cl_mem data_buffer;
  cl_int opencl_error;
  parameters params;
  uint32_t* image = NULL;
  size_t data_size;
  
  parameters_init(&params);
  
//  usage();
  // read command line parameters
  
  printf("Naive Mandelbrot set generator\n\n");

  if (!opencl_discover(&opencl, CL_DEVICE_TYPE_ALL))
    return 1;

  // Make it single device now.
  if (!opencl_setup(&opencl, 1))
    return 1;

  // Load the kernel from a source file
  if (!opencl_load_source_file("mandelbrot.cl", opencl.context, &program))
    return false;
  
  // And finally build the program for all devices in the context:
  // TODO think of options, wrap this into a utility function to to also get build info easily
  opencl_error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  OPENCL_CHECK(opencl_error);
  
  mandelbrot_kernel = clCreateKernel(program, "mandelbrot", &opencl_error);
  OPENCL_CHECK(opencl_error);
  
  data_size = params.dim[0]*params.dim[1]*sizeof(uint32_t);
  image = (uint32_t*) calloc(1, data_size);
  if (image == NULL) {
    printf("Out of memory!\n");
    return 1;
  }
  
  // Create a buffer. Try first with the usual method, without mappings.
  data_buffer = clCreateBuffer(opencl.context, CL_MEM_WRITE_ONLY, data_size, NULL, &opencl_error);
  OPENCL_CHECK(opencl_error);
  
  // Set kernel arguments
  clSetKernelArg(mandelbrot_kernel, 0, sizeof(cl_mem), (void *)&data_buffer);
  
  const size_t global_size[] = {params.dim[0], params.dim[1]};
  opencl_error = clEnqueueNDRangeKernel(opencl.queues[0], mandelbrot_kernel, 2,
                                        NULL, global_size, NULL, 0, NULL, NULL);

  opencl_error = clEnqueueReadBuffer(opencl.queues[0], data_buffer, CL_TRUE, 0, data_size,
                                     (void*) image, 0, NULL, NULL);
  OPENCL_CHECK(opencl_error);

  // Hard coded output file now, until we get to options.
  if (!write_image("mandelbrot.raw", &params, image))
    return 1;
  
  opencl_error = clReleaseMemObject(data_buffer);
  OPENCL_CHECK(opencl_error);

  opencl_error = clReleaseKernel(mandelbrot_kernel);
  OPENCL_CHECK(opencl_error);
  
  opencl_error = clReleaseProgram(program);
  OPENCL_CHECK(opencl_error);
  
  if (!opencl_free(&opencl))
    return 1;
  
  return 0;
}