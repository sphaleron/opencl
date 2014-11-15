#define _GNU_SOURCE // for asprintf

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "opencl_utils.h"

typedef struct {
   cl_float x[2];
   cl_float y[2];
   size_t dim[2];
   cl_uint max_iter;
   char* outfile;
} parameters;

static void debug_print_parameters(const parameters* param);


static void usage(FILE* stream) {
   fprintf(stream, "Usage: mandelbrot [-w width] [-h height] [-x lo:hi] [-y lo:hi] [-o outfile]\n");
   fprintf(stream, "                  [-m max_iter] [-d]\n");
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
   asprintf(&params->outfile, "mandelbrot.raw");
   return;
}

static bool write_image(const parameters* params, uint32_t* data) {
   size_t data_size = params->dim[0] * params->dim[1] * sizeof(uint32_t);
   FILE* out_fid = fopen(params->outfile, "w");
   if (out_fid == NULL) {
      printf("Creating output file '%s' failed!\n", params->outfile);
      return false;
   }
   fwrite(data, data_size, 1, out_fid);
   fclose(out_fid);

   return true;
}


int main(int argc, char* argv[]) {
   opencl_handle opencl;
   cl_program program;
   cl_kernel* kernels = NULL;
   cl_int n_kernels;
   cl_kernel mandelbrot_kernel;
   cl_mem data_buffer;
   cl_int opencl_error;
   parameters params;
   uint32_t* image = NULL;
   size_t data_size;

   parameters_init(&params);

   // read command line parameters
   char opt;
   while ( (opt = getopt(argc, argv, "w:h:x:y:o:m:d")) != -1) {
      switch(opt) {
         case 'w':
            params.dim[0] = atoi(optarg);
            break;
         case 'h':
            params.dim[1] = atoi(optarg);
            break;
         case 'x':
            params.x[0] = strtof(strsep(&optarg, ":"), NULL);
            params.x[1] = strtof(strsep(&optarg, ":"), NULL);
            break;
         case 'y':
            params.y[0] = strtof(strsep(&optarg, ":"), NULL);
            params.y[1] = strtof(strsep(&optarg, ":"), NULL);
            break;
         case 'o':
            free(params.outfile);
            params.outfile = strdup(optarg);
            break;
         case 'm':
            params.max_iter = atoi(optarg);
            break;
         case 'd':
            fprintf(stderr, "double precision not yet implemented\n");
            break;
         default:
            usage(stderr);
            return EXIT_FAILURE;
      }
   }
   // debug_print_parameters(&params);

   printf("Naive Mandelbrot set generator\n\n");

   if (!opencl_discover(&opencl, CL_DEVICE_TYPE_ALL))
      return EXIT_FAILURE;

   // Make it single device now.
   if (!opencl_setup(&opencl, 1))
      return EXIT_FAILURE;

   // Load the kernel from a source file
   if (!opencl_load_source_file("mandelbrot.cl", opencl.context, &program))
         return EXIT_FAILURE;

   char* options = NULL;
   if (asprintf(&options, "-DMAX_ITER=%d", params.max_iter) < 0)
            return EXIT_FAILURE;
   n_kernels = opencl_build_kernels(program, options, false, &kernels);
   if (n_kernels < 0)
      return EXIT_FAILURE;
   free(options);

   // This is getting silly, but I just want to test all features:
   mandelbrot_kernel = opencl_get_named_kernel("mandelbrot", kernels, n_kernels);
   if (mandelbrot_kernel == NULL)
      return EXIT_FAILURE;

   data_size = params.dim[0]*params.dim[1]*sizeof(uint32_t);
   image = (uint32_t*) calloc(1, data_size);
   if (image == NULL) {
      printf("Out of memory!\n");
      return EXIT_FAILURE;
   }

   // Create a buffer. Try first with the usual method, without mappings.
   data_buffer = clCreateBuffer(opencl.context, CL_MEM_WRITE_ONLY, data_size, NULL, &opencl_error);
   OPENCL_CHECK(opencl_error);

   // Set kernel arguments
   clSetKernelArg(mandelbrot_kernel, 0, sizeof(cl_mem), (void *)&data_buffer);
   clSetKernelArg(mandelbrot_kernel, 1, sizeof(cl_float), (void *)&params.x[0]);
   clSetKernelArg(mandelbrot_kernel, 2, sizeof(cl_float), (void *)&params.x[1]);
   clSetKernelArg(mandelbrot_kernel, 3, sizeof(cl_float), (void *)&params.y[0]);
   clSetKernelArg(mandelbrot_kernel, 4, sizeof(cl_float), (void *)&params.y[1]);

   opencl_error = clEnqueueNDRangeKernel(opencl.queues[0], mandelbrot_kernel, 2,
                                          NULL, params.dim, NULL, 0, NULL, NULL);

   opencl_error = clEnqueueReadBuffer(opencl.queues[0], data_buffer, CL_TRUE, 0, data_size,
                                       (void*) image, 0, NULL, NULL);
   OPENCL_CHECK(opencl_error);

   // Hard coded output file now, until we get to options.
   if (!write_image(&params, image))
      return EXIT_FAILURE;

   opencl_error = clReleaseMemObject(data_buffer);
   OPENCL_CHECK(opencl_error);

   free(image);

   opencl_error = clReleaseKernel(mandelbrot_kernel);
   OPENCL_CHECK(opencl_error);
   free(kernels);

   opencl_error = clReleaseProgram(program);
   OPENCL_CHECK(opencl_error);

   if (!opencl_free(&opencl))
      return EXIT_FAILURE;

   return EXIT_SUCCESS;
}


static void debug_print_parameters(const parameters* param) {
   printf("Current parameters:\n");
   printf("x: %g %g,  y: %g %g,  nx: %ld, ny: %ld, max_iter: %d, outfile: %s\n",\
   param->x[0], param->x[1], param->y[0], param->y[1], param->dim[0], param->dim[1], param->max_iter, param->outfile);
   return;
}
