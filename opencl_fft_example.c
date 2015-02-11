#include <stdio.h>
#include <math.h>

#include "opencl_utils.h"
#include "owl/owl_fft.h"

#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

int main (void)
{
   int i;
   const int n = 128;
   float data[2*n];

   cl_int opencl_error;
   opencl_handle opencl;
   cl_context context;

   owl_opencl_handle* opencl_handle;
   owl_fft_handle* fft_handle;
   owl_fft_complex_workspace* workspace;

   for (i = 0; i < n; i++) {
      REAL(data, i) = 0.0f;
      IMAG(data, i) = 0.0f;
   }

   data[0] = 1.0;

   for (i = 1; i <= 10; i++) {
         REAL(data, i) = REAL(data, n-i) = 1.0f;
   }

   for (i = 0; i < n; i++) {
      printf("%d: %e %e\n", i, REAL(data, i), IMAG(data, i));
   }
   printf ("\n");

   // allocate stuff
   opencl_discover(&opencl, CL_DEVICE_TYPE_ALL);
   context = clCreateContext(NULL, 1, opencl.devices, NULL, NULL, &opencl_error);

   opencl_handle = owl_opencl_init(context, NULL);
   if (opencl_handle == NULL) {
      printf("OpenCL init failed!\n");
      return EXIT_FAILURE;
   }

   fft_handle = owl_fft_init(opencl_handle);
   if (fft_handle == NULL) {
      printf("OpenCL init failed!\n");
      return EXIT_FAILURE;
   }


   workspace = owl_fft_complex_workspace_alloc(fft_handle, n);
   if (workspace == NULL) {
      printf("Workspace allocation failed!\n");
      return EXIT_FAILURE;
   }

   // forward DFT of data
   owl_fft_complex_forward(fft_handle, data, 1, n, NULL, workspace);

   for (i = 0; i < n; i++) {
      printf ("%d: %e %e\n", i, REAL(data, i), IMAG(data, i));
   }

   owl_fft_complex_workspace_free(workspace);

   owl_fft_free(fft_handle);
   owl_opencl_free(opencl_handle);
   // free stuff
   clReleaseContext(context);

   return 0;
}
