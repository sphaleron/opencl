/*
 * These functions implement part of the GNU Scientific Library
 * FFT functionality in OpenCL.
 *
 * The API is intentionally very similar to GSL. Additional init/free functions
 * for setting up the necessary OpenCL structures are provided.
 *
 * In the future I will split this so that there are separate functions for
 * up/download and transfer, and GSL-like functions only wrap those.
 * User should have access directly to the cl_mem objects containing
 * the data and transform for fiddling with them.
 */

#include <CL/cl.h>

typedef struct {
   cl_context context;
   cl_uint dev_n;
   cl_device_id* devices;
   cl_program program;
   cl_command_queue* queues;
   cl_kernel fft_kernel;
} owl_fft_opencl_handle;

typedef struct {
   cl_uint n;                   // data size
   cl_uint nf;                  // number of factors
   cl_mem factor;               // array of factors
   cl_mem twiddle;              // twiddle factors
   cl_mem trig;                 // trigonometric lookup table
} owl_fft_complex_wavetable;

typedef struct {
   cl_uint n;
   cl_mem buffers[2];
} owl_fft_complex_workspace;


owl_fft_opencl_handle* owl_fft_opencl_init(cl_context context);
void owl_fft_opencl_free(owl_fft_opencl_handle* handle);


owl_fft_complex_workspace* owl_fft_complex_workspace_alloc(owl_fft_opencl_handle* handle, size_t n);

void owl_fft_complex_workspace_free(owl_fft_complex_workspace* workspace);

// Should we really define "owl_complex_packed_array" as in gsl?
int owl_fft_complex_forward (owl_fft_opencl_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace);

int owl_fft_complex_inverse (owl_fft_opencl_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace);
