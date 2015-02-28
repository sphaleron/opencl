/*
 * Fast Fourier Transforms.
 */

#ifndef OWL_FFT_H
#define OWL_FFT_H

#include "owl_opencl.h"

#include <CL/cl.h>

typedef struct {
   owl_opencl_handle* opencl;
   cl_program program;
   cl_kernel fft_kernel;
} owl_fft_handle;

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


owl_fft_handle* owl_fft_init(owl_opencl_handle* opencl);
void owl_fft_free(owl_fft_handle* handle);

owl_fft_complex_workspace* owl_fft_complex_workspace_alloc(owl_fft_handle* handle, size_t n);

void owl_fft_complex_workspace_free(owl_fft_complex_workspace* workspace);

// Should we really define "owl_complex_packed_array" as in gsl?
int owl_fft_complex_forward (owl_fft_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace);

int owl_fft_complex_inverse (owl_fft_handle* handle, float* data, size_t stride, size_t n,
                             const owl_fft_complex_wavetable* wavetable,
                             owl_fft_complex_workspace* workspace);

#endif
