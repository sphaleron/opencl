#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "opencl_utils.h"

typedef struct {
  float x[2];
  float y[2];
  uint32_t nx;
  uint32_t ny;
  uint32_t max_iter;
} parameters;

static void usage() {
  printf("Usage: mandelbrot [-w width] [-h height] [-x lo,hi] [-y lo,hi]\n");
  return;
}

static void parameters_init(parameters* params) {
  params->x[0] = -1.0;
  params->x[1] =  1.0;
  params->y[0] = -1.0;
  params->y[1] =  1.0;
  params->nx = 256;
  params->ny = 256;
  params->max_iter = 1000;
  
  return;
}


int main(int argc, char* argv[]) {
  opencl_handle opencl;
  parameters params;
  
  parameters_init(&params);
  
//  usage();
  // read command line parameters
  
  printf("Naive Mandelbrot set generator\n\n");

  if (!opencl_discover(&opencl, CL_DEVICE_TYPE_ALL))
    return 1;

  // Make it single device now.
  if (!opencl_setup(&opencl, 1))
    return 1;

  


  if (!opencl_free(&opencl))
    return 1;
  
  return 0;
}