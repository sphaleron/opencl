// Let's try this way, see if it breaks
typedef float2 complex;

__kernel void mandelbrot(__global uint* image, float x0, float x1, float y0, float y1) {
  complex c;
  complex z = (complex)(0.0f, 0.0f);
  float tmp; // for storing new z.x while still calculating z.y

  // TODO change everything to vectors?
  uint px = get_global_id(0);
  uint py = get_global_id(1);
  uint counter = 0;
  
  // Will no longer work if round the size up to a multiple of 32!
  uint nx = get_global_size(0);
  uint ny = get_global_size(1);

  c.x = (x1*px + x0*(nx - 1 - px))/(nx - 1);
  c.y = (y1*py + y0*(ny - 1 - py))/(ny - 1);

  while(z.x*z.x + z.y*z.y < 4 && counter < MAX_ITER) {
     tmp = z.x*z.x - z.y*z.y + c.x;
     z.y = 2.0f*z.x*z.y + c.y;
     z.x = tmp;
     counter++;
  }

  image[py*nx + px] = counter;
}
