__kernel void mandelbrot(__global uint* image) {
  uint px = get_global_id(0);
  uint py = get_global_id(1);
  // Will no longer work if round the size up to a multiple of 32!
  uint nx = get_global_size(0);

  image[py*nx + px] = 1;
}
