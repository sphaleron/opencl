// Let's try this way, see if it breaks
typedef float2 complex;

__kernel void mandelbrot(__global uint* image, float x0, float x1, float y0, float y1,
                         __global uint* histogram) {
  complex c;
  complex z = (complex)(0.0f, 0.0f);
  float tmp; // for storing new z.x while still calculating z.y

  // TODO change everything to vectors?
  uint px = get_global_id(0);
  uint py = get_global_id(1);
  uint counter = MAX_ITER;

  // Will no longer work if round the size up to a multiple of 32!
  uint nx = get_global_size(0);
  uint ny = get_global_size(1);

  c.x = (x1*px + x0*(nx - 1 - px))/(nx - 1);
  c.y = (y1*py + y0*(ny - 1 - py))/(ny - 1);

  while(z.x*z.x + z.y*z.y < 4 && counter > 0) {
     tmp = z.x*z.x - z.y*z.y + c.x;
     z.y = 2.0f*z.x*z.y + c.y;
     z.x = tmp;
     counter--;
  }

  image[py*nx + px] = counter;

  // This is a terrible idea, most counts go to the first bins and we serialize the access
  // using atomic ops. But histograms are hard, I don't want to put too much effort there now.
  if (counter > 0)
     atomic_inc(&histogram[counter - 1]);
}


// Parallel (inclusive) prefix sum, following GPU gems & some lecture notes
__kernel void scan(__global uint* data, __global uint* sums, __local uint* workspace, uint data_size) {
   uint thread_id = get_local_id(0);
   uint scan_size = get_local_size(0);
   uint offset = 2*scan_size*get_group_id(0);

   data = data + offset;
   data_size = min(data_size - offset, 2*scan_size);

   // Fetch data to workspace
   while (thread_id < data_size) {
      workspace[thread_id] = data[thread_id];
      thread_id += scan_size;
   }
   // In the last WG we may have some empty workspace:
   while (thread_id < 2*scan_size) {
      workspace[thread_id] = 0;
      thread_id += scan_size;
   }
   thread_id = get_local_id(0);

   // Reduction
   offset = 1;
   for (int d = scan_size; d > 0; d >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (thread_id < d) {
         int ai = offset*(2*thread_id + 1) - 1;
         int bi = ai + offset;
         workspace[bi] += workspace[ai];
      }
      offset <<= 1;
   }

   // Build the complete scan
   offset = scan_size >> 1;
   for (int d = 2; d <= scan_size; d <<= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (thread_id + 1 < d ) {
         int ai = offset*(2*thread_id + 2) - 1;
         int bi = ai + offset;
         workspace[bi] += workspace[ai];
      }
      offset >>= 1;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   if (sums && thread_id == 0)
      sums[get_group_id(0)] = workspace[2*scan_size - 1];

   while (thread_id < data_size) {
      data[thread_id] = workspace[thread_id];
      thread_id += scan_size;
   }
}


// Add the totals of each workgroup to data
__kernel void add_totals(__global uint* data, __global const uint* sums, uint data_size) {
   uint thread_id = get_local_id(0);
   uint scan_size = get_local_size(0);
   uint group     = get_group_id(0);

   if (group > 0) {
      uint to_add = sums[group];
      uint offset = group*2*scan_size + thread_id;
      if (offset < data_size)
         data[offset] += to_add;
      offset += scan_size;
      if (offset < data_size)
         data[offset] += to_add;
   }
}
