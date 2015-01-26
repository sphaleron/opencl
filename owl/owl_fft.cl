// This version is adapted from www.bealto.com
// Parameter p is the length of sub-FFT
#define PI 3.14159265359f
#define  DFT2(a, b) { float2 tmp = a - b; a = a + b; b = tmp; }

float2 mul(float2 a, float2 b) {
   return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}


float2 twiddle(float2 x, int k, float alpha) {
   float s, c;
   s = sincos((float)k*alpha, &c);
   return mul(x, (float2)(c, s));
}


__kernel void owl_fft_radix2(__global const float2* data, __global float2* output, unsigned int p) {
   const uint thread_id = get_global_id(0);

   const uint T = get_global_size(0);  // Number of threads
   const uint N = 2*T;                 // Amount of (complex) data points
   const uint k = thread_id & (p - 1);   // index only for powers of 2
   const uint j = ((thread_id - k) << 1) + k;    // output index (TODO why?!)

   float2 u0 = data[thread_id];
   float2 u1 = data[thread_id + T];

   float alpha = -PI*(float)k / (float)p;

   u1 = twiddle(u1, 1, alpha);

   DFT2(u0, u1);

   output[j] = u0;
   output[j + p] = u1;
}
