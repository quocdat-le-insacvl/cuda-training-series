#include <stdio.h>

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1);\
    }\
  } while (0) 

__global__ void vector_add(float* a, float* b, float* c, int N){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) c[i] = b[i] + a[i]; 
}


int main() {
  int n = 1000;
  int threadPerBlock = 256;
  size_t size = n * sizeof(float);
  
  float* a_h = (float*)malloc(size);
  float* b_h = (float*)malloc(size);
  float* c_h = (float*)malloc(size);

  for (int i=0; i<n; i++) {
    a_h[i] = rand()/(float)RAND_MAX;
    b_h[i] = rand()/(float)RAND_MAX;
    c_h[i] = 0;
  }

  float* a_d;
  cudaMalloc(&a_d, size);
  float* b_d;
  cudaMalloc(&b_d, size);
  float* c_d;
  cudaMalloc(&c_d, size);

  cudaCheckErrors("cudaMalloc failed");
  // Send data to device
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

  cudaCheckErrors("cudaMemcpy failed");
  // launch the kernel
  vector_add<<<(n+threadPerBlock-1)/threadPerBlock, threadPerBlock>>>(a_d, b_d, c_d, n);

  cudaCheckErrors("kernel launch failed");

  // Get data from device
  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

  cudaCheckErrors("kernel or cudaMem failed");

  printf("A[0]:%f\n", a_h[0]);
  printf("B[0]:%f\n", b_h[0]);
  printf("C[0]:%f\n", c_h[0]);
  
  free(a_h);
  free(b_h);
  free(c_h);
  // free
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

}
