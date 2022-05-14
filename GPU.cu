#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void gauss_determinant(double *a, int *n, double *dev_det);

__global__ void triangulation(double *a, int *n, double *dev_det);

__global__ void kernel(double *a, int i, int *n);

int main() {
  srand(time(NULL));
  FILE *write;

  if ((write = fopen("write.txt", "w")) == NULL) return 0;

  int *dev_size;
  cudaMalloc((void **)&dev_size, sizeof(int));  // todo mem check

  double det;
  double *dev_det;
  cudaMalloc((void **)&dev_det, sizeof(double));  // todo mem check

  double *matrix;
  double *dev_matrix;

  clock_t time_start, time_finish;

  for (int size = 0; size < 1000; ++size) {
    cudaMalloc((void **)&dev_matrix, size * size * sizeof(double));
    matrix = (double *)malloc(size * size * sizeof(double));

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i * size + j] = rand() % 10;
      }
    }

    time_start = clock();
    cudaMemcpy(dev_matrix, matrix, size * size * sizeof(double),
               cudaMemcpyHostToDevice);  // todo mem check
    cudaMemcpy(dev_size, &size, sizeof(int), cudaMemcpyHostToDevice);  // todo m

    triangulation<<<1, 1>>>(dev_matrix, dev_size, dev_det);
    cudaDeviceSynchronize();

    cudaMemcpy(matrix, dev_matrix, size * size * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&det, dev_det, sizeof(double),
               cudaMemcpyDeviceToHost);  // todo mem check

    time_finish = clock();
    fprintf(write, "%f\n",
            (float)(time_finish - time_start) / (float)CLOCKS_PER_SEC);

    free(matrix);
    cudaFree(dev_matrix);
  }

  return 0;
}

__global__ void kernel(double *a, int i, int *n) {
  unsigned int j = threadIdx.x;
  if (j > i && j <= *n - 1) {
    double mu = a[j * (*n) + i] / a[j * (*n) + i];
    for (int k = 0; k < *n; ++k) a[j * (*n) + k] += a[i * (*n) + k] * mu;
  }
}

__global__ void triangulation(double *a, int *n, double *dev_det) {
  int size = *n;

  for (int k = 0; k < *n - 1; k++) {
    // ex kernel func
    kernel<<<1, size>>>(a, k, n);
    cudaDeviceSynchronize();
  }
  gauss_determinant(a, n, dev_det);
}

__device__ void gauss_determinant(double *a, int *n, double *dev_det) {
  *dev_det = 1;
  for (int i = 0, j = 0; i < *n, j < *n; ++i, ++j) {
    *dev_det *= a[i * (*n) + j];
  }
}

void info() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  printf("Device name: %s\n", deviceProp.name);
  printf("Total global memory: %ull\n", deviceProp.totalGlobalMem);
  printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
  printf("Registers per block: %d\n", deviceProp.regsPerBlock);
  printf("Warp size: %d\n", deviceProp.warpSize);
  printf("Memory pitch: %d\n", deviceProp.memPitch);
  printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

  printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
         deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
         deviceProp.maxThreadsDim[2]);

  printf("Max grid size: x = %d, y = %d, z = %d\n", deviceProp.maxGridSize[0],
         deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

  printf("Clock rate: %d\n", deviceProp.clockRate);
  printf("Total constant memory: %d\n", deviceProp.totalConstMem);
  printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Texture alignment: %d\n", deviceProp.textureAlignment);
  printf("Device overlap: %d\n", deviceProp.deviceOverlap);
  printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

  printf("Kernel execution timeout enabled: %s\n",
         deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
}
