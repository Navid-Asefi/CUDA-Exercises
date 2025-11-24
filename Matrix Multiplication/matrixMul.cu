#include <cmath>
#include <cuda.h>
#include <iostream>

#define M 1024 // rows in A
#define K 512  // col in A, row in B
#define N 2048 // col in B
#define BLOCK_SIZE 16

// 1024_512 * 512*2048

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int column = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && column < n) {
    float sum = 0.0f;
    for (int l = 0; l < k; l++) {
      sum += A[row * k + l] * B[l * n + column];
    }
    C[row * n + column] = sum;
  }
}

int main() {
  int m = M;
  int k = K;
  int n = N;

  size_t sizeA = m * k * sizeof(float);
  size_t sizeB = k * n * sizeof(float);
  size_t sizeC = m * n * sizeof(float);

  // Allocate host memory
  float *A = new float[m * k];
  float *B = new float[k * n];
  float *C = new float[m * n];
  float *C_ref = new float[m * n];

  // Initialize input matrices (simple pattern)
  for (int i = 0; i < m * k; i++)
    A[i] = 1.0f;
  for (int i = 0; i < k * n; i++)
    B[i] = 1.0f;

  // Compute reference on CPU
  matmul_cpu(A, B, C_ref, m, k, n);

  // Allocate device memory
  float *dA, *dB, *dC;
  cudaMalloc((void **)&dA, sizeA);
  cudaMalloc((void **)&dB, sizeB);
  cudaMalloc((void **)&dC, sizeC);

  // Copy host → device
  cudaMemcpy(dA, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeB, cudaMemcpyHostToDevice);

  // Kernel launch geometry
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch
  matmul_gpu<<<grid, block>>>(dA, dB, dC, m, k, n);
  cudaDeviceSynchronize();

  // Copy device → host
  cudaMemcpy(C, dC, sizeC, cudaMemcpyDeviceToHost);

  // Verify correctness
  bool ok = true;
  for (int i = 0; i < m * n; i++) {
    if (fabs(C[i] - C_ref[i]) > 1e-3) {
      std::cout << "Mismatch at index " << i << ": " << C[i] << " vs "
                << C_ref[i] << "\n";
      ok = false;
      break;
    }
  }

  if (ok) {
    std::cout << "GPU result matches CPU result.\n";
  }

  // Clean up
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_ref;
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return 0;
}
