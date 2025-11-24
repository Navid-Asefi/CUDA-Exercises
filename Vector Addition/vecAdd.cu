#include <cuda.h>
#include <iostream>

// Vector Addition on CPU
void vecAddHost(float *A_h, float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

// This is the logic.
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Vector Addition on GPU
void vecAddDevice(float *A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // This technically fills the pointer with the address of the first memory
    // cell of device that we create with size n
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // This copies what A_h contains in the place we created with cudaMalloc
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    int blocks = (n + 255) / 256; // ceiling division for integers
    vecAddKernel<<<blocks, 256>>>(A_d, B_d, C_d, n);
    // This copies the results back into the host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Deletes the place we created in the device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    long long int n = 10000; // size of our test vector

    // Allocate host memory
    float *A_h = new float[n];
    float *B_h = new float[n];
    float *C_h = new float[n];

    // Fill A_h and B_h with some data
    for (int i = 0; i < n; i++) {
        A_h[i] = i * 1.0f;       // 0,1,2,...
        B_h[i] = (i + 1) * 2.0f; // 2,4,6,...
    }

    // Run the GPU vector addition
    vecAddDevice(A_h, B_h, C_h, n);

    // Verify the results
    std::cout << "Results:\n";
    for (int i = 0; i < n; i++) {
        std::cout << A_h[i] << " + " << B_h[i] << " = " << C_h[i] << "\n";
    }

    // Clean up host memory
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    return 0;
}
