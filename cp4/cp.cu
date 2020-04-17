#include "cp.h"
#include "cudacheck.h"
#include <cuda_runtime.h>
#include <cmath>

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void normalize(float *normal, const float* data, int ny, int nx) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;

    if (y >= ny)
        return;

    float mean = 0, std = 0;

    for (int x = 0; x < nx; x++) {
        mean += data[x + y*nx] / nx;
    }

    for (int x = 0; x < nx; x++) {
        float diff = data[x + y*nx] - mean;
        std += diff * diff;
    }

    std = sqrt(std);

    for (int x = 0; x < nx; x++) {
        normal[x + y*nx] = (data[x + y*nx] - mean) / std;
    }
}

__global__ void matrix_mult(float* r, const float* normal, int ny, int nx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || i < j)
        return;
    
    float t = 0;

    for (int k = 0; k < nx; k++) {
        t += normal[k + i*nx] * normal[k + j*nx];
    }

    r[ny*j + i] = t;
}

void correlate(int ny, int nx, const float* data, float* result) {
    float* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
    
    float* normalGPU = NULL;
    CHECK(cudaMalloc((void**)&normalGPU, ny * nx * sizeof(float)));

    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    normalize<<<divup(ny, 32), 32>>>(normalGPU, dataGPU, ny, nx); 
    CHECK(cudaGetLastError());

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));

    matrix_mult<<<dimGrid, dimBlock>>>(resultGPU, normalGPU, ny, nx);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(normalGPU));
    CHECK(cudaFree(resultGPU));
}
