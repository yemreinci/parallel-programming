#include "cp.h"
#include "timer.h"
#include "cudacheck.h"
#include <cuda_runtime.h>
#include <cmath>

#define D 10
#define B 16

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void normalize(float *normal, const float* data, int ny, int nx, int nny) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;

    if (y >= ny) {
        for (int x = 0; x < nx; x++) {
            normal[y + x*nny] = 0;
        }
        return;
    }

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
        normal[y + x*nny] = (data[x + y*nx] - mean) / std;
    }
}

__global__ void matrix_mult(float* r, const float* normal, int ny, int nx, int nny) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    if (ic > jc)
        return;

    float v[D][D] = {};

    for (int k = 0; k < nx; k++) {
        for (int ib = 0; ib < D; ++ib) {
            int i = ic * B * D + ib * B + ia;
            for (int jb = 0; jb < D; ++jb) {
                int j = jc * B * D + jb * B + ja;
                v[ib][jb] += normal[nny*k + i] * normal[nny*k + j];
            }
        }
    }

    for (int ib = 0; ib < D; ++ib) {
        for (int jb = 0; jb < D; ++jb) {
            int i = ic * B * D + ib * B + ia;
            int j = jc * B * D + jb * B + ja;
            if (i < ny && j < ny) {
                r[ny*i + j] = v[ib][jb];
            }
        }
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
    int nny = (ny + (B*D) - 1) / (B*D) * (B*D);

    float* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, 2 * nny * nx * sizeof(float)));
    float* normalGPU = dataGPU + nny * nx;
    
    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    
    normalize<<<divup(ny, 32), 32>>>(normalGPU, dataGPU, ny, nx, nny); 
    CHECK(cudaGetLastError());

    dim3 dimBlock(B, B);
    dim3 dimGrid(nny / (B*D), nny / (B*D));

    matrix_mult<<<dimGrid, dimBlock>>>(resultGPU, normalGPU, ny, nx, nny);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(resultGPU));
}
