#include "cp.h"
#include "timer.h"
#include "cudacheck.h"
#include <cuda_runtime.h>
#include <cmath>

#define C 5

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

__global__ void normalize(float *normal, const float* data, int ny, int nx, int nny, int nnx) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;

    if (y >= ny) {
        for (int x = 0; x < nnx; x++) {
            normal[y + x*nny] = 0;
        }
        return;
    }

    float mean = 0, std = 0;

    for (int x = 0; x < nx; x++) {
        mean += data[x + y*nx];
    }

    mean /= nx;

    for (int x = 0; x < nx; x++) {
        float diff = data[x + y*nx] - mean;
        std += diff * diff;
    }

    std = sqrt(std);

    for (int x = 0; x < nx; x++) {
        normal[y + x*nny] = (data[x + y*nx] - mean) / std;
    }
    for (int x = nx; x < nnx; x++) {
        normal[y + x*nny] = 0;
    }
}

__global__ void matrix_mult(float* r, const float* normal, int ny, int nx, int nny) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    if (ic > jc)
        return;

    float v[8][8] = {};
    __shared__ float xx[C][8*8];
    __shared__ float yy[C][8*8];

    for (int ks = 0; ks < nx; ks += C) {
        int ija = ja*8 + ia;
        int i = ic * 8*8 + ija;
        int j = jc * 8*8 + ija;
        for (int f = 0; f < C; f++) {
            int k = ks + f;
            xx[f][ija] = normal[nny*k + i];
            yy[f][ija] = normal[nny*k + j];
        }

        __syncthreads();
        
        for (int f = 0; f < C; f++) {
            for (int ib = 0; ib < 8; ++ib) {
                for (int jb = 0; jb < 8; ++jb) {
                    v[ib][jb] += xx[f][ib*8 + ia] * yy[f][jb*8 + ja];
                }
            }
        }

        __syncthreads();
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 8 * 8 + ib * 8 + ia;
            int j = jc * 8 * 8 + jb * 8 + ja;
            if (i < ny && j < ny) {
                r[ny*i + j] = v[ib][jb];
            }
        }
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
    int nny = (ny + (8*8) - 1) / (8*8) * (8*8);
    int nnx = (nx + C - 1) / C * C;

    float* dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, (ny * nx + nny * nnx) * sizeof(float)));
    float* normalGPU = dataGPU + ny * nx;
    
    float* resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    
    normalize<<<divup(ny, 64), 64>>>(normalGPU, dataGPU, ny, nx, nny, nnx); 
    CHECK(cudaGetLastError());

    dim3 dimBlock(8, 8);
    dim3 dimGrid(nny / (8*8), nny / (8*8));

    matrix_mult<<<dimGrid, dimBlock>>>(resultGPU, normalGPU, ny, nx, nny);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(resultGPU));
}
