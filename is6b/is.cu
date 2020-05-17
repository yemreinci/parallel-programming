#include <memory>
#include <iostream>
#include <iomanip>
#include "is.h"   
#include "cudacheck.h"

#define DBG(x) std::cout << #x << ": " << x << std::endl; 

static void printall(int ny, int nx, int nnx, const float* data, const float* sum) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::endl;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            std::cout << "[ ";
            for (int c = 0; c < 3; c++)
                std::cout << data[c + i*3 + j*nx*3] << " ";
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i < nnx; i++) {
            std::cout << sum[i + j*nnx] << " ";
        }
        std::cout << std::endl;
    }
}

struct GPUResult {
    Result r;
    float val;
};


__global__ void segment_kernel(int ny, int nx, int nnx, int nb, int nd, 
        const float* sum, GPUResult* results) {

    int ly = blockIdx.x+1;
    int lxb = blockIdx.y;
    int ti1 = threadIdx.x;
    int ti2 = threadIdx.y;

    float sumall = sum[nx + ny*nnx];
    
    float best = 0;
    int best_j = 0;
        
    float area1 = (lxb*nd + ti2 - ti1)*ly;
    float area2 = 1.0 / (ny*nx - area1);
    area1 = 1.0 / area1;
    float areac = area1 + area2;
    float temp = area2 * sumall;
    float pro = temp * sumall;
    temp *= 2;

    for (int j = 0; j <= ny-ly; j++) {

        float best_in_loop = 0;

        for (int ib = 0; ib < nb - lxb; ib++) {

            int i1 = ib*nd + ti1;
            int i2 = (ib+lxb)*nd + ti2;

            float t = + sum[i2 + (j + ly)*nnx] 
                - sum[i2 + j*nnx]
                - sum[i1 + (j + ly)*nnx]
                + sum[i1 + j*nnx];


            float val = t * (t * areac - temp) + pro;

            if (val > best_in_loop) {
                best_in_loop = val;
            }
        }

        if (best_in_loop > best) {
            best = best_in_loop;
            best_j = j;
        }
    }

    Result res;

    { 
        int j = best_j;
        best = 0;

        for (int ib = 0; ib < nb - lxb; ib++) {
            int i1 = ib*nd + ti1;
            int i2 = (ib+lxb)*nd + ti2;

            float t = + sum[i2 + (j + ly)*nnx] 
                - sum[i2 + j*nnx]
                - sum[i1 + (j + ly)*nnx]
                + sum[i1 + j*nnx];


            float val = t * (t * areac - temp) + pro;

            if (val > best) {
                best = val;
                res.x0 = i1;
                res.x1 = i2;
                res.y0 = j;
                res.y1 = j + ly;
            }
        }

    }


    results[ ti2 + nd*( ti1 + nd * (lxb + (nb+1) * (ly-1)) ) ] = GPUResult {res, best};
}


Result segment(int ny, int nx, const float* data) {
    constexpr int nd = 8;
    const int nb = ((nx + 1) + nd - 1) / nd;
    const int nnx = nb * nd;

    float* sum = new float[(ny+1) * nnx];

    for (int i = 0; i < nnx; i++) {
        sum[i] = 0;
    }

    for (int j = 1; j <= ny; j++) {
        sum[j*nnx] = 0;
        
        for (int i = 1; i <= nx; i++) {
            sum[i + j*nnx] = + sum[i + (j-1)*nnx] 
                             + sum[i-1 + j*nnx] 
                             - sum[i-1 + (j-1)*nnx] 
                             + data[(i-1)*3 + (j-1)*nx*3];
        }

        for (int i = nx+1; i < nnx; i++) {
            sum[i + j*nnx] = nan("");
        } 

    }

    // printall(ny, nx, nnx, data, sum);

    float *sumGPU = NULL;
    GPUResult *resultsGPU = NULL;
    CHECK(cudaMalloc((void**)&sumGPU, ((ny+1) * nnx) * sizeof(float)));
    
    CHECK(cudaMalloc((void**)&resultsGPU, (nd*nd*(ny)*(nb+1)) * sizeof(GPUResult)));

    CHECK(cudaMemcpy(sumGPU, sum, ((ny+1) * nnx) * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(nd, nd);
    dim3 dimGrid(ny, nb+1);

    segment_kernel<<<dimGrid, dimBlock>>>(ny, nx, nnx, nb, nd, sumGPU, resultsGPU);

    GPUResult* results = new GPUResult[nd*nd*(ny)*(nb+1)];

    CHECK(cudaMemcpy(results, resultsGPU, (nd*nd*(ny)*(nb+1)) * sizeof(GPUResult),
                cudaMemcpyDeviceToHost));
    

    Result res;
    float best = 0;

    for (int i = 0; i < nd*nd*(ny)*(nb+1); i++) {
        if (results[i].val > best) {
            best = results[i].val;
            res = results[i].r;
        }
    }

    float sumall = sum[nx + ny*nnx];
    
    float innersum = + sum[res.x1 + res.y1*nnx] 
                     - sum[res.x1 + res.y0*nnx] 
                     - sum[res.x0 + res.y1*nnx] 
                     + sum[res.x0 + res.y0*nnx];

    res.inner[2] = 
        res.inner[1] = res.inner[0] = innersum / ((res.y1 - res.y0) * (res.x1 - res.x0));
    res.outer[2] = 
        res.outer[1] = res.outer[0] = (sumall - innersum) / (ny*nx - (res.y1 - res.y0) * (res.x1 - res.x0));

    
    delete[] sum;
    delete[] results;
    CHECK(cudaFree(resultsGPU));
    CHECK(cudaFree(sumGPU));

    return res;}
