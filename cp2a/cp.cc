#include "cp.h"
#include <cmath>
#include <iostream>

void correlate(int ny, int nx, const float* data, float* result) {
    double* normal = new double[nx * ny];

    for (int y = 0; y < ny; y++) {
        double mean = 0, std = 0;

        for (int x = 0; x < nx; x++) {
            mean += data[x + y*nx] / nx;
        }

        for (int x = 0; x < nx; x++) {
            double diff = data[x + y*nx] - mean;
            std += diff * diff;
        }

        std = sqrt(std);

        for (int x = 0; x < nx; x++) {
            normal[x + y*nx] = (data[x + y*nx] - mean) / std;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double tt[4] = {};

            for (int k = 0; k < nx / 4; k++) {
                for (int m = 0; m < 4; m++) {
                    tt[m] += normal[m + k*4 + i*nx] * normal[m + k*4 + j*nx];
                }
            }

            for (int m = 0; m < nx % 4; m++) {  
                tt[m] += 
                    normal[m + (nx / 4) * 4 + i*nx] * normal[m + (nx / 4) * 4 + j*nx];
            }

            result[i + j*ny] = tt[0] + tt[1] + tt[2] + tt[3];
        }
    }

    delete[] normal;
}