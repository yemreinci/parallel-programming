#include "cp.h"
#include <cmath>

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
            double t = 0;

            for (int k = 0; k < nx; k++) {
                t += normal[k + i*nx] * normal[k + j*nx];
            }

            result[i + j*ny] = t;
        }
    }

    delete[] normal;
}