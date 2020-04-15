#include "cp.h"
#include <cmath>
#include <new>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 4;
    int na = (nx + nb - 1) / nb;

    double4_t* normal = double4_alloc(ny * na); 

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

        for (int x = 0; x < na; x++) {
            for (int z = 0; z < nb; z++) {
                if (z + x*nb < nx)
                    normal[x + y*na][z] = data[z + x*nb + y*nx];
                else
                    normal[x + y*na][z] = mean;
            }

            normal[x + y*na] -= mean;
            normal[x + y*na] /= std;
        }

    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double4_t t = {0, 0, 0, 0};

            for (int k = 0; k < na; k++) {
                t += normal[k + i*na] * normal[k + j*na];
            }

            result[i + j*ny] = t[0] + t[1] + t[2] + t[3];
        }
    } 

    free(normal);
}
