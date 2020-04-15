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

    constexpr int nd = 6;
    int nc = (ny + nd - 1) / nd;
    int ncd = nc * nd;

    double4_t* normal = double4_alloc(ncd * na);

    #pragma omp parallel for schedule(static, 1)
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

    for (int y = ny; y < ncd; y++) {
        for (int x = 0; x < na; x++) {
            for (int z = 0; z < nb; z++) {
                normal[x + y*na][z] = 0;
            }
        }
    }

    #pragma omp parallel for schedule(static, 1)
    for (int jc = 0; jc < nc; jc++) {
        for (int ic = jc; ic < nc; ic++) {
            double4_t t[nd][nd] = {};
            
            for (int k = 0; k < na; k++) {

                for (int jd = 0; jd < nd; jd++) {
                    for (int id = 0; id < nd; id++) {
                        t[jd][id] += normal[k + (jc*nd + jd)*na] * normal[k + (ic*nd + id)*na];
                    }
                }

            }
            
            for (int jd = 0; jd < nd; jd++) {
                int j = jc * nd + jd;
                for (int id = 0; id < nd; id++) {
                    int i = ic * nd + id;
                    if (i >= j && i < ny && j < ny) {
                        result[i + j*ny] = t[jd][id][0] + t[jd][id][1] + t[jd][id][2] + t[jd][id][3];
                    }
                }
            }

        }
    }

    free(normal);
}
