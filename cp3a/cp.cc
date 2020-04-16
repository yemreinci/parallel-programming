#include "cp.h"
#include "vector.h"
#include <cmath>
#include <new>
#include <x86intrin.h>

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 5); }
static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 1); }

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 4; 
    constexpr int nd = 2;

    int na = ((ny + nb - 1) / nb + nd - 1) / nd * nd;

    double4_t* normal = double4_alloc(na * nx);

    #pragma omp parallel for schedule(static, 1)
    for (int ja = 0; ja < na; ja++) {
        for (int jb = 0; jb < nb; jb++) {
            int j = ja*nb + jb;

            if (j < ny) {
                double mean = 0, std = 0;

                for (int i = 0; i < nx; i++) {
                    mean += (double) data[i + j*nx] / nx;
                }

                for (int i = 0; i < nx; i++) {
                    double diff = data[i + j*nx] - mean;
                    std += diff * diff;
                }

                std = sqrt(std);

                for (int i = 0; i < nx; i++) {
                    normal[i + ja*nx][jb] = (data[i + j*nx] - mean) / std;
                }
            }
            else {
                for (int i = 0; i < nx; i++) {
                    normal[i + ja*nx][jb] = 0;
                }
            }
        }

    }
   
    #pragma omp parallel for schedule(static, 1)
    for (int ja = 0; ja < na/nd; ja++) {
        for (int ia = ja; ia < na/nd; ia++) {
            double4_t t[nd][nd][nb] = {};

            for (int k = 0; k < nx; k++) {
                for (int jd = 0; jd < nd; jd++) {
                    for (int id = 0; id < nd; id++) {
                        //constexpr int PF = 4;
                        //__builtin_prefetch(&normal[ia*nx + k + PF]);
                        //__builtin_prefetch(&normal[ja*nx + k + PF]);

                        double4_t a00 = normal[k + (ia*nd + jd)*nx];
                        double4_t b00 = normal[k + (ja*nd + id)*nx];
                        double4_t a01 = swap1(a00);
                        double4_t b10 = swap2(b00);

                        t[jd][id][0] += a00 * b00;
                        t[jd][id][1] += a01 * b00;
                        t[jd][id][2] += a00 * b10;
                        t[jd][id][3] += a01 * b10;
                    }
                }
            }


            for (int jd = 0; jd < nd; jd++) {
                for (int id = 0; id < nd; id++) {
                    for (int k = 0; k < nb; k++) {
                        for (int r = 0; r < nb; r++) {
                            int j = ((ja*nd + id)*nb + (k ^ (r & 2)));
                            int i = ((ia*nd + jd)*nb + (k ^ (r & 1)));
                            if (j < ny && i < ny && i >= j)
                                result[i + j*ny] = t[jd][id][r][k];
                        }
                    } 
                }
            }

        }
    }

    free(normal);
}
