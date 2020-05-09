#include "cp.h"
#include "vector.h"
#include <cmath>
#include <new>
#include <x86intrin.h>
#include "timer.h"

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 5); }
static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 1); }

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 4; 
    constexpr int nd = 2;
    constexpr int PF = 15;

    int na = ((ny + nb - 1) / nb + nd - 1) / nd * nd;
    int nc = na / nd;
    int ne = 1 << (32 - _lzcnt_u32(nc - 1)); // smallest power of 2 larger than na/nd

    double4_t* normal = double4_alloc(na * nx);
    double* db_res = new double[ny*ny];

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
                    normal[ja%2 + i*2 + ja/2*nx*2][jb] = (data[i + j*nx] - mean) / std;
                }
            }
            else {
                for (int i = 0; i < nx; i++) {
                    normal[ja%2 + i*2 + ja/2*nx*2][jb] = 0;
                }
            }
        }

    }

    // z-order
    int n_jia = nc * (nc+1) / 2;
    std::vector< std::pair<int, int> > jia_list(n_jia);
    
    for (int z = 0, it = 0; z < ne*ne; z++) {
        int ja = _pext_u32(z, 0xAAAAAAAA);
        int ia = _pext_u32(z, 0x55555555);
        
        if (ja < nc && ia < nc && ia >= ja) {
            jia_list[it++] = std::make_pair(ja, ia);
        }
    }   

    // WITH Z
    #pragma omp parallel for schedule(static, 20)
    for (int it = 0; it < n_jia; it++) {
        int ja = jia_list[it].first;
        int ia = jia_list[it].second;

    // WITHOUT Z
    // #pragma omp parallel for schedule(static, 1)
    // for (int ja = 0; ja < nc; ja++)
    //     for (int ia = ja; ia < nc; ia++) {

            double4_t t[nd][nd][nb] = {};

            for (int k = 0; k < nx/3; k++) {
                __builtin_prefetch(&normal[k*2 + ja*nx*2 + PF]);
                __builtin_prefetch(&normal[k*2 + ia*nx*2 + PF]);

                for (int jd = 0; jd < nd; jd++) {
                    double4_t b00 = normal[jd + k*2 + ja*nx*2];
                    double4_t b10 = swap2(b00);
                    double4_t b01 = swap1(b00);
                    double4_t b11 = swap1(b10);

                    for (int id = 0; id < nd; id++) {
                        double4_t a00 = normal[id + k*2 + ia*nx*2];

                        t[jd][id][0] += a00 * b00;
                        t[jd][id][1] += a00 * b01;
                        t[jd][id][2] += a00 * b10;
                        t[jd][id][3] += a00 * b11;
                    }
                }
            }


            for (int jd = 0; jd < nd; jd++) {
                for (int id = 0; id < nd; id++) {
                    for (int k = 0; k < nb; k++) {
                        for (int r = 0; r < nb; r++) {
                            int j = (ja*nd + jd)*nb + (k ^ r);
                            int i = (ia*nd + id)*nb + (k);
                            if (j < ny && i < ny && i >= j)
                                db_res[i + j*ny] = t[jd][id][r][k];
                        }
                    } 
                }
            }

        }

    // WITH Z
    #pragma omp parallel for schedule(static, 20)
    for (int it = 0; it < n_jia; it++) {
        int ja = jia_list[it].first;
        int ia = jia_list[it].second;

    // WITHOUT Z
    // #pragma omp parallel for schedule(static, 1)
    // for (int ja = 0; ja < nc; ja++)
    //     for (int ia = ja; ia < nc; ia++) {

            double4_t t[nd][nd][nb] = {};

            for (int k = nx/3; k < nx/3*2; k++) {
                __builtin_prefetch(&normal[k*2 + ja*nx*2 + PF]);
                __builtin_prefetch(&normal[k*2 + ia*nx*2 + PF]);

                for (int jd = 0; jd < nd; jd++) {
                    double4_t b00 = normal[jd + k*2 + ja*nx*2];
                    double4_t b10 = swap2(b00);
                    double4_t b01 = swap1(b00);
                    double4_t b11 = swap1(b10);

                    for (int id = 0; id < nd; id++) {
                        double4_t a00 = normal[id + k*2 + ia*nx*2];

                        t[jd][id][0] += a00 * b00;
                        t[jd][id][1] += a00 * b01;
                        t[jd][id][2] += a00 * b10;
                        t[jd][id][3] += a00 * b11;
                    }
                }
            }


            for (int jd = 0; jd < nd; jd++) {
                for (int id = 0; id < nd; id++) {
                    for (int k = 0; k < nb; k++) {
                        for (int r = 0; r < nb; r++) {
                            int j = (ja*nd + jd)*nb + (k ^ r);
                            int i = (ia*nd + id)*nb + (k);
                            if (j < ny && i < ny && i >= j)
                              result[i + j*ny] = db_res[i + j*ny] + t[jd][id][r][k];
                        }
                    } 
                }
            }

        }
    
    // WITH Z
    #pragma omp parallel for schedule(static, 20)
    for (int it = 0; it < n_jia; it++) {
        int ja = jia_list[it].first;
        int ia = jia_list[it].second;

    // WITHOUT Z
    // #pragma omp parallel for schedule(static, 1)
    // for (int ja = 0; ja < nc; ja++)
    //     for (int ia = ja; ia < nc; ia++) {

            double4_t t[nd][nd][nb] = {};

            for (int k = nx/3*2; k < nx; k++) {
                __builtin_prefetch(&normal[k*2 + ja*nx*2 + PF]);
                __builtin_prefetch(&normal[k*2 + ia*nx*2 + PF]);

                for (int jd = 0; jd < nd; jd++) {
                    double4_t b00 = normal[jd + k*2 + ja*nx*2];
                    double4_t b10 = swap2(b00);
                    double4_t b01 = swap1(b00);
                    double4_t b11 = swap1(b10);

                    for (int id = 0; id < nd; id++) {
                        double4_t a00 = normal[id + k*2 + ia*nx*2];

                        t[jd][id][0] += a00 * b00;
                        t[jd][id][1] += a00 * b01;
                        t[jd][id][2] += a00 * b10;
                        t[jd][id][3] += a00 * b11;
                    }
                }
            }


            for (int jd = 0; jd < nd; jd++) {
                for (int id = 0; id < nd; id++) {
                    for (int k = 0; k < nb; k++) {
                        for (int r = 0; r < nb; r++) {
                            int j = (ja*nd + jd)*nb + (k ^ r);
                            int i = (ia*nd + id)*nb + (k);
                            if (j < ny && i < ny && i >= j)
                                result[i + j*ny] += t[jd][id][r][k];
                        }
                    } 
                }
            }

        }

    free(normal);
    delete[] db_res;
}
