#include "is.h"
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "vector.h"
#include <x86intrin.h>

//#define DBG(x) std::cout << #x << " = " << x << std::endl;
#define DBG(x) {}

static void printall(int ny, int nx, int nb, int nd, const float* data, const double4_t* sum) {
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
        for (int ib = 0; ib < nb; ib++) {
            std::cout << "{ ";
            for (int id = 0; id < nd; id++) {
                std::cout << "[ ";
                for(int c = 0; c < 3; c++) {
                    std::cout << sum[c + ib*3 + j*nb*3][id] << " ";
                }
                std::cout << "]";
            }
            std::cout << " }";
        }
        std::cout << std::endl;
    }
}

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 5); }
static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 1); }

Result segment(int ny, int nx, const float* data) {
    constexpr int nd = 4;
    int nb = ((nx + 1) + nd - 1) / nd;

    double4_t* sum = double4_alloc((ny+1) * nb * 3);

    for (int ib = 0; ib <= nb; ib++) {
        for (int c = 0; c < 3; c++) {
            sum[c + ib*3] = double4_0;
        }
    }

    for (int j = 1; j <= ny; j++) {
        sum[0 + j*nb*3][0] = sum[1 + j*nb*3][0] = sum[2 + j*nb*3][0] = 0;

        for (int ib = 0; ib < nb; ib++) {
            for (int id = (ib==0); id < nd; id++) {
                int i = ib*nd+id;
                for (int c = 0; c < 3; c++) {
                    double &t = sum[c + ib*3 + j*nb*3][id];
                    t = sum[c + ib*3 + (j-1)*nb*3][id];
                    if (id > 0) {
                        t += sum[c + ib*3 + j*nb*3][id-1] - sum[c + ib*3 + (j-1)*nb*3][id-1];
                    }
                    else {
                        t += sum[c + (ib-1)*3 + j*nb*3][nd-1] - sum[c + (ib-1)*3 + (j-1)*nb*3][nd-1];
                    }

                    if (i > 0) {
                        if (i-1 < nx)
                            t += data[c + (i-1)*3 + (j-1)*nx*3];
                        else
                            t = nan("");
                    }
                }
            }
        }
    }

    double best = {};
    Result res = {};
    double sumall[3];
    for (int c = 0; c < 3; c++) {
        sumall[c] = sum[c + nx/nd*3 + ny*nb*3][nx%nd];
    }

    #pragma omp parallel
    {
        double my_best[4] = {};
        Result my_res[4];

        #pragma omp for schedule(static, 1) 
        for (int ly = 1; ly <= ny; ly++) {
            for (int lxb = 0; lxb <= nb; lxb++) {
                for (int j = 0; j <= ny-ly; j++) {
                    double4_t areac[nd] = {}, area2[nd] = {};

                    for (int id1 = 0; id1 < nd; id1++) {
                        for (int k = 0; k < nd; k++) {
                            int id2 = id1 ^ k;
                            double area1 = 1.0 / ((lxb*nd + id2 - id1) * ly);
                            area2[k][id1] = 1.0 / (ny*nx - (lxb*nd + id2 - id1) * ly);
                            areac[k][id1] = area1 + area2[k][id1];
                        }
                    }


                    for (int ib = 0; ib < nb-lxb; ib++) {
                        double4_t val[4] = {};
                        
                        for (int c = 0; c < 3; c++) {
                            double4_t a00 = sum[c + ib*3 + j*nb*3];
                            double4_t b00 = sum[c + (ib+lxb)*3 + j*nb*3];
                            double4_t c00 = sum[c + ib*3 + (j+ly)*nb*3];
                            double4_t d00 = sum[c + (ib+lxb)*3 + (j+ly)*nb*3];
                            double4_t ac = a00 - c00;

                            double4_t b10 = swap2(b00);
                            double4_t b01 = swap1(b00);
                            double4_t b11 = swap1(b10);
                            
                            double4_t d10 = swap2(d00);
                            double4_t d01 = swap1(d00);
                            double4_t d11 = swap1(d10);

                            double4_t t0 = d00 - b00 + ac;
                            val[0] += t0 * t0 * areac[0] + sumall[c] * area2[0] * (sumall[c] - 2*t0);

                            double4_t t1 = d01 - b01 + ac;
                            val[1] += t1 * t1 * areac[1] + sumall[c] * area2[1] * (sumall[c] - 2*t1);
                            
                            double4_t t2 = d10 - b10 + ac;
                            val[2] += t2 * t2 * areac[2] + sumall[c] * area2[2] * (sumall[c] - 2*t2);
                            
                            double4_t t3 = d11 - b11 + ac;
                            val[3] += t3 * t3 * areac[3] + sumall[c] * area2[3] * (sumall[c] - 2*t3);
                        }

                        for (int k = 0; k < nd; k++) {
                            for (int id1 = 0; id1 < nd; id1++) {
                                int id2 = id1 ^ k;
                                if (val[k][id1] > my_best[id1]) {
                                    my_best[id1] = val[k][id1];
                                    my_res[id1].y0 = j;
                                    my_res[id1].y1 = j + ly;
                                    my_res[id1].x0 = id1 + ib*nd;
                                    my_res[id1].x1 = id2 + (ib+lxb)*nd;
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < nd; i++) {
                if (my_best[i] > best) {
                    best = my_best[i];
                    res = my_res[i];
                }
            }
        }
    }

    double innerarea = (res.y1 - res.y0) * (res.x1 - res.x0);

    for (int c = 0; c < 3; c++) {
        double innersum = + sum[c + res.x1/nd*3 + res.y1*nb*3][res.x1%nd]
                       - sum[c + res.x0/nd*3 + res.y1*nb*3][res.x0%nd]
                       - sum[c + res.x1/nd*3 + res.y0*nb*3][res.x1%nd]
                       + sum[c + res.x0/nd*3 + res.y0*nb*3][res.x0%nd];
        
        res.inner[c] = innersum / innerarea;
        res.outer[c] = (sumall[c] - innersum) / (ny*nx - innerarea);
    }

    free(sum);

    return res;
}
