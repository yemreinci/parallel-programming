#include "is.h"
#include <omp.h>
#include <iostream>
#include <cmath>
#include "vector.h"

//#define DBG(x) std::cout << #x << " = " << x << std::endl;
#define DBG(x) {}

Result segment(int ny, int nx, const float* data) { // std::cout << std::endl;
    constexpr int nd = 4;
    int nb = (nx + nd - 1) / nd;
    int na = nb * nd;

    double4_t* sum = double4_alloc((ny+1) * (na+nd));

    #define SUM(j, i) sum[(i) + (j)*(na+nd)]
    #define SUM4(j, i, ly, lx) SUM((j) + (ly), (i) + (lx)) - SUM((j) + (ly), (i)) - SUM((j), (i) + (lx)) + SUM((j), (i))  

    for (int i = 0; i <= nx; i++) {
        SUM(0, i) = double4_0;
    }
    for (int j = 0; j < ny; j++) {
        SUM(j + 1, 0) = double4_0;
        for (int i = 0; i < nx; i++) {
            for (int c = 0; c < 3; c++) {
                SUM(j + 1, i + 1)[c] = data[c + i*3 + j*nx*3];
            }
            
            SUM(j + 1, i + 1) += SUM(j, i + 1) + SUM(j + 1, i) - SUM(j, i);
        }

        for (int i = nx; i < na+nd-1; i++) {
            SUM(j + 1, i + 1) = double4_t {nan(""), nan(""), nan(""), 0};
        }
    }

    double best = 0;
    Result res = {};
    double4_t sumall = SUM(ny, nx);

    #pragma omp parallel
    {
        double my_best[nd] = {};
        Result my_res[nd];

        #pragma omp for schedule(static, 1) nowait
        for (int ly = 1; ly <= ny; ly++) {
            for (int lxb = 0; lxb < nb; lxb++) {
                double area1[nd], area2[nd], areac[nd];

                for (int lxd = 1; lxd <= nd; lxd++) {
                    int lx = lxb*nd + lxd;
                    area1[lxd-1] = 1.0 / (ly*lx);
                    area2[lxd-1] = 1.0 / (ny*nx - ly*lx);
                    areac[lxd-1] = area1[lxd-1] + area2[lxd-1];
                }

                for (int j = 0; j < ny-ly+1; j++) {
                    for (int ib = 0; ib < nb-lxb; ib++) {

                        for (int lxd = 1; lxd <= nd; lxd++)
                            for (int id = 0; id < nd; id++) {
                                int lx = lxd + lxb*nd;
                                int i = id + ib*nd;

                                double4_t t = SUM4(j, i, ly, lx);

                                t = t * t * areac[lxd-1] + sumall * area2[lxd-1] * (sumall  - 2 * t);

                                double val = t[0] + t[1] + t[2];

                                if (val > my_best[id]) {
                                    my_best[id] = val;
                                    my_res[id].y0 = j;
                                    my_res[id].y1 = j + ly;
                                    my_res[id].x0 = i;
                                    my_res[id].x1 = i + lx;
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

    double4_t innersum = SUM4(res.y0, res.x0, res.y1 - res.y0, res.x1 - res.x0);
    double innerarea = (res.y1 - res.y0) * (res.x1 - res.x0);
    
    double4_t inner = innersum / innerarea;
    double4_t outer = (SUM(ny, nx) - innersum) / (ny*nx - innerarea);

    for (int c = 0; c < 3; c++) {
        res.inner[c] = inner[c];
        res.outer[c] = outer[c];
    }

    free(sum);

    return res;
}
