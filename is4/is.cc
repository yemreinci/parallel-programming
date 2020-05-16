#include "is.h"
#include <omp.h>
#include <cmath>
#include "vector.h"
#include <x86intrin.h>

//#define DBG(x) std::cout << #x << " = " << x << std::endl;
#define DBG(x) {}
/*
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
}*/

static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 5); }
static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 1); }

typedef struct InterResult {
    int lxb, j, ly;
} InterResult;

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
        double my_best = 0;
        InterResult my_ires = {};
        Result my_res;

        // To avoid repeatedly updating my_res in the innermost loop, first just find in which loop the best 
        // value exist, then we find the exact position of the best result.

        #pragma omp for schedule(static, 1) 
        for (int ly = 1; ly <= ny; ly++) {
            for (int lxb = 0; lxb <= nb; lxb++) {
                double4_t areac[nd] = {}, temp[3][nd] = {}, pro[nd] = {};

                for (int k = 0; k < nd; k++) {
                    for (int id1 = 0; id1 < nd; id1++) {
                        int id2 = id1 ^ k;
                        double area1 = 1.0 / ((lxb*nd + id2 - id1) * ly);
                        double area2 = 1.0 / (ny*nx - (lxb*nd + id2 - id1) * ly);
                        areac[k][id1] = area1 + area2;

                        temp[0][k][id1] = area2 * sumall[0];
                        temp[1][k][id1] = area2 * sumall[1];
                        temp[2][k][id1] = area2 * sumall[2];
                    }
                    
                    pro[k] = temp[0][k] * sumall[0] + temp[1][k] * sumall[1] + temp[2][k] * sumall[2];
                }
                
                for (int j = 0; j <= ny-ly; j++) {

                    double4_t best_in_loop[2] = {};

                    for (int ib = 0; ib < nb-lxb; ib++) {
                        double4_t val[4] = { pro[0], pro[1], pro[2], pro[3] };
                        
                        for (int c = 0; c < 3; c++) {
                            double4_t b00 = sum[c + (ib+lxb)*3 + j*nb*3];
                            double4_t d00 = sum[c + (ib+lxb)*3 + (j+ly)*nb*3];
                            
                            double4_t db00 = d00 - b00;
                            
                            double4_t a00 = sum[c + ib*3 + j*nb*3];
                            double4_t c00 = sum[c + ib*3 + (j+ly)*nb*3];
                            double4_t ac = a00 - c00;

                            double4_t t0 = db00 + ac;
                            val[0] += t0 * (t0 * areac[0] - 2*temp[c][0]) ;

                            double4_t db01 = swap1(db00);
                            double4_t t1 = db01 + ac;
                            val[1] += t1 * (t1 * areac[1] - 2*temp[c][1]) ;
                            
                            double4_t db10 = swap2(db00);
                            double4_t t2 = db10 + ac;
                            val[2] += t2 * (t2 * areac[2] - 2*temp[c][2]) ;
                            
                            double4_t db11 = swap2(db01);
                            double4_t t3 = db11 + ac;
                            val[3] += t3 * (t3 * areac[3] - 2*temp[c][3]) ;
                        }

                        best_in_loop[0] = _mm256_max_pd(val[0], best_in_loop[0]);
                        best_in_loop[1] = _mm256_max_pd(val[1], best_in_loop[1]);
                        best_in_loop[0] = _mm256_max_pd(val[2], best_in_loop[0]);
                        best_in_loop[1] = _mm256_max_pd(val[3], best_in_loop[1]);
                    }

                    for (int k = 0; k < 2; k++) {
                        for(int i = 0; i < 4; i++)
                        if (best_in_loop[k][i] > my_best) {
                            my_best = best_in_loop[k][i];
                            my_ires.j = j;
                            my_ires.ly = ly;
                            my_ires.lxb = lxb;
                        }
                    }
                }
            }
        }

        // similar to the innermost loop above, but this one also updates my_res
        {
            int lxb = my_ires.lxb;
            int j = my_ires.j;
            int ly = my_ires.ly;
            my_best = 0;
                    
            double4_t areac[nd] = {}, temp[3][nd] = {};

            for (int k = 0; k < nd; k++) {
                for (int id1 = 0; id1 < nd; id1++) {
                    int id2 = id1 ^ k;
                    double area1 = 1.0 / ((lxb*nd + id2 - id1) * ly);
                    double area2 = 1.0 / (ny*nx - (lxb*nd + id2 - id1) * ly);
                    areac[k][id1] = area1 + area2;

                    temp[0][k][id1] = area2 * sumall[0];
                    temp[1][k][id1] = area2 * sumall[1];
                    temp[2][k][id1] = area2 * sumall[2];
                }
            }

            for (int ib = 0; ib < nb-lxb; ib++) {
                double4_t val[4] = {};
                
                for (int c = 0; c < 3; c++) {
                    double4_t b00 = sum[c + (ib+lxb)*3 + j*nb*3];
                    double4_t d00 = sum[c + (ib+lxb)*3 + (j+ly)*nb*3];
                    
                    double4_t db00 = d00 - b00;
                    
                    double4_t a00 = sum[c + ib*3 + j*nb*3];
                    double4_t c00 = sum[c + ib*3 + (j+ly)*nb*3];
                    double4_t ac = a00 - c00;

                    double4_t t0 = db00 + ac;
                    val[0] += t0 * (t0 * areac[0] - 2*temp[c][0]) + temp[c][0] * sumall[c];

                    double4_t db01 = swap1(db00);
                    double4_t t1 = db01 + ac;
                    val[1] += t1 * (t1 * areac[1] - 2*temp[c][1]) + temp[c][1] * sumall[c];
                    
                    double4_t db10 = swap2(db00);
                    double4_t t2 = db10 + ac;
                    val[2] += t2 * (t2 * areac[2] - 2*temp[c][2]) + temp[c][2] * sumall[c];
                    
                    double4_t db11 = swap2(db01);
                    double4_t t3 = db11 + ac;
                    val[3] += t3 * (t3 * areac[3] - 2*temp[c][3]) + temp[c][3] * sumall[c];
                }
                
                for (int k = 0; k < nd; k++) {
                    for (int id1 = 0; id1 < nd; id1++) {
                        int id2 = id1 ^ k;
                        if (val[k][id1] > my_best) {
                            my_best = val[k][id1];
                            my_res.y0 = j;
                            my_res.y1 = j + ly;
                            my_res.x0 = id1 + ib*nd;
                            my_res.x1 = id2 + (ib+lxb)*nd;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (my_best > best) {
                best = my_best;
                res = my_res;
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
