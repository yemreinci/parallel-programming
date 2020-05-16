#include "is.h"
#include <omp.h>
#include <cmath>
#include "vector.h"
#include "timer.h"
#include <x86intrin.h>

//#define DBG(x) std::cout << #x << " = " << x << std::endl;
#define DBG(x) {}
/*
static void printall(int ny, int nx, int nb, int nd, const float* data, const float4_t* sum) {
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

static inline float4_t swap1(float4_t x) { return _mm_permute_ps(x, ((((2 << 2) | 3) << 2) | 0) << 2 | 1 ); }
static inline float4_t swap2(float4_t x) { return _mm_permute_ps(x, ((((1 << 2) | 0) << 2) | 3) << 2 | 2 ); }

void print(float4_t x) {
        std::cout << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << std::endl;
}

typedef struct InterResult { // intermediate result
    int ly, j, lxb;
} InterResult;

Result segment(int ny, int nx, const float* data) {
    constexpr int nd = 4;
    int nb = ((nx + 1) + nd - 1) / nd;

    float4_t* sum = float4_alloc((ny+1) * nb);

    for (int ib = 0; ib <= nb; ib++) {
        sum[ib] = float4_0;
    }

    for (int j = 1; j <= ny; j++) {
        sum[j*nb][0] = 0;

        for (int ib = 0; ib < nb; ib++) {
            for (int id = (ib==0); id < nd; id++) {
                int i = ib*nd+id;
                    float &t = sum[ib + j*nb][id];
                    t = sum[ib + (j-1)*nb][id];

                    if (id > 0) {
                        t += sum[ib + j*nb][id-1] - sum[ib + (j-1)*nb][id-1];
                    }
                    else {
                        t += sum[(ib-1) + j*nb][nd-1] - sum[(ib-1) + (j-1)*nb][nd-1];
                    }

                    if (i > 0) {
                        if (i-1 < nx)
                            t += data[(i-1)*3 + (j-1)*nx*3];
                        else
                            t = nan("");
                    }
            }
        }
    }

    float best = 0;
    InterResult ires = {};
    float sumall = sum[nx/nd + ny*nb][nx%nd];

    #pragma omp parallel
    {
        float my_best = 0;
        InterResult my_ires; 

        #pragma omp for schedule(static, 1) 
        for (int ly = 1; ly <= ny; ly++) {
            for (int j = 0; j <= ny-ly; j++) {
                for (int lxb = 0; lxb <= nb; lxb++) {
                    float4_t areac[nd] = {}, temp[nd] = {};

                    for (int k = 0; k < nd; k++) {
                        for (int id1 = 0; id1 < nd; id1++) {
                            int id2 = id1 ^ k;
                            float area1 = 1.0 / ((lxb*nd + id2 - id1) * ly);
                            float area2 = 1.0 / (ny*nx - (lxb*nd + id2 - id1) * ly);
                            areac[k][id1] = area1 + area2;

                            temp[k][id1] = area2 * sumall;
                            /*temp[1][k][id1] = area2 * sumall[1];
                            temp[2][k][id1] = area2 * sumall[2]; */
                        }
                    }

                    float4_t best_in_loop[2] = {};

                    for (int ib = 0; ib < nb-lxb; ib++) {
                        float4_t val[4] = {};
                        
                        float4_t b00 = sum[(ib+lxb) + j*nb];
                        float4_t d00 = sum[(ib+lxb) + (j+ly)*nb];
                        
                        float4_t db00 = d00 - b00;
                        
                        float4_t a00 = sum[ib + j*nb];
                        float4_t c00 = sum[ib + (j+ly)*nb];
                        float4_t ac = a00 - c00;

                        float4_t t0 = db00 + ac;
                        val[0] = t0 * (t0 * areac[0] - 2*temp[0]) + temp[0] * sumall;
                        best_in_loop[0] = _mm_max_ps(val[0], best_in_loop[0]);

                        float4_t db01 = swap1(db00);
                        float4_t t1 = db01 + ac;
                        val[1] = t1 * (t1 * areac[1] - 2*temp[1]) + temp[1] * sumall;
                        best_in_loop[1] = _mm_max_ps(val[1], best_in_loop[1]);
                        
                        float4_t db10 = swap2(db00);
                        float4_t t2 = db10 + ac;
                        val[2] = t2 * (t2 * areac[2] - 2*temp[2]) + temp[2] * sumall;
                        best_in_loop[0] = _mm_max_ps(val[2], best_in_loop[0]);
                        
                        float4_t db11 = swap2(db01);
                        float4_t t3 = db11 + ac;
                        val[3] = t3 * (t3 * areac[3] - 2*temp[3]) + temp[3] * sumall;
                        best_in_loop[1] = _mm_max_ps(val[3], best_in_loop[1]);
                    
                    }
                    
                    for (int k = 0; k < 2; k++)
                    for (int i = 0; i < 4; i++) {
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

        #pragma omp critical
        {
            if (my_best > best) {
                best = my_best;
                ires = my_ires;
            }
        }
    }

    Result res;
    best = 0;

    
    {
        int lxb = ires.lxb;
        int j = ires.j;
        int ly = ires.ly;
        res.y0 = j;
        res.y1 = j + ly;
                    
        float4_t areac[nd] = {}, temp[nd] = {};

        for (int k = 0; k < nd; k++) {
            for (int id1 = 0; id1 < nd; id1++) {
                int id2 = id1 ^ k;
                float area1 = 1.0 / ((lxb*nd + id2 - id1) * ly);
                float area2 = 1.0 / (ny*nx - (lxb*nd + id2 - id1) * ly);
                areac[k][id1] = area1 + area2;

                temp[k][id1] = area2 * sumall;
                /*temp[1][k][id1] = area2 * sumall[1];
                  temp[2][k][id1] = area2 * sumall[2]; */
            }
        }

        for (int ib = 0; ib < nb-lxb; ib++) {
            float4_t val[4] = {};
            
            float4_t b00 = sum[(ib+lxb) + j*nb];
            float4_t d00 = sum[(ib+lxb) + (j+ly)*nb];
            
            float4_t db00 = d00 - b00;
            
            float4_t a00 = sum[ib + j*nb];
            float4_t c00 = sum[ib + (j+ly)*nb];
            float4_t ac = a00 - c00;

            float4_t t0 = db00 + ac;
            val[0] = t0 * (t0 * areac[0] - 2*temp[0]) + temp[0] * sumall;
           // best_in_loop = _mm_max_ps(best_in_loop, val[0]);

            float4_t db01 = swap1(db00);
            float4_t t1 = db01 + ac;
            val[1] = t1 * (t1 * areac[1] - 2*temp[1]) + temp[1] * sumall;
           // best_in_loop = _mm_max_ps(best_in_loop, val[1]);
            
            float4_t db10 = swap2(db00);
            float4_t t2 = db10 + ac;
            val[2] = t2 * (t2 * areac[2] - 2*temp[2]) + temp[2] * sumall;
           // best_in_loop = _mm_max_ps(best_in_loop, val[2]);
            
            float4_t db11 = swap2(db01);
            float4_t t3 = db11 + ac;
            val[3] = t3 * (t3 * areac[3] - 2*temp[3]) + temp[3] * sumall;
           // best_in_loop = _mm_max_ps(best_in_loop, val[3]);
               
            for (int k = 0; k < nd; k++) {
                for (int id1 = 0; id1 < nd; id1++) {
                    int id2 = id1 ^ k;
                    if (val[k][id1] > best) {
                        best = val[k][id1];
                        res.x0 = id1 + ib*nd;
                        res.x1 = id2 + (ib+lxb)*nd;
                    }
                }
            } 
        }

    }

    float innerarea = (res.y1 - res.y0) * (res.x1 - res.x0);

    for (int c = 0; c < 3; c++) {
        float innersum = + sum[res.x1/nd + res.y1*nb][res.x1%nd]
                       - sum[res.x0/nd + res.y1*nb][res.x0%nd]
                       - sum[res.x1/nd + res.y0*nb][res.x1%nd]
                       + sum[res.x0/nd + res.y0*nb][res.x0%nd];
        
        res.inner[c] = innersum / innerarea;
        res.outer[c] = (sumall - innersum) / (ny*nx - innerarea);
    }

    free(sum);

    return res;
}
