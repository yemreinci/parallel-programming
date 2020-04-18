#include "is.h"
#include <omp.h>
#include <iostream>
#include "vector.h"

//#define DBG(x) std::cout << #x << " = " << x << std::endl;
#define DBG(x) {}

Result segment(int ny, int nx, const float* data) {
    double4_t* sum = double4_alloc((ny+1) * (nx+1));

    #define SUM(j, i) sum[(i) + (j)*(nx+1)]
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
    }

    double best = 0;
    Result res = {};

    #pragma omp parallel
    {
        double my_best = 0;
        Result my_res;

        #pragma omp for schedule(static, 1) nowait
        for (int ly = 1; ly <= ny; ly++) {
            for (int lx = 1; lx <= nx; lx++) {
                if (ly == ny && lx == nx)
                    continue;
                double area1 = ly*lx;
                double area2 = ny*nx - area1;

                for (int j = 0; j < ny-ly+1; j++) {
                    for (int i = 0; i < nx-lx+1; i++) {
                        double4_t t1 = SUM4(j, i, ly, lx);
                        double4_t t2 = SUM(ny, nx) - t1;
                        
                        t1 = t1 * t1;
                        t2 = t2 * t2;

                        double avg1 = (t1[0] + t1[1] + t1[2]) / area1;
                        double avg2 = (t2[0] + t2[1] + t2[2]) / area2;

                        if (avg1 + avg2 > my_best) {
                            my_best = avg1 + avg2;
                            my_res.y0 = j;
                            my_res.y1 = j + ly;
                            my_res.x0 = i;
                            my_res.x1 = i + lx;
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
