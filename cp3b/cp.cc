#include "cp.h"
#include "vector.h"
#include <cmath>
#include <new>
#include <x86intrin.h>

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

void correlate(int ny, int nx, const float* data, float* result) {
    constexpr int nb = 8; 

    int na = (ny + nb - 1) / nb;

    float8_t* normal = float8_alloc(na * nx);

    #pragma omp parallel for schedule(static, 1)
    for (int ja = 0; ja < na; ja++) {
        for (int jb = 0; jb < nb; jb++) {
            int j = ja*nb + jb;

            if (j < ny) {
                float mean = 0, std = 0;

                for (int i = 0; i < nx; i++) {
                    mean += data[i + j*nx] / nx;
                }

                for (int i = 0; i < nx; i++) {
                    float diff = data[i + j*nx] - mean;
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
    for (int ja = 0; ja < na; ja++) {
        for (int ia = ja; ia < na; ia++) {
            float8_t t[nb] = {};

            for (int k = 0; k < nx; k++) {
                constexpr int PF = 16;
                __builtin_prefetch(&normal[ia*nx + k + PF]);
                __builtin_prefetch(&normal[ja*nx + k + PF]);

                float8_t a000 = normal[k + ia*nx];
                float8_t b000 = normal[k + ja*nx];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);

                t[0] += a000 * b000;
                t[1] += a000 * b001;
                t[2] += a010 * b000;
                t[3] += a010 * b001;
                t[4] += a100 * b000;
                t[5] += a100 * b001;
                t[6] += a110 * b000;
                t[7] += a110 * b001;
            }


            for (int k = 0; k < nb; k++) {
                for (int r = 0; r < nb; r++) {
                    int j = ja*nb + (k ^ (r & 1));
                    int i = ia*nb + (k ^ (r & 6));
                    if (j < ny && i < ny && i >= j)
                        result[i + j*ny] = t[r][k];
                }
            } 

        }
    }

    free(normal);
}
