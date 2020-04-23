#include "mf.h"
#include <algorithm>

using namespace std;

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        float* window = new float[(2*hx + 1) * (2*hy + 1)];
        
        for (int x = 0; x < nx; x++) {
            
            int wi = 0;
            for (int j = max(0, y-hy); j <= min(y+hy, ny-1); j++) { 
                for (int i = max(0, x-hx); i <= min(x+hx, nx-1); i++) {
                     window[wi++] = in[i + nx*j];
                }
            }
            
            nth_element(window, window + (wi/2), window + wi);
            float t = window[wi/2];
            if (wi % 2) {
                out[x + nx*y] = t;
            }
            else {
                nth_element(window, window + (wi/2 - 1), window + wi);
                out[x + nx*y] = (t + window[wi/2 - 1]) / 2;
            }
        }
        
        delete[] window;
    }

}
