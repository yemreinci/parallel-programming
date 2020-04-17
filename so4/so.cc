#include "so.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <omp.h>

void psort(int n, data_t* data) {
    if (n <= 1) 
        return;

    int n_threads, share;

    // partition the array into #threads subarrays
    // and sort each of them using std::sort parallelly
    #pragma omp parallel
    {
        n_threads = omp_get_num_threads();
        share = (n + n_threads - 1) / n_threads;
        
        int id = omp_get_thread_num();
        int start = share * id;
        int end = std::min(n, start + share);
        
        if (start < end) {
            std::sort(data+start, data+end);
        }
    }
    
    data_t *t = new data_t[n];

    // merge subarrays
    // first, merge every consecutive pair of subarrays of
    // length "share" together, then merge every consecutive
    // pair of subarrays of length "2*share" then "4*share"... 
    for (; share < n; share *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n-share; i += 2*share) {
            int end = std::min(n, i + 2*share); 
            std::merge(data + i, data + i + share,
                       data + i + share, data + end,
                       t + i);
            std::copy(t + i, t + end, data + i);
        }
    }

    delete[] t;
}
