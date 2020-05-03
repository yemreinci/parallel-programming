#include "so.h"
#include "timer.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <vector>

void parallel_qsort(int n, data_t* data, int n_thr) {
    // if there is only one thread left or the #elements is less than
    // some threshold, use std::sort
    if (n_thr <= 1 || n <= 256) {
        std::sort(data, data+n);
        return;
    } 

    int block_sz = (n + n_thr - 1) / n_thr;
    int num_blocks = (n + block_sz - 1) / block_sz;
    std::vector< std::pair<int, int> > mids(num_blocks);

    #pragma omp parallel num_threads(num_blocks)
    {
        int thread_n = omp_get_thread_num();
        int i = block_sz * omp_get_thread_num();
        int end = std::min(i + block_sz/2, n);
        int mid = (i + end) / 2;
        std::nth_element(data + i,
                         data + mid, 
                         data + end);

        mids[thread_n] = std::make_pair(data[mid], mid);
    }
    
    std::nth_element(mids.begin(), mids.begin() + num_blocks/2, mids.end());

    int p = mids[num_blocks/2].second;

    data_t pivot = data[p];
    std::swap(data[n-1], data[p]);
    int mid = std::partition(data, data+n-1, [pivot](const auto& em){ return em < pivot; }) - data;
    std::swap(data[n-1], data[mid]);

    #pragma omp parallel sections 
    {
        #pragma omp section
        {
            parallel_qsort(mid, data, n_thr/2);
        }

        #pragma omp section
        {
            parallel_qsort(n - mid - 1, data + mid + 1, (n_thr+1) / 2);
        }
    }
}

void psort(int n, data_t* data) {
    omp_set_nested(1);
    
    parallel_qsort(n, data, omp_get_max_threads());
}
