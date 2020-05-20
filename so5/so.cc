#include "so.h"
#include "timer.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cassert>
#include <x86intrin.h>
#include <cstdlib>
#include <vector>

template<typename T>
int parallel_partition(int n, T* data, int p, T* temp, int n_thr) {
    T pivot = data[p]; 
    std::swap(data[n-1], data[p]);

    int* mids = new int[n_thr];
    int* mids_sum = new int[n_thr+1];
    int block_sz = (n-1 + n_thr-1) / n_thr;

    #pragma omp parallel for
    for (int i = 0; i < n_thr; i++) {
        int start = block_sz * i;
        int end = std::min(n-1, block_sz * (i+1));
        mids[i] = std::partition(data+start, data+end, [pivot](const auto& em){ return em < pivot; }) - data;
    }

    mids_sum[0] = 0;
    for (int i = 0; i < n_thr; i++) {
        mids_sum[i+1] = mids_sum[i] + mids[i] - block_sz*i;
    }
            
    temp[mids_sum[n_thr]] = data[n-1];

    #pragma omp parallel for
    for (int i = 0; i < n_thr; i++) {
        int start = block_sz * i;
        int end = std::min(n-1, block_sz * (i+1));
        std::copy(data+start, data+mids[i], temp+mids_sum[i]);
        std::copy(data+mids[i], data+end, temp+mids_sum[n_thr]+1+block_sz*i-mids_sum[i]);
    }

    
    int t = mids_sum[n_thr];
    
    delete[] mids;
    delete[] mids_sum;

    return t;
}

template<typename T>
bool parallel_qsort(int n, T* data, T* temp, int n_thr) {
    // if there is only one thread left or the #elements is less than
    // some threshold, use std::sort
    if (n_thr <= 1 || n <= 256) {
        std::sort(data, data+n);
        return false;
    } 
    
    int block_sz = (n + n_thr - 1) / n_thr;
    int num_blocks = (n + block_sz - 1) / block_sz;
    std::vector< std::pair<T, int> > mids(num_blocks);

    #pragma omp parallel for
    for (int thread_n = 0; thread_n < num_blocks; thread_n++)
    {
        int thread_n = omp_get_thread_num();
        int i = block_sz * omp_get_thread_num();
        int end = std::min(i + block_sz/16, n);
        int mid = (i + end) / 2;
        std::nth_element(data + i,
                         data + mid, 
                         data + end);

        mids[thread_n] = std::make_pair(data[mid], mid);
    }
    
    std::nth_element(mids.begin(), mids.begin() + num_blocks/2, mids.end());
    int p = mids[num_blocks/2].second;

    int mid = parallel_partition(n, data, p, temp, n_thr);

    std::copy(temp, temp+n, data);

    #pragma omp task
    parallel_qsort(mid, data, temp, n_thr/2);

    parallel_qsort(n - mid - 1, data + mid + 1, temp + mid + 1, (n_thr+1) / 2);

    #pragma omp taskwait

    return false;
}

void psort(int n, data_t* data) { 
    omp_set_nested(1);
    
    data_t* temp = new data_t[n];
    
    int n_thr = 1 << (63 - __lzcnt64(omp_get_max_threads()));

    bool flag;
    
    #pragma omp parallel
    #pragma omp single
    flag = parallel_qsort(n, data, temp, n_thr);

    if (flag) {
        std::copy(temp, temp+n, data);
    }

    delete[] temp;
}
