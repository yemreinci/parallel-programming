#include "so.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <iostream>

// merge [f1, l1) and [f2, l2) into dst
static void parallel_merge(data_t* f1, data_t* l1, data_t* f2, data_t* l2, data_t* dst, int n_thr) {
    if (n_thr == 1) {
        std::merge(f1, l1, f2, l2, dst);
        return;
    } 

    data_t* mid1 = f1 + (l1 - f1) / 2;
    data_t* mid2 = std::upper_bound(f2, l2, *mid1);

    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            parallel_merge(f1, mid1, f2, mid2, dst, n_thr/2);
        }

        #pragma omp section 
        {
            parallel_merge(mid1, l1, mid2, l2, dst + (mid1 - f1) + (mid2 - f2), n_thr/2);
        }
    }
}

// sorts the values in array src
// if the sorted version is in dst, return true
// if the sorted version is in src, return false
// this helps us to alternate between the data array and the temporary array
// to avoid doing unnecessary std::copy's.
static bool parallel_sort(int n, data_t* src, data_t* dst, int n_thr) {
    if (n_thr == 1 || n < 16384) { 
        std::sort(src, src + n);
        return false;
    } 

    bool flag1, flag2;

    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            flag1 = parallel_sort(n / 2, src, dst, n_thr / 2);
        }

        #pragma omp section 
        {
            flag2 = parallel_sort((n + 1) / 2, src + n/2, dst + n/2, (n_thr+1) / 2);
        }
    }

    if (flag1 && flag2) { // no need to do copy, just swap pointers
        std::swap(src, dst);
    }
    else if (flag1) {
        std::copy(dst, dst + n/2, src);
        flag1 = false;
    }
    else if (flag2) {
        std::copy(dst + n/2, dst + n, src + n/2);
        flag2 = false;
    }

    parallel_merge(src, src + n/2, src + n/2, src + n, dst, n_thr);

    // here, flag1 is equal to flag2
    return !flag1;
}

void psort(int n, data_t* data) {
    if (n <= 1) 
        return;

    omp_set_nested(1);
    
    data_t *t = new data_t[n];

    bool flag = parallel_sort(n, data, t, omp_get_max_threads());

    if (flag) {
        std::copy(t, t+n, data);
    }

    delete[] t;
}
