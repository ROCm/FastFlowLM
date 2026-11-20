// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.


#ifndef __SORT_HPP__
#define __SORT_HPP__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#include <aie_api/aie.hpp>


#include "aie_kernel_utils.h"

template<typename T, int size>
void bubble_sort_largest_to_smallest(T * data, int* index) {
    // Initialize the index array with original indices (0, 1, 2, ...)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size, size)
    for (int i = 0; i < size; i++) {
        index[i] = i;
    }

    bool swapped;

    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < size - 1; i++) {
        swapped = false;

        AIE_PREPARE_FOR_PIPELINING
        for (int j = 0; j < size - i - 1; j++) {
            // Compare the adjacent data elements.
            if ((float)data[j] <(float) data[j + 1]) {
                // Swap the data elements.
                T tempData = data[j];
                data[j] = data[j + 1];
                data[j + 1] = tempData;

                // IMPORTANT: Swap the corresponding indices in parallel.
                int tempIndex = index[j];
                index[j] = index[j + 1];
                index[j + 1] = tempIndex;
                
                swapped = true;
            }
        }

        // If no elements were swapped, the array is sorted.
        if (swapped == false) {
            break;
        }
    }
}

template<int size>
void bubble_sort_largest_to_smallest_bf16_with_index(bfloat16* data, int* index) {
    bubble_sort_largest_to_smallest<bfloat16, size>(data, index);
}

// Iterative quicksort, O(n log n) time, O(log n) stack space.
// Always processes the smaller partition first to bound stack depth at O(log n).
template<typename T, int size>
void quick_sort_largest_to_smallest(T* data, int* index) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size, size)
    for (int i = 0; i < size; i++) {
        index[i] = i;
    }

    // Stack depth is O(log n): worst case ceil(log2(size)) levels.
    // 32 is safe for size up to 2^16.
    constexpr int STACK_SIZE = 32;
    int lo_stk[STACK_SIZE];
    int hi_stk[STACK_SIZE];
    int top = 0;
    lo_stk[0] = 0;
    hi_stk[0] = size - 1;

    while (top >= 0) {
        int lo = lo_stk[top];
        int hi = hi_stk[top--];
        if (lo >= hi) continue;

        // Partition around pivot = data[hi], descending order.
        float pivot = (float)data[hi];
        int i = lo - 1;
        for (int j = lo; j < hi; j++) {
            if ((float)data[j] >= pivot) {
                ++i;
                T td = data[i]; data[i] = data[j]; data[j] = td;
                int ti = index[i]; index[i] = index[j]; index[j] = ti;
            }
        }
        ++i;
        T td = data[i]; data[i] = data[hi]; data[hi] = td;
        int ti = index[i]; index[i] = index[hi]; index[hi] = ti;

        int p = i;
        int left_sz  = p - lo;
        int right_sz = hi - p;

        // Push larger partition first so the smaller one is processed first,
        // keeping max stack depth at O(log n).
        if (left_sz >= right_sz) {
            if (lo < p - 1)  { lo_stk[++top] = lo;    hi_stk[top] = p - 1; }
            if (p + 1 < hi)  { lo_stk[++top] = p + 1; hi_stk[top] = hi;    }
        } else {
            if (p + 1 < hi)  { lo_stk[++top] = p + 1; hi_stk[top] = hi;    }
            if (lo < p - 1)  { lo_stk[++top] = lo;    hi_stk[top] = p - 1; }
        }
    }
}

template<int size>
void quick_sort_largest_to_smallest_bf16_with_index(bfloat16* data, int* index) {
    quick_sort_largest_to_smallest<bfloat16, size>(data, index);
}

// Top-K partial selection sort: O(n*K) time, O(1) extra space.
// Produces the K largest values in descending order at data[0..K-1] / index[0..K-1].
// Elements beyond K are left in an unspecified order.
template<typename T, int size, int K>
void partial_select_largest(T* data, int* index) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(size, size)
    for (int i = 0; i < size; i++) {
        index[i] = i;
    }

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(K, K)
    for (int i = 0; i < K; i++) {
        int max_j = i;

        AIE_PREPARE_FOR_PIPELINING
        for (int j = i + 1; j < size; j++) {
            if (data[j] > data[max_j]) {
                max_j = j;
            }
        }

        if (max_j != i) {
            T td = data[i]; data[i] = data[max_j]; data[max_j] = td;
            int ti = index[i]; index[i] = index[max_j]; index[max_j] = ti;
        }
    }
}

template<int size, int K>
void partial_select_largest_bf16_with_index(bfloat16* data, int* index) {
    partial_select_largest<bfloat16, size, K>(data, index);
}
#endif