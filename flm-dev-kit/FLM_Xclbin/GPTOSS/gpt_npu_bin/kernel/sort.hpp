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
#endif