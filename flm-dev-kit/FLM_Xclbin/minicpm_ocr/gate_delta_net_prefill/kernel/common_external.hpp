// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __COMMON_EXTERNAL_HPP__
#define __COMMON_EXTERNAL_HPP__











#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>






extern "C"{

    void zero_int32_1(int32 *pt)
    {
        *pt = 0;
    }

    void zero_int32_array(int32 *pt, int size)
    {
        for(int i = 0; i < size; i++){
            *(pt + i) = 0;
        }
    }
}


#endif