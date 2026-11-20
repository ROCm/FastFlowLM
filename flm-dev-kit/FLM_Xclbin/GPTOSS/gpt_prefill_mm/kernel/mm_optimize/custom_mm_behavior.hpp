// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __CUSTOM_MM_BEHAVIOR_HPP_
#define __CUSTOM_MM_BEHAVIOR_HPP_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "../common_external.hpp"








template <int m>
void copy_float_to_bfloat16_lock_aware_with_nonlinear_gptoss(bfloat16 *y_pin, bfloat16 *y_pon, const float *y_acc, int y_prod_lock, int y_con_lock, int MODE)
{
    // acquire vs release for every 16 value
    // thus assume y size is 16 or greater
    static_assert(m % 16 == 0);
    constexpr int ping_pong_size = 512;    
    static_assert(m % (ping_pong_size * 2) == 0); // for ping pong assertion with each ping-pong write 32
    
    constexpr int lenPerLoop = ping_pong_size * 2;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MAX_ITERATION_COUNT(m / lenPerLoop)
    for (int i = 0; i < m / lenPerLoop; i++) {
        // Process ping buffer (first 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                if (MODE == 1){
                    // do a clamp before doing gelu
                    constexpr bfloat16 min_clamp =-3.30e+38f;
                    constexpr bfloat16 max_clamp =7.0f;        
                    aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(min_clamp);
                    aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(max_clamp);       
                    aie::vector<bfloat16, 16> acc_b16_clampped = aie::clamp( acc_b16, min_value, max_value);                          
                    aie::vector<bfloat16, 16> gelu_res = getGeluBf16_nonLUT<16>(acc_b16_clampped);
                    aie::store_v(y_pin + j * 16, gelu_res);
                }else{

                    //do a clamp and +1 before doing 
                    constexpr bfloat16 min_clamp = -7.0f;
                    constexpr bfloat16 max_clamp =7.0f;
                    constexpr bfloat16 one = 1.0f;
                    aie::vector<bfloat16, 16> one_vec = aie::broadcast<bfloat16, 16>(one);      
                    aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(min_clamp);
                    aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(max_clamp);       
                    aie::vector<bfloat16, 16> acc_b16_clampped = aie::clamp( acc_b16, min_value, max_value);                          
                    
                    aie::vector<bfloat16, 16> up_add_one = aie::add(acc_b16_clampped, one_vec);
                    aie::store_v(y_pin + j * 16, up_add_one);

                }

            }
            
            release(y_con_lock, 1);
        }
        
        // Process pong buffer (next 64 elements)
        {
            acquire_greater_equal(y_prod_lock, 1);
            
            // Load, convert and store  vectors of 16 elements each
            for (int j = 0; j < (ping_pong_size)/16; j++) {
                aie::accum<accfloat, 16> acc;
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + ping_pong_size + j * 16);
                acc.from_vector(y_vec);
                aie::vector<bfloat16, 16> acc_b16 = to_v16bfloat16(acc);
                if(MODE == 1){
                    // do a clamp before doing gelu
                    constexpr bfloat16 min_clamp =-3.30e+38f;
                    constexpr bfloat16 max_clamp =7.0f;        
                    aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(min_clamp);
                    aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(max_clamp);       
                    aie::vector<bfloat16, 16> acc_b16_clampped = aie::clamp( acc_b16, min_value, max_value);                          
                    aie::vector<bfloat16, 16> gelu_res = getGeluBf16_nonLUT<16>(acc_b16_clampped);
                    aie::store_v(y_pon + j * 16, gelu_res);
                }else{
                    constexpr bfloat16 min_clamp = -7.0f;
                    constexpr bfloat16 max_clamp =7.0f;
                    constexpr bfloat16 one = 1.0f;
                    aie::vector<bfloat16, 16> one_vec = aie::broadcast<bfloat16, 16>(one);      
                    aie::vector<bfloat16, 16> min_value = aie::broadcast<bfloat16, 16>(min_clamp);
                    aie::vector<bfloat16, 16> max_value = aie::broadcast<bfloat16, 16>(max_clamp);       
                    aie::vector<bfloat16, 16> acc_b16_clampped = aie::clamp( acc_b16, min_value, max_value);                          
                    
                    aie::vector<bfloat16, 16> up_add_one = aie::add(acc_b16_clampped, one_vec);
                    aie::store_v(y_pon + j * 16, up_add_one);
                }

            }
            
            release(y_con_lock, 1);
        }
        
        y_acc += lenPerLoop;
    }
}

#endif