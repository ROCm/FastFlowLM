// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../aie_kernel_utils.h"
#include "../window_attention_core.h"
#include "../window_attention_common.h"
extern "C"
{



    // NOW, do a simple passthrough kernel for testing the dataflow


    constexpr unsigned DH = 64;
    constexpr unsigned LQ = 56;
    constexpr unsigned LK = 56;
    constexpr unsigned r = 4;
    constexpr unsigned s = 8;
    constexpr unsigned t = 8;

    void attention_bf16(
        bfloat16 *q_buffer_ping, bfloat16* q_buffer_pong,// size of LQxDH,
        bfloat16 *k_buffer_ping,  // size of LKxDH,
        bfloat16 *v_buffer_ping, // size of LVxDH
        bfloat16 *S_pij_buffer,    // size of LQ * LK,
        bfloat16 *output_buffer,       // size of LQ * DH
        int32_t *ping_pong_buffer
    ){
        //NOTE: no point of using low precision at 488 no-emulation setting
        window_attention_bf16<DH, LQ, LK, r, s, t, false, false>(
            q_buffer_ping, q_buffer_pong,
            k_buffer_ping,
            v_buffer_ping,
            S_pij_buffer,
            output_buffer,
            ping_pong_buffer
        );
    }





}