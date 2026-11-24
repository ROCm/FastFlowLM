// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../passThroughCommon.hpp"

#include "../../common_external.hpp"
#include "../../lock_util.hpp"



#include "../attention_core.h"

extern "C"
{
    void attention_bf16(
        bfloat16 *q_buffer,        // size of LQ_internal * DH
        bfloat16 *k_v_buffer_ping, // size of LK_V_internal * DH
        bfloat16 *k_v_buffer_pong, // size of LK_V_internal * DH
        bfloat16 *S_pij_buffer,    // size of LQ_internal * LK,
        float *O_buffer,           // size of LQ * DH
        int32_t *rtp_buffer,
        int32_t *ping_pong_buffer
    )
    {

        // NOTE: need to sync with the python definition, for now just hardcode it

 
        constexpr int DH = 64;

        constexpr int LQ_internal = 16;
        constexpr int LK_V_internal = 16;

        constexpr int r = 8;
        constexpr int s = 8;
        constexpr int t = 8;

        // The dimension that each CT process
        constexpr int LQ = 128;
        constexpr int LK = 512;


        core_attention_flow_q_repeat<LQ_internal, LK_V_internal, DH, LQ, LK, r, s, t, false>(
            q_buffer,
            k_v_buffer_ping,
            k_v_buffer_pong,
            S_pij_buffer,
            O_buffer,
            rtp_buffer,
            ping_pong_buffer);
        
    }

} // extern "C"
