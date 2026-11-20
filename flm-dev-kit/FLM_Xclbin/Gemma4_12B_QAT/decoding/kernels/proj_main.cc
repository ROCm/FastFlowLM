// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "q4_core.h"

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_ping_lock = 4;
constexpr int y_prod_pong_lock = 5;
constexpr int rtp_available_lock = 6;
constexpr int y_cons_ping_lock = 7;
constexpr int y_cons_pong_lock = 8;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
AIE_PERSIST_ZERO bool is_y_ping = false;
AIE_PERSIST_ZERO bool is_w_ping = false;
AIE_PERSIST_ZERO bool is_x_ping = false;

void mvm(
    int M, int K,
    bf16 * x_ping,
    bf16 * x_pong,
    QUANT_BLOCK * w_ping,
    QUANT_BLOCK * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    float* y_acc,
    const int32_t send_x_output,
    const uint32 pkt_id
 
) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;
    
    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);

    for (int i = 0; i < M / m; i++) {
        zero_vectorized<float, m>(y_acc);
        for (int j = 0; j < K / k; j++) {

            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            is_w_ping = !is_w_ping;
            QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;

            _Q40_x_bf16(y_acc, x_using, w_using);

            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        is_y_ping = !is_y_ping;

        if(send_x_output != 0){
            uint32* pkt_id_using = is_y_ping ? pkt_id_ping : pkt_id_pong;
            bf16* y_using = is_y_ping ? y_ping : y_pong;
            *pkt_id_using = pkt_id;
            if (is_y_ping) {
                _lock_acquire(y_prod_ping_lock);
            }
            else{
                _lock_acquire(y_prod_pong_lock);
            }
            copy_float_to_bf16<m>(y_using + 16, y_acc);
            if (is_y_ping) {
                _lock_release(y_cons_ping_lock);
            }
            else {
                _lock_release(y_cons_pong_lock);
            }
        }else{

            bf16* y_using = is_y_ping ? y_ping : y_pong;
            if (is_y_ping) {
                _down_lock_acquire(y_prod_ping_lock);
            }
            else {
                _down_lock_acquire(y_prod_pong_lock);
            }
            copy_float_to_bf16<m>(y_using + 16 + m, y_acc); // offset of 16+m, 16 for packed_it
            if (is_y_ping) {
                _down_lock_release(y_cons_ping_lock);
            }
            else {
                _down_lock_release(y_cons_pong_lock);
            }
        }
    }
}


extern "C" {
    
void proj_main(
    bf16 * y_ping,
    QUANT_BLOCK * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    QUANT_BLOCK * w_pong,
    bf16 * x_pong,
    int* IS_SWA,
    int send_x_output
)
{
    alignas(aie::vector_decl_align) float  y_acc[QXNX_ROW_BLOCK_SIZE];

    _lock_acquire(rtp_available_lock);
    if (IS_SWA[0]) {
        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        static_assert((SWA_DQ + SWA_DK + SWA_DV) % (MVM_CORES * QXNX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
        mvm((SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_swa_rope);
        
        mvm(MODEL_DIM / MVM_CORES, SWA_DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_rms);
    }
    else {
        // global layer: skip v proj
        static_assert((DQ + DK) % (MVM_CORES * QXNX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
        mvm((DQ + DK) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_rope);
      
        mvm(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_rms);
    }
   
    mvm(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_glu);

    // // phase 4.1 down-> 4D -> D, first D -> D
    mvm(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, send_x_output, pkt_id_to_rms);

}


}