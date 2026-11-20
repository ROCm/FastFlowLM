// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "q4_k.h"

template<int m>
void zero_y(float * y) {
    zero_vectorized<float, m>(y);
}


template<int m>
void copy_y(bf16 * y, const float * y_acc) {
    for (int i = 0; i < m / 16; i++){
        aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + i * 16);
        aie::accum<accfloat, 16> acc;
        acc.from_vector(y_vec);
        aie::store_v(y + i * 16, acc.template to_vector<bf16>());
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;
constexpr int rtp_available_lock = 6;
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
__attribute__((section(".data")))
#endif
bool is_y_ping = false;
#if !defined(__chess__)
__attribute__((section(".data")))
#endif
bool is_wx_ping = false;
float y_acc[Q4NX_ROW_BLOCK_SIZE] = {0};


template<int M, int K, int m, int k>
void linear_proj_iD(
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block_t * w_ping,
    q4k_block_t * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id,
    bool & is_wx_ping,
    bool & is_y_ping
) {
    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);
    #ifndef Q4_0
    bfloat16 b_col_reduce_add[K / 32]; 
    #endif
    AIE_LOCK_LOOP_CTR i = 0;
    do {
        zero_y<m>(y_acc);
        AIE_LOCK_LOOP_CTR j = 0;
        do {
            is_wx_ping = !is_wx_ping;
            q4k_block_t* w_using = is_wx_ping ? w_ping : w_pong;
            bf16* x_using = is_wx_ping ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            #ifndef Q4_0
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + j *(k/32);
            // special logic for i == 0, avoid recompute if it is repeat.
            if(i == 0){
                // fill the b_col_reduce_add
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / 32; l++)AIE_LOOP_FLATTEN{
                    *(b_col_reduce_add_ptr+ l) = bf16(aie::reduce_add(  aie::load_v<32>(x_using + 32*l) ));
                }
            }
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
            #else
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc);
            #endif
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            j++;
        } while (j < K / k);
        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        uint32* pkt_id_using = is_y_ping ? pkt_id_ping : pkt_id_pong;
        bf16* y_using = is_y_ping ? y_ping : y_pong;
        *pkt_id_using = pkt_id;
        copy_y<m>(y_using + 16, y_acc);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M / m);
}

extern "C" {

void proj(
    bf16 * y_ping,
    q4k_block_t * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block_t * w_pong,
    bf16 * x_pong,
    int* IS_ATTN
)
{
    _lock_acquire(rtp_available_lock);
    bool wx_ping = is_wx_ping;
    bool y_ping_sel = is_y_ping;
    if (IS_ATTN[0] == 1) {
        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        linear_proj_iD<(DQ + DK + DV) / MVM_CORES, MODEL_DIM, Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, pkt_id_to_rope, wx_ping, y_ping_sel);

        // // phase 2: o proj, D -> D to  RMS
        linear_proj_iD<MODEL_DIM / MVM_CORES, DQ, Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, pkt_id_to_rms_norm, wx_ping, y_ping_sel);

        // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
        linear_proj_iD<2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, pkt_id_to_glu, wx_ping, y_ping_sel);

        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD<MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, pkt_id_to_rms_norm, wx_ping, y_ping_sel);
    }
    else{
        linear_proj_iD<VOCAB_SIZE_PADDED / MVM_CORES, MODEL_DIM, Q4NX_ROW_BLOCK_SIZE, Q4NX_COL_BLOCK_SIZE>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, pkt_id_to_rms_norm, wx_ping, y_ping_sel);
    }
    is_wx_ping = wx_ping;
    is_y_ping = y_ping_sel;

}


}