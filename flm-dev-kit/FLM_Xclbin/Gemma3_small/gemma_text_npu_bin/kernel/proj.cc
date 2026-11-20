// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "common.h"
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
// phase 1: qkv proj, D -> (D + D // 2) to RoPE
// phase 2: o_proj, D -> D to rms_residual
// phase 3: up/gate proj, D -> D * 8 to MT_GLU
// phase 4: down proj, D * 4 -> D, to rms_residual
// phase 4 is special, do not clear y to zero
AIE_PING_FLAG is_y_ping = false;
AIE_PING_FLAG is_wx_ping = false;
#if defined(__chess__)
float y_acc[16] = {0};
#else
alignas(aie::vector_decl_align) float y_acc[16] = {0};
#endif

template<int M, int K, int m, int k>
void linear_proj_iD(
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block * w_ping,
    q4k_block * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const uint32 pkt_id
) {
    aie_pkt_word* pkt_id_ping = reinterpret_cast<aie_pkt_word*>(y_ping + 14);
    aie_pkt_word* pkt_id_pong = reinterpret_cast<aie_pkt_word*>(y_pong + 14);
    // Under Peano the ping/pong flags are read into locals once and written back at
    // the end of the call, so no other store in the call can change them midway.
#if defined(__chess__)
    bool &wx = is_wx_ping;
    bool &yp = is_y_ping;
#else
    bool wx = is_wx_ping;
    bool yp = is_y_ping;
#endif
    bfloat16 b_col_reduce_add[ K/32]; 
    int i = 0;
    do {
        zero_y<m>(y_acc);
        int j = 0;
        do {
            wx = !wx;
            q4k_block* w_using = wx ? w_ping : w_pong;
            bf16* x_using = wx ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + j *(k/32);
            // special logic for i==0:
            if(i == 0){
                // fill the b_col_reduce_add
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / 32; l++)AIE_LOOP_FLATTEN{
                    *(b_col_reduce_add_ptr+ l) = bf16(aie::reduce_add(  aie::load_v<32>(x_using + 32*l) ));
                }
            }
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            j++;
        } while (j < K / k);
        _lock_acquire(y_prod_lock);
        yp = !yp;
        aie_pkt_word* pkt_id_using = yp ? pkt_id_ping : pkt_id_pong;
        bf16* y_using = yp ? y_ping : y_pong;
        *pkt_id_using = pkt_id;
        copy_y<m>(y_using + 16, y_acc);
        _lock_release(y_cons_lock);
        i++;
    } while (i < M / m);
#if !defined(__chess__)
    is_wx_ping = wx;
    is_y_ping = yp;
#endif
}

extern "C" {
constexpr int total_cores = 16;
void proj(
    bf16 * y_ping,
    q4k_block * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block * w_pong,
    bf16 * x_pong
)
{
    constexpr int m = 16;
    constexpr int k = 128;
    // phase 1: qkv proj, D -> (D + D / 2) to RoPE
    linear_proj_iD<(DQ + DK + DV) / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 1);

    // // phase 2: o proj, D -> D to  RMS
    linear_proj_iD<PADDED_D / total_cores, DQ, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 4);

    // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
    linear_proj_iD<2 * HIDDEN_SIZE / total_cores, D, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 8);

    // // phase 4.1 down-> 4D -> D, first D -> D
    linear_proj_iD<PADDED_D / total_cores, HIDDEN_SIZE, m, k>(x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, 5);

}


}