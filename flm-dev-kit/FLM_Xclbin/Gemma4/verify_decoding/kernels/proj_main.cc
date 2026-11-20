// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "q4_k.h"
#if L_TOKENS > 1
#include "q4_l8.h"
#endif


constexpr int pkt_id_to_rope     = 1;
constexpr int pkt_id_to_swa_rope = 2;
constexpr int pkt_id_to_rms      = 4;
constexpr int pkt_id_to_glu      = 8;

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
bool is_y_ping = false;
bool is_w_ping = false;
bool is_x_ping = false;

#if L_TOKENS > 1
// Replays of x per phase, fixed by the x replay tile (iron/proj_tile.py proj_x_replay_mt): a layer variant that needs fewer rounds
// (skip layers compute only q rows, SWA layers have fewer heads) consumes the surplus x without weights.
constexpr int QKV_REPLAYS = QKV_REPEATS;

/// consume `slices` x slices without touching weights or y: keeps the x ping/pong in step with the replay stream
static inline void drain_x_slices(int slices) {
    for (int j = 0; j < slices; j++) {
        is_x_ping = !is_x_ping;
        _lock_acquire(x_cons_lock);
        _lock_release(x_prod_lock);
    }
}
#endif

// template<int M, int K>
void linear_proj_iD(
    int M, int K,
    bf16 * x_ping,
    bf16 * x_pong,
    q4k_block_t * w_ping,
    q4k_block_t * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    float* y_acc,
    bf16 * b_col_reduce_add,
    const int32_t send_x_output,
    const uint32 pkt_id
) {

    constexpr int m = Q4NX_ROW_BLOCK_SIZE;
    constexpr int k = Q4NX_COL_BLOCK_SIZE;
    // elements per tile per row block: L_TOKENS_K tokens x 32 rows
#if L_TOKENS > 1
    constexpr int y_rows = L8_TOKENS * m;
#else
    constexpr int y_rows = m;
#endif

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);

    for (int i = 0; i < M / m; i++) {
#if L_TOKENS > 1
        _q4k_l8_zero(y_acc);
#else
        zero_vectorized<float, m>(y_acc);
#endif
        for (int j = 0; j < K / k; j++) {
            is_w_ping = !is_w_ping;
            is_x_ping = !is_x_ping;

            q4k_block_t* w_using = is_w_ping ? w_ping : w_pong;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
#if L_TOKENS > 1
            _qmm_q4k_l8(w_using, x_using, y_acc);
#else
            #ifndef Q4_0
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + j *(k/32);
            // special logic for i == 0, avoid recompute if it is repeat.
            if(i == 0){
                // fill the b_col_reduce_add
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / 32; l++)AIE_LOOP_FLATTEN{
                    *(b_col_reduce_add_ptr + l) = bf16(aie::reduce_add(aie::load_v<32>(x_using + 32*l)));
                }
            }
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc, b_col_reduce_add_ptr);
            #else
            _qmm_q4k_bf16<m, k>(w_using, x_using, y_acc);
            #endif
#endif
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
#if L_TOKENS > 1
            _q4k_l8_store(y_using + 16, y_acc);
#else
            copy_float_to_bf16<m>(y_using + 16, y_acc);
#endif
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
#if L_TOKENS > 1
            _q4k_l8_store(y_using + 16 + y_rows, y_acc);   // after the lower tile's rows; 16 for the packet header
#else
            copy_float_to_bf16<m>(y_using + 16+m, y_acc); // offset of 16+m, 16 for packed_it
#endif
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
    q4k_block_t * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    q4k_block_t * w_pong,
    bf16 * x_pong,
    int* IS_SWA,
    int* SKIP_KV,
    int send_x_output
)
{
#if L_TOKENS > 1
    // 8 tokens x 32 rows, static so it does not count against the stack; no column sums (the min path is folded into the dequantised weights)
    alignas(aie::vector_decl_align) static float y_acc[L8_TOKENS * Q4NX_ROW_BLOCK_SIZE];
    bfloat16* b_col_reduce_add = nullptr;
#else
    alignas(aie::vector_decl_align) float  y_acc[Q4NX_ROW_BLOCK_SIZE];
    #ifdef DOUBLE_WIDE_MLP
    alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[2 * INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE];
    #else
    alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE];
    #endif
#endif

    _lock_acquire(rtp_available_lock);
    if (IS_SWA[0]) {
        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        if (SKIP_KV[0] == 0) {
            static_assert((SWA_DQ + SWA_DK + SWA_DV) % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
#if L_TOKENS > 1
            // surplus qkv replays first (not after): nothing may be left in the x stream behind the last compute round, so the next phase's x
            // (which only exists once this phase's y has been produced) can never overtake it. See iron/proj_tile.py proj_x_replay_mt.
            drain_x_slices((QKV_REPLAYS - SWA_QKV_REPEATS) * (MODEL_DIM / Q4NX_COL_BLOCK_SIZE));
#endif
            linear_proj_iD((SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_swa_rope);
        }
        else{
            // if skip kv, only compute q_proj
            static_assert(SWA_DQ % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
#if L_TOKENS > 1
            // surplus qkv replays first (not after): nothing may be left in the x stream behind the last compute round, so the next phase's x
            // (which only exists once this phase's y has been produced) can never overtake it. See iron/proj_tile.py proj_x_replay_mt.
            drain_x_slices((QKV_REPLAYS - SWA_Q_REPEATS) * (MODEL_DIM / Q4NX_COL_BLOCK_SIZE));
#endif
            linear_proj_iD(SWA_DQ / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_swa_rope);
        }
        
        linear_proj_iD(MODEL_DIM / MVM_CORES, SWA_DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms);
    }
    else {
        // phase 1: qkv proj, D -> (D + D / 2) to RoPE
        if (SKIP_KV[0] == 0) {
            static_assert((DQ + DK + DV) % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
#if L_TOKENS > 1
            // surplus qkv replays first (not after): nothing may be left in the x stream behind the last compute round, so the next phase's x
            // (which only exists once this phase's y has been produced) can never overtake it. See iron/proj_tile.py proj_x_replay_mt.
            drain_x_slices((QKV_REPLAYS - QKV_REPEATS) * (MODEL_DIM / Q4NX_COL_BLOCK_SIZE));
#endif
            linear_proj_iD((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope);
        }
        else{
            // if skip kv, only compute q_proj
            static_assert(DQ % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0, "M must be divisible by MVM_CORES");
#if L_TOKENS > 1
            // surplus qkv replays first (not after): nothing may be left in the x stream behind the last compute round, so the next phase's x
            // (which only exists once this phase's y has been produced) can never overtake it. See iron/proj_tile.py proj_x_replay_mt.
            drain_x_slices((QKV_REPLAYS - Q_REPEATS) * (MODEL_DIM / Q4NX_COL_BLOCK_SIZE));
#endif
            linear_proj_iD(DQ / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope);
        }

        linear_proj_iD(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms);
    }
    #ifdef DOUBLE_WIDE_MLP
    if (SKIP_KV[0]) {
        // This part is not related with the Layer Type
        // // phase 3: up/gate proj
        linear_proj_iD(4 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD(MODEL_DIM / MVM_CORES, 2 * INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms);
    }
    else {
        // This part is not related with the Layer Type
        // // phase 3: up/gate proj
        linear_proj_iD(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

        // // phase 4.1 down-> 4D -> D, first D -> D
        linear_proj_iD(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms);
    }
    #else
    linear_proj_iD(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

    // // phase 4.1 down-> 4D -> D, first D -> D
    linear_proj_iD(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms);
    #endif
}


}