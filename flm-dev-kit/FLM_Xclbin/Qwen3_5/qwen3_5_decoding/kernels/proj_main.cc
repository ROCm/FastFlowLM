// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "q4_core.h"
#include "q8_core.h"



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
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
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
    bf16 * b_col_reduce_add,
    const int32_t send_x_output,
    const uint32 pkt_id
 
) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;

    #if QUANT_TYPE == __Q4_1__
        // Q4_1 has no super-block, so the deferred bias accumulator is the
        // row-level one and lives here next to y_acc.
        alignas(aie::vector_decl_align) float y_bias[m];
    #endif

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);

    for (int i = 0; i < M / m; i++) {
        #if QUANT_TYPE == __Q4_K__
            // the scale path and the min path stay in registers for the whole
            // K loop; nothing touches y_acc until _Q4K_finalize()
            q4k_accum_t c_accum_s;
            q4k_accum_t c_accum_m;
            q4k_accum_t c_accum_b;
            _Q4K_zero(c_accum_s, c_accum_m, c_accum_b);
        #else
            zero_vectorized<float, m>(y_acc);
        #endif
        #if QUANT_TYPE == __Q4_1__
            // the deferred nibble bias, sum_g s_g sum_i x_i; _Q41_finalize
            // turns it into the -128 sum_i x_i the MAC operand owes
            zero_vectorized<float, m>(y_bias);
        #endif
        for (int j = 0; j < K / k; j++) {

            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            is_w_ping = !is_w_ping;
            QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;


            #if QUANT_TYPE == __Q4_K__ || QUANT_TYPE == __Q4_1__
                // two bf16 limbs per group, interleaved (l0, l1)
                bf16* b_col_reduce_add_ptr =
                    b_col_reduce_add + REDUCE_LIMBS * j * (k / GGML_GROUP_SIZE);
                // special logic for i == 0, avoid recompute if it is repeat.
                if(i == 0){
                    // Fill the b_col_reduce_add.  aie::reduce_add over a bf16
                    // vector already sums in accfloat, but returns bf16 and
                    // throws 16 of those mantissa bits away; that error rides
                    // the whole min path and is then amplified by the
                    // scale/min cancellation in _Q4K_finalize.  Reduce in
                    // float32 instead and hand the kernel REDUCE_LIMBS bf16
                    // limbs.  This runs once per mvm() and is read back by
                    // every row block, so the split costs nothing in the
                    // steady state.
                    //
                    // Both limbs are read twice per group: once by the min
                    // path, once by the bias path (see NIBBLE_BIAS).
                    AIE_PREPARE_FOR_PIPELINING
                    AIE_LOOP_UNROLL_FULL
                    for(int l = 0; l < k / GGML_GROUP_SIZE; l++)AIE_LOOP_FLATTEN{
                        aie::accum<accfloat, GGML_GROUP_SIZE> x_acc;
                        x_acc.from_vector(aie::load_v<GGML_GROUP_SIZE>(x_using + GGML_GROUP_SIZE * l));
                        float sum = aie::reduce_add(x_acc.template to_vector<float>());
                        bf16 l0 = bf16(sum);
                        *(b_col_reduce_add_ptr + REDUCE_LIMBS * l)     = l0;
                        *(b_col_reduce_add_ptr + REDUCE_LIMBS * l + 1) =
#if defined(__chess__)
                            bf16(sum - float(l0));
#else
                            bf16(AIE_FSUB(sum, float(l0)));  // round like chess
#endif
                    }
                }
                #if QUANT_TYPE == __Q4_K__
                    _Q4K_x_bf16(c_accum_s, c_accum_m, c_accum_b, x_using, w_using, b_col_reduce_add_ptr);
                #else
                    // Q4_1 has no super-block, so s and m are the final scales:
                    // both paths land straight in y_acc. The deferred nibble
                    // bias is the one thing carried across the K loop, in
                    // y_bias, and _Q41_finalize settles it below.
                    _Q41_x_bf16(y_acc, y_bias, x_using, w_using, b_col_reduce_add_ptr);
                #endif
            #elif QUANT_TYPE == __Q8_0__
                _Q80_x_bf16(y_acc, x_using, w_using);
            #endif

            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        #if QUANT_TYPE == __Q4_K__
            // the whole row is done: scale path + min path - the nibble bias, once
            _Q4K_finalize(y_acc, c_accum_s, c_accum_m, c_accum_b);
        #endif
        #if QUANT_TYPE == __Q4_1__
            _Q41_finalize(y_acc, y_bias);
        #endif
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
    int send_x_output
)
{
    alignas(aie::vector_decl_align) float  y_acc[QXNX_ROW_BLOCK_SIZE];
    // REDUCE_LIMBS bf16 limbs per group, see the fill loop in mvm()
    alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[REDUCE_LIMBS * INTERMEDIATE_SIZE / GGML_GROUP_SIZE];

    for (int layer = 0; layer < LAYER_PERIOD; layer++) {
        if (layer == LAYER_PERIOD - 1) {
            // phase 1: qkv proj, D -> (D + D / 2) to RoPE
            mvm((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope);


            mvm(DQ / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

            // // phase 2: o proj, D -> D to  RMS
            mvm(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

            // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
            mvm(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

            // // phase 4.1 down-> 4D -> D, first D -> D
            mvm(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);
        }
        else{ // linear layer
            // phase 1: qkv proj, D -> (D + D / 2) to RoPE
            mvm((Linear_DQ + Linear_DK + Linear_DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_conv);


            mvm(Linear_DV / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

            // // phase 2: o proj, D -> D to  RMS
            mvm(MODEL_DIM / MVM_CORES, Linear_DV, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

            // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
            mvm(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

            // // phase 4.1 down-> 4D -> D, first D -> D
            mvm(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);
        }
    }
}


}