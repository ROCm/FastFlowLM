// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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
AIE_PERSIST_ZERO bool is_y_ping = false;
// weight ring: W_RING_BLOCKS one-block slots, 2 in each of w_ping / w_pong (so
// the stream can run further ahead of the math; MoE step 2, item 3a)
constexpr int W_RING_BLOCKS = 4;
int w_slot = W_RING_BLOCKS - 1;
AIE_PERSIST_ZERO bool is_x_ping = false;

#if QUANT_TYPE == __Q4_K__ && Q4K_INT8_MMUL
// int8 x of every x slice of one mvm() (see q4k_prepare_x): K <= MAX_MVM_K
// the A matrices of up to 16 x slices (1 KB each), as four 4 KB arrays: data
// memory is four 16 KB banks and an array must sit in one; with the 4-block
// weight ring, 8 KB pieces no longer fit
constexpr int Q4K_SLICES = MAX_MVM_K / QXNX_COL_BLOCK_SIZE;
constexpr int Q4K_SLICES_PER_PART = (Q4K_SLICES + 3) / 4;
alignas(64) static int8 q4k_a_cache_0[Q4K_SLICES_PER_PART * Q4K_A_BYTES];
alignas(64) static int8 q4k_a_cache_1[Q4K_SLICES_PER_PART * Q4K_A_BYTES];
alignas(64) static int8 q4k_a_cache_2[Q4K_SLICES_PER_PART * Q4K_A_BYTES];
alignas(64) static int8 q4k_a_cache_3[Q4K_SLICES_PER_PART * Q4K_A_BYTES];
alignas(64) static bf16 q4k_e_cache[MAX_MVM_K / QXNX_COL_BLOCK_SIZE * Q4K_E_ELEMS];
#endif

// x arrives by circuit broadcast: every core sees every x slice. In the routed
// expert down step the 4 columns run 4 different experts and the slices come
// round robin (expert 0, 1, 2, 3, then the next k columns), so a core keeps
// slice x_pick of every x_stride and hands the rest straight back.
static inline void skip_x_slices(int n) {
    for (AIE_LOCK_LOOP_CTR t = 0; t < n; t++) {
        _lock_acquire(x_cons_lock);
        is_x_ping = !is_x_ping;
        _lock_release(x_prod_lock);
    }
}

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
    const uint32 pkt_id,
    const int x_stride = 1,   // x slices per K step (NUM_MVM_COLUMN in routed down)
    const int x_pick = 0      // the slice this core uses, 0 .. x_stride - 1
) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);

    for (AIE_LOCK_LOOP_CTR i = 0; i < M / m; i++) {
        #if QUANT_TYPE == __Q4_K__
            // the scale path and the min path stay in registers for the whole
            // K loop; nothing touches y_acc until _Q4K_finalize()
            q4k_accum_t c_accum_s;
            q4k_accum_t c_accum_m;
            _Q4K_zero(c_accum_s, c_accum_m);
        #else
            zero_vectorized<float, m>(y_acc);
        #endif
        for (AIE_LOCK_LOOP_CTR j = 0; j < K / k; j++) {

            skip_x_slices(x_pick);
            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            w_slot = (w_slot + 1) % W_RING_BLOCKS;
            QUANT_BLOCK* w_using = (w_slot < W_RING_BLOCKS / 2 ? w_ping : w_pong) + w_slot % (W_RING_BLOCKS / 2);


            #if QUANT_TYPE == __Q4_K__
                // two bf16 limbs per group, interleaved (hi, lo)
                bf16* b_col_reduce_add_ptr = b_col_reduce_add + 2 * j * (k / GGML_GROUP_SIZE);
                // special logic for i == 0, avoid recompute if it is repeat.
                if(i == 0){
                    // Fill the b_col_reduce_add.  aie::reduce_add over a bf16
                    // vector already sums in accfloat, but returns bf16 and
                    // throws 16 of those mantissa bits away; that error rides
                    // the whole min path and is then amplified by the
                    // scale/min cancellation in _Q4K_finalize.  Reduce in
                    // float32 instead and hand the kernel two bf16 limbs.
                    // This runs once per mvm() and is read back by every row
                    // block, so the split costs nothing in the steady state.
                    AIE_PREPARE_FOR_PIPELINING
                    AIE_LOOP_UNROLL_FULL
                    for(int l = 0; l < k / GGML_GROUP_SIZE; l++)AIE_LOOP_FLATTEN{
                        aie::accum<accfloat, GGML_GROUP_SIZE> x_acc;
                        x_acc.from_vector(aie::load_v<GGML_GROUP_SIZE>(x_using + GGML_GROUP_SIZE * l));
                        float sum = aie::reduce_add(x_acc.template to_vector<float>());
                        bf16 hi = bf16(sum);
                        *(b_col_reduce_add_ptr + 2 * l)     = hi;
                        *(b_col_reduce_add_ptr + 2 * l + 1) = bf16(sum - float(hi));
                    }
                }
                #if Q4K_INT8_MMUL
                    // x slice j as int8 A matrices + group scales, made once (row
                    // block 0) and reused by every row block, like the sums above
                    int8* const parts[4] = {q4k_a_cache_0, q4k_a_cache_1, q4k_a_cache_2, q4k_a_cache_3};
                    int8* a_tiles = parts[j / Q4K_SLICES_PER_PART] + (j % Q4K_SLICES_PER_PART) * Q4K_A_BYTES;
                    bf16* e64 = q4k_e_cache + j * Q4K_E_ELEMS;
                    if (i == 0) {
                        // also rewrites this slice's min-path sums (of the rounded x)
                        q4k_prepare_x(x_using, a_tiles, e64, b_col_reduce_add_ptr);
                    }
                #endif
                // PROJ_COMPUTE_REPEAT is a timing experiment only (wrong results unless 1):
                // 0 = stream the weights but skip the math, 2 = do the math twice
                #ifndef PROJ_COMPUTE_REPEAT
                #define PROJ_COMPUTE_REPEAT 1
                #endif
                for (int rep = 0; rep < PROJ_COMPUTE_REPEAT; rep++) {
                #if Q4K_INT8_MMUL
                    _Q4K_x_int8(c_accum_s, c_accum_m, a_tiles, e64, w_using, b_col_reduce_add_ptr);
                #else
                    _Q4K_x_bf16(c_accum_s, c_accum_m, x_using, w_using, b_col_reduce_add_ptr);
                #endif
                }
            #elif QUANT_TYPE == __Q8_0__
                _Q80_x_bf16(y_acc, x_using, w_using);
            #endif

            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
            skip_x_slices(x_stride - 1 - x_pick);
        }
        #if QUANT_TYPE == __Q4_K__
            // the whole row is done: scale path + min path, once
            _Q4K_finalize(y_acc, c_accum_s, c_accum_m);
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
    int send_x_output,
    int col_idx       // this core's proj column, 0 .. NUM_MVM_COLUMN - 1
)
{
#if QUANT_TYPE == __Q4_K__ && Q4K_INT8_MMUL && !defined(__chess__)
    // q4k_prepare_x never writes 3 of every 4 A rows and needs them 0. chess's
    // loader zeroes .bss; Peano's does not, so zero the caches on the first call.
    AIE_PERSIST_ZERO static bool a_cache_zeroed = false;
    if (!a_cache_zeroed) {
        zero_vectorized<int8, Q4K_SLICES_PER_PART * Q4K_A_BYTES>(q4k_a_cache_0);
        zero_vectorized<int8, Q4K_SLICES_PER_PART * Q4K_A_BYTES>(q4k_a_cache_1);
        zero_vectorized<int8, Q4K_SLICES_PER_PART * Q4K_A_BYTES>(q4k_a_cache_2);
        zero_vectorized<int8, Q4K_SLICES_PER_PART * Q4K_A_BYTES>(q4k_a_cache_3);
        a_cache_zeroed = true;
    }
#endif
    alignas(aie::vector_decl_align) float  y_acc[QXNX_ROW_BLOCK_SIZE];
    // 2 bf16 limbs per group, see the fill loop in mvm()
    alignas(aie::vector_decl_align) bfloat16 b_col_reduce_add[2 * MAX_MVM_K / GGML_GROUP_SIZE];

    // phase 1: qkv proj, D -> DQ + DK + DV, to RoPE
    mvm((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope);

    // phase 2: o proj, DQ -> D, to RMS
    mvm(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

    // phase 3: shared expert up/gate, D -> 2 * I, to GLU
    mvm(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

    // phase 4: shared expert down, I -> D, to RMS
    mvm(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_aux);

    // phase 5: routed experts. Each column runs one expert, so the 4 columns
    // take NUM_ACT_EXPERTS / NUM_MVM_COLUMN rounds of gate/up + down.
    for(int k = 0; k < NUM_ACT_EXPERTS/NUM_MVM_COLUMN; k++){
        mvm((2*INTERMEDIATE_SIZE) / MVM_CORES_Per_Column, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);
        // down: x is the 4 experts' h, round robin; keep this column's expert
        mvm((MODEL_DIM) /MVM_CORES_Per_Column , INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu,
            NUM_MVM_COLUMN, col_idx);
    }
}


}
