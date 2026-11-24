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
// phase 1: qkv proj, MODEL_DIM -> Linear_DQ + Linear_DK + Linear_DV, to the conv memtile
// phase 2: z proj,   MODEL_DIM -> Linear_DV, to the gated norm's memtile
// phase 3: o_proj,   Linear_DV -> MODEL_DIM, to MT[3] and the residual
// phase 4: up|gate,  MODEL_DIM -> 2 * INTERMEDIATE_SIZE, to the GLU tile
// phase 5: down,     INTERMEDIATE_SIZE -> MODEL_DIM, to MT[3] and the residual
// Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
AIE_PERSIST_ZERO bool is_y_ping = false;
AIE_PERSIST_ZERO bool is_w_ping = false;
AIE_PERSIST_ZERO bool is_x_ping = false;

// ---------------------------------------------------------------------------
// Scratch.
//
// At L_TOKENS rows this no longer fits the 10 KiB core stack, and none of it
// is live across a call anyway, so it is placed in tile data memory by the
// linker instead.  Sizes at L = 8: 1 KiB each for the two accumulator paths and
// the group sub-total, 1 KiB for y_acc, 2 KiB for the unpacked group, 5 KiB for
// the column sums.
// ---------------------------------------------------------------------------
alignas(aie::vector_decl_align) static float c_accum_s[L_TOKENS * QXNX_ROW_BLOCK_SIZE];
alignas(aie::vector_decl_align) static float c_accum_m[L_TOKENS * QXNX_ROW_BLOCK_SIZE];
alignas(aie::vector_decl_align) static float acc_s_group[L_TOKENS * QXNX_ROW_BLOCK_SIZE];
alignas(aie::vector_decl_align) static float y_acc[L_TOKENS * QXNX_ROW_BLOCK_SIZE];
// one 32-column group of a weight block, unpacked to bf16 once and read back
// by all L rows
alignas(aie::vector_decl_align) static bf16 q_scratch[GGML_GROUP_SIZE * QXNX_ROW_BLOCK_SIZE];
// 2 bf16 limbs per group, per row, per column block; see the fill loop in mvm().
// Sized by the widest K any mvm() below uses -- down's INTERMEDIATE_SIZE, 68
// column blocks against o's 24 and qkv's 20. One column block costs 256 B, so
// the cache is MAX_MVM_K bytes, 17 KiB.
//
// Split in two halves because 17 KiB does not fit a hole. This is not an
// aie.buffer, so the linker drops it wherever the bank-aligned buffers leave
// room; on a proj core that is four sub-bank chunks of 0.3 / 10.3 / 12 / 12 KiB
// and none of them takes 17. Cutting on a column-block boundary makes the index
// a branch on j and nothing else.
//
// The alternative -- refilling the sums on every row block instead of only at
// i == 0 -- is ~44 k extra reduce_adds per core per down mvm, several times what
// the mvm itself costs.
constexpr int MAX_MVM_K_    = MODEL_DIM > Linear_DV ? MODEL_DIM : Linear_DV;
constexpr int MAX_MVM_K     = MAX_MVM_K_ > INTERMEDIATE_SIZE ? MAX_MVM_K_ : INTERMEDIATE_SIZE;
constexpr int B_BLOCK_LIMBS = 2 * L_TOKENS * QXNX_COL_BLOCK_SIZE / GGML_GROUP_SIZE;
constexpr int B_BLOCKS      = MAX_MVM_K / QXNX_COL_BLOCK_SIZE;
constexpr int B_BLOCKS_LO   = B_BLOCKS / 2;
static_assert(MAX_MVM_K >= MODEL_DIM && MAX_MVM_K >= Linear_DV &&
              MAX_MVM_K >= INTERMEDIATE_SIZE,
              "every K passed to mvm() below has to fit the column-sum cache");
#if defined(__chess__)
alignas(aie::vector_decl_align) static bf16 b_col_reduce_add_lo[B_BLOCKS_LO * B_BLOCK_LIMBS];
alignas(aie::vector_decl_align) static bf16 b_col_reduce_add_hi[(B_BLOCKS - B_BLOCKS_LO) * B_BLOCK_LIMBS];
#else
// Peano links a core's statics into a single region, which fits the scratch
// above but not this cache as well. The two halves are buffers of the tile
// instead (iron/proj_tile.py), passed to proj_main.
static bf16 *b_col_reduce_add_lo;
static bf16 *b_col_reduce_add_hi;
#endif

void mvm(
    int M, int K,
    bf16 * x_ping,
    bf16 * x_pong,
    QUANT_BLOCK * w_ping,
    QUANT_BLOCK * w_pong,
    bf16 * y_ping,
    bf16 * y_pong,
    const int32_t send_x_output,
    const uint32 pkt_id

) {

    constexpr int m = QXNX_ROW_BLOCK_SIZE;
    constexpr int k = QXNX_COL_BLOCK_SIZE;
    constexpr int L = L_TOKENS;
    constexpr int groups_per_block = k / GGML_GROUP_SIZE;

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + 14);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + 14);

    for (int i = 0; i < M / m; i++) {
        #if QUANT_TYPE == __Q4_K__
            // the scale path and the min path are L x m floats now, too big for
            // the accumulator register file; they live in memory and nothing
            // touches y_acc until _Q4K_finalize_L()
            _Q4K_zero_L<L>(c_accum_s, c_accum_m);
        #else
            zero_vectorized<float, L * m>(y_acc);
        #endif
        for (int j = 0; j < K / k; j++) {

            _lock_acquire(x_cons_lock);
            _lock_acquire(w_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            is_w_ping = !is_w_ping;
            QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;


            #if QUANT_TYPE == __Q4_K__
                // [column block][row][group], two bf16 limbs per group. The
                // cache is cut in half on a column-block boundary; see above.
                static_assert(B_BLOCK_LIMBS == L * 2 * groups_per_block,
                              "the split assumes one column block is B_BLOCK_LIMBS limbs");
                bf16* b_col_reduce_add_ptr = j < B_BLOCKS_LO
                    ? b_col_reduce_add_lo + j * B_BLOCK_LIMBS
                    : b_col_reduce_add_hi + (j - B_BLOCKS_LO) * B_BLOCK_LIMBS;
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
                    for(int t = 0; t < L; t++){
                        AIE_PREPARE_FOR_PIPELINING
                        AIE_LOOP_UNROLL_FULL
                        for(int l = 0; l < groups_per_block; l++)AIE_LOOP_FLATTEN{
                            aie::accum<accfloat, GGML_GROUP_SIZE> x_acc;
                            x_acc.from_vector(aie::load_v<GGML_GROUP_SIZE>(
                                x_using + t * k + GGML_GROUP_SIZE * l));
                            float sum = aie::reduce_add(x_acc.template to_vector<float>());
                            bf16 hi = bf16(sum);
                            bf16* dst = b_col_reduce_add_ptr + t * 2 * groups_per_block + 2 * l;
                            *dst       = hi;
#if defined(__chess__)
                            *(dst + 1) = bf16(sum - float(hi));
#else
                            // Peano's scalar fsub is not exact, so the low limb
                            // can land one ulp off chess (see AIE_FSUB).
                            *(dst + 1) = bf16(AIE_FSUB(sum, float(hi)));
#endif
                        }
                    }
                }
                _Q4K_x_bf16_L<L>(c_accum_s, c_accum_m, acc_s_group,
                                 x_using, w_using, b_col_reduce_add_ptr, q_scratch);
            #elif QUANT_TYPE == __Q8_0__
                _Q80_x_bf16(y_acc, x_using, w_using);
            #endif

            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        #if QUANT_TYPE == __Q4_K__
            // the whole row block is done: scale path + min path, once
            _Q4K_finalize_L<L>(y_acc, c_accum_s, c_accum_m);
        #endif
        is_y_ping = !is_y_ping;

        // y is [row][token][32]: the L rows of one 32-row block travel together
        // so the memtile keeps assembling whole packets per row block.
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
            copy_float_to_bf16<L * m>(y_using + 16, y_acc);
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
            copy_float_to_bf16<L * m>(y_using + 16 + L * m, y_acc); // offset of 16+L*m, 16 for packed_it
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
#if !defined(__chess__)
    , bf16 *b_col_lo, bf16 *b_col_hi
#endif
)
{
#if !defined(__chess__)
    b_col_reduce_add_lo = b_col_lo;
    b_col_reduce_add_hi = b_col_hi;
#endif
    // qkv, to the conv tile's memtile. 20 x replays out of MT[1].
    mvm((Linear_DQ + Linear_DK + Linear_DV) / MVM_CORES, MODEL_DIM,
        x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, send_x_output, pkt_id_to_conv);

    // z, the gate.
    mvm(Linear_DV / MVM_CORES, MODEL_DIM,
        x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, send_x_output, pkt_id_to_aux);

    // o = W_o * gated_attn.
    mvm(MODEL_DIM / MVM_CORES, Linear_DV,
        x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, send_x_output, pkt_id_to_rms);

    // the MLP's fused up | gate projection.
    mvm(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM,
        x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, send_x_output, pkt_id_to_glu);

    // mlp.down_proj, back to the rms tile for the second residual.
    mvm(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE,
        x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, send_x_output, pkt_id_to_rms);

}


}
