// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "model_spec.h"
#include "aie_kernel_utils.h"
#include "aie_array_layout.h"
#include "qmm_core.h"



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

// Scratch for proj_main().  These live in data memory rather than on the stack
// because chess addresses stack slots with an SP-indexed immediate whose range
// runs out a little past 2 KB: b_col_reduce_add alone is
// 2 * INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE bf16, which is 1792 B for
// LLAMA_3_1_8B (INTERMEDIATE_SIZE 14336) and pushes a spill slot to -2052,
// past that range -- the backend then fails to map the bundle
// ("Mist detected bundle(s) with no mappings") instead of degrading.  Both are
// written before they are read on every call, and one core runs one proj_main
// at a time, so there is nothing to carry across calls.
alignas(aie::vector_decl_align) static float y_acc_buf[Q4NX_ROW_BLOCK_SIZE];
// 2 bf16 limbs per group, see the fill loop in linear_proj_iD()
alignas(aie::vector_decl_align) static bfloat16
    b_col_reduce_add_buf[2 * INTERMEDIATE_SIZE / Q4NX_GROUP_SIZE];

#if QUANT_TYPE != __Q41_Q80_MIX__
// template<int M, int K>
void linear_proj_iD(
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

    constexpr int m = Q4NX_ROW_BLOCK_SIZE;
    constexpr int k = Q4NX_COL_BLOCK_SIZE;
    

    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + PKT_ID_OFFSET);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + PKT_ID_OFFSET);

    for (int i = 0; i < M / m; i++) {
        // What carries the running row across the K loop is the format's
        // business: Q4_K keeps two accumulators in registers, Q4_0/Q4_1 use
        // y_acc. _QMM_zero() resets whichever it is.
        qmm_accum_t c_accum;
        _QMM_zero(c_accum, y_acc);
        for (AIE_LOCK_LOOP_CTR j = 0; j < K / k; j++) {
            is_w_ping = !is_w_ping;
            is_x_ping = !is_x_ping;

            QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;
            bf16* x_using = is_x_ping ? x_ping : x_pong;
            _lock_acquire(w_cons_lock);
            _lock_acquire(x_cons_lock);
            #if QUANT_HAS_MIN
            // two bf16 limbs per group, interleaved (hi, lo)
            bf16* b_col_reduce_add_ptr = b_col_reduce_add + 2 * j * (k / Q4NX_GROUP_SIZE);
            // special logic for i == 0, avoid recompute if it is repeat.
            if(i == 0){
                // Fill the b_col_reduce_add.  aie::reduce_add over a bf16
                // vector already sums in accfloat, but returns bf16 and throws
                // 16 of those mantissa bits away; that error then rides the
                // whole min path.  Reduce in float32 instead and hand the
                // kernel two bf16 limbs.  This runs once per projection and is
                // read back by every row block, so in the steady state the
                // split is free.
                AIE_PREPARE_FOR_PIPELINING
                AIE_LOOP_UNROLL_FULL
                for(int l = 0; l < k / Q4NX_GROUP_SIZE; l++)AIE_LOOP_FLATTEN{
                    aie::accum<accfloat, Q4NX_GROUP_SIZE> x_acc;
                    x_acc.from_vector(aie::load_v<Q4NX_GROUP_SIZE>(x_using + Q4NX_GROUP_SIZE * l));
                    float sum = aie::reduce_add(x_acc.template to_vector<float>());
                    bf16 hi = bf16(sum);
                    *(b_col_reduce_add_ptr + 2 * l)     = hi;
                    // Peano's scalar fsub is not exact, so this limb can land
                    // one ulp off chess (see AIE_FSUB).
                    *(b_col_reduce_add_ptr + 2 * l + 1) = bf16(AIE_FSUB(sum, float(hi)));
                }
            }
            #else
            // Q4_0 is symmetric: no min path, so no column sums are needed.
            bf16* b_col_reduce_add_ptr = nullptr;
            #endif
            _QMM_x_bf16<m, k>(c_accum, y_acc, x_using, w_using, b_col_reduce_add_ptr);
            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }
        // the whole row is done: for Q4_K this is where the scale path and the
        // min path finally meet, for Q4_0/Q4_1 it is a no-op
        _QMM_finalize(y_acc, c_accum);
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
            copy_float_to_bf16<m>(y_using + PKT_PAYLOAD_OFFSET, y_acc);
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
            copy_float_to_bf16<m>(y_using + PKT_PAYLOAD_OFFSET + m, y_acc); // second half of the payload
            if (is_y_ping) {
                _down_lock_release(y_cons_ping_lock);
            }
            else {
                _down_lock_release(y_cons_pong_lock);
            }

        }

    }
}
#endif

#if QUANT_TYPE == __Q41_Q80_MIX__
///@brief linear_proj_iD for the mixed format: row blocks [q8_begin, q8_end) of this core are Q8_0 (one 256-column block per
///       weight unit), the others Q4_1 (two 256-column blocks per unit, i.e. two activation slices per unit).
void linear_proj_mix(
    int M, int K,
    bf16 * x_ping, bf16 * x_pong,
    QUANT_BLOCK * w_ping, QUANT_BLOCK * w_pong,
    bf16 * y_ping, bf16 * y_pong,
    float* y_acc,
    bf16 * b_col_reduce_add,
    const int32_t send_x_output,
    const uint32 pkt_id,
    const int q8_begin, const int q8_end
) {
    constexpr int m = Q4NX_ROW_BLOCK_SIZE;
    constexpr int k = Q4NX_COL_BLOCK_SIZE;
    uint32* pkt_id_ping = reinterpret_cast<uint32*>(y_ping + PKT_ID_OFFSET);
    uint32* pkt_id_pong = reinterpret_cast<uint32*>(y_pong + PKT_ID_OFFSET);
    bool reduce_ready = false;   // b_col_reduce_add holds this projection's column sums (filled by its first Q4_1 row block)

    for (int i = 0; i < M / m; i++) {
        qmm_accum_t c_accum;
        _QMM_zero(c_accum, y_acc);
        if (i >= q8_begin && i < q8_end) {
            for (AIE_LOCK_LOOP_CTR j = 0; j < K / k; j++) {
                is_w_ping = !is_w_ping;
                is_x_ping = !is_x_ping;
                QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;
                bf16* x_using = is_x_ping ? x_ping : x_pong;
                _lock_acquire(w_cons_lock);
                _lock_acquire(x_cons_lock);
                _q80_block(c_accum.c, x_using, reinterpret_cast<const q80_block_t*>(w_using));
                _lock_release(w_prod_lock);
                _lock_release(x_prod_lock);
            }
        } else {
            for (AIE_LOCK_LOOP_CTR u = 0; u < K / (2 * k); u++) {
                is_w_ping = !is_w_ping;
                QUANT_BLOCK* w_using = is_w_ping ? w_ping : w_pong;
                _lock_acquire(w_cons_lock);
                for (int h = 0; h < 2; h++) {
                    is_x_ping = !is_x_ping;
                    bf16* x_using = is_x_ping ? x_ping : x_pong;
                    _lock_acquire(x_cons_lock);
                    const int s = 2 * u + h;   // the activation slice, 256 columns
                    bf16* b_col_reduce_add_ptr = b_col_reduce_add + 2 * s * (k / Q4NX_GROUP_SIZE);
                    if (!reduce_ready) {
                        // column sums of the activations, in float32, as two bf16 limbs (see linear_proj_iD)
                        AIE_PREPARE_FOR_PIPELINING
                        AIE_LOOP_UNROLL_FULL
                        for (int l = 0; l < k / Q4NX_GROUP_SIZE; l++) AIE_LOOP_FLATTEN {
                            aie::accum<accfloat, Q4NX_GROUP_SIZE> x_acc;
                            x_acc.from_vector(aie::load_v<Q4NX_GROUP_SIZE>(x_using + Q4NX_GROUP_SIZE * l));
                            float sum = aie::reduce_add(x_acc.template to_vector<float>());
                            bf16 hi = bf16(sum);
                            *(b_col_reduce_add_ptr + 2 * l)     = hi;
                            *(b_col_reduce_add_ptr + 2 * l + 1) = bf16(sum - float(hi));
                        }
                    }
                    _q41_block(c_accum.c, x_using, &w_using->q41[h], b_col_reduce_add_ptr);
                    _lock_release(x_prod_lock);
                }
                _lock_release(w_prod_lock);
            }
            reduce_ready = true;
        }
        _QMM_finalize(y_acc, c_accum);
        is_y_ping = !is_y_ping;

        if(send_x_output != 0){
            uint32* pkt_id_using = is_y_ping ? pkt_id_ping : pkt_id_pong;
            bf16* y_using = is_y_ping ? y_ping : y_pong;
            *pkt_id_using = pkt_id;
            if (is_y_ping) _lock_acquire(y_prod_ping_lock);
            else           _lock_acquire(y_prod_pong_lock);
            copy_float_to_bf16<m>(y_using + PKT_PAYLOAD_OFFSET, y_acc);
            if (is_y_ping) _lock_release(y_cons_ping_lock);
            else           _lock_release(y_cons_pong_lock);
        }else{
            bf16* y_using = is_y_ping ? y_ping : y_pong;
            if (is_y_ping) _down_lock_acquire(y_prod_ping_lock);
            else           _down_lock_acquire(y_prod_pong_lock);
            copy_float_to_bf16<m>(y_using + PKT_PAYLOAD_OFFSET + m, y_acc); // second half of the payload
            if (is_y_ping) _down_lock_release(y_cons_ping_lock);
            else           _down_lock_release(y_cons_pong_lock);
        }
    }
}

// Row blocks of one core within a phase: the host deals 32-row blocks to the MVM_CORES cores round robin, so in the fused q|k|v
// phase (bands K, V, Q in that order, see FLM_DLL/detail/qwen3_asr_npu) a core's blocks are K, then V, then Q.
constexpr int MIX_BLOCKS_K = DK / (MVM_CORES * Q4NX_ROW_BLOCK_SIZE);
constexpr int MIX_BLOCKS_V = DV / (MVM_CORES * Q4NX_ROW_BLOCK_SIZE);
static_assert(DK % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0 && DV % (MVM_CORES * Q4NX_ROW_BLOCK_SIZE) == 0,
              "each band must be whole rounds of row blocks");
static_assert(MODEL_DIM % (2 * Q4NX_COL_BLOCK_SIZE) == 0 && DQ % (2 * Q4NX_COL_BLOCK_SIZE) == 0,
              "Q4_1 units span 512 columns");
#endif

///@brief The four projection phases of one decoder layer.
static void proj_decoder_layer(
    bf16 * y_ping, QUANT_BLOCK * w_ping, bf16 * x_ping,
    bf16 * y_pong, QUANT_BLOCK * w_pong, bf16 * x_pong,
    float* y_acc, bf16 * b_col_reduce_add, int send_x_output
){
#if QUANT_TYPE == __Q41_Q80_MIX__
    constexpr int all = 1 << 20;
    linear_proj_mix((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope,
                    MIX_V_Q8 ? MIX_BLOCKS_K : 0, MIX_V_Q8 ? MIX_BLOCKS_K + MIX_BLOCKS_V : 0);
    linear_proj_mix(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm, 0, 0);
    linear_proj_mix(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu, 0, 0);
    linear_proj_mix(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm,
                    0, MIX_DOWN_Q8 ? all : 0);
#else
    // phase 1: qkv proj, D -> (D + D / 2) to RoPE
    linear_proj_iD((DQ + DK + DV) / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rope);

    // // phase 2: o proj, D -> D to  RMS
    linear_proj_iD(MODEL_DIM / MVM_CORES, DQ, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm);

    // // phase 3: up/gate proj, D -> D * 8 to MT_GLU
    linear_proj_iD(2 * INTERMEDIATE_SIZE / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_glu);

    // // phase 4.1 down-> 4D -> D, first D -> D
    linear_proj_iD(MODEL_DIM / MVM_CORES, INTERMEDIATE_SIZE, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm);
#endif
}


///@brief The LM head projection, D -> padded vocab.
static void proj_lm_head(
    bf16 * y_ping, QUANT_BLOCK * w_ping, bf16 * x_ping,
    bf16 * y_pong, QUANT_BLOCK * w_pong, bf16 * x_pong,
    float* y_acc, bf16 * b_col_reduce_add, int send_x_output
){
#if QUANT_TYPE == __Q41_Q80_MIX__
    linear_proj_mix(VOCAB_SIZE_PADDED / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm,
                    0, MIX_HEAD_Q8 ? (1 << 20) : 0);
#else
    linear_proj_iD(VOCAB_SIZE_PADDED / MVM_CORES, MODEL_DIM, x_ping, x_pong, w_ping, w_pong, y_ping, y_pong, y_acc, b_col_reduce_add, send_x_output, pkt_id_to_rms_norm);
#endif
}


extern "C" {

///@brief One unit of work for a projection core.
///
/// With SELF_SEQUENCED_LM_HEAD (models/<model>.h) one call is one whole token:
/// NUM_HIDDEN_LAYERS decoder layers followed by the LM head, so the core knows
/// which phase it is in from its own position in the loop and IS_ATTN is unused
/// -- the host writes no RTP_PROJ_IS_ATTN_BUFFER_* and never releases lock 6.
/// Without it, one call is one phase and the host names it through IS_ATTN,
/// gated on lock 6.  The parameter stays in the signature either way so the RTP
/// buffer keeps its address; see iron/proj_tile.py.
void proj_main(
    bf16 * y_ping,
    QUANT_BLOCK * w_ping,
    bf16 * x_ping,
    bf16 * y_pong,
    QUANT_BLOCK * w_pong,
    bf16 * x_pong,
    int* IS_ATTN,
    int send_x_output
)
{
    float*    y_acc            = y_acc_buf;
    bfloat16* b_col_reduce_add = b_col_reduce_add_buf;

    // _lock_acquire(rtp_available_lock);
    _lock_acquire(x_cons_lock);
    _lock_release(x_cons_lock);
    if (IS_ATTN[0] == 1) {
        proj_decoder_layer(y_ping, w_ping, x_ping, y_pong, w_pong, x_pong, y_acc, b_col_reduce_add, send_x_output);
    }
    else{
        proj_lm_head(y_ping, w_ping, x_ping, y_pong, w_pong, x_pong, y_acc, b_col_reduce_add, send_x_output);
    }
}


}
