// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#include <aie_api/aie.hpp>
#include "common.h"
#include "aie_kernel_utils.h"
#include "q8_core.h"
#include "flm_rsqrt.h"


#define QWEN3_8_27B       0

#ifndef MODEL_TYPE
#define MODEL_TYPE QWEN3_8_27B
#endif


constexpr int VOCAB_SIZE = 248320;
constexpr int NUM_CTS    = 32;

#if MODEL_TYPE == QWEN3_8_27B
constexpr int MODEL_DIM  = 5120;
#else
#error "Unsupported MODEL_TYPE"
#endif

// Token rows per dispatch. The verify phase of speculative decoding scores 8
// candidates at once; the L = 1 variant of this design is ../lm_head_npu_bin.
#ifndef L_TOKENS
#define L_TOKENS 8
#endif
constexpr int L = L_TOKENS;

// x arrives as [k_block][lane][256] with L + 1 lanes: the L token rows, then
// this column block's 256 values of model.norm.weight. The weight rides in the
// activation stream rather than in a buffer of its own because a one-shot fill
// BD is exactly what broke this design's second dispatch -- see the post-mortem
// in README.md. As a lane it is an ordinary per-round operand with no lifetime
// of its own, at a cost of 1/8 more x bytes (10 KB per round per column).
constexpr int X_LANES = L + 1;

constexpr int   M_PER_ROUND = 32 * 32;

constexpr int   VOCAB_SIZE_PADDED = (VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND * M_PER_ROUND;
constexpr int   MODEL_DIM_PADDED  = (MODEL_DIM + QXNX_COL_BLOCK_SIZE - 1) / QXNX_COL_BLOCK_SIZE * QXNX_COL_BLOCK_SIZE;

constexpr int ROUNDS   = VOCAB_SIZE_PADDED / M_PER_ROUND;
constexpr int K_BLOCKS = MODEL_DIM_PADDED  / QXNX_COL_BLOCK_SIZE;

// ---------------------------------------------------------------------------
// rms_norm, without a resident row
//
// The L = 1 design normalises x in place before the round loop, which needs the
// whole 5120-wide row on the core. At L = 8 that is 80 KB against a 64 KB core
// data memory (AIETargetModel.h:453), so x streams in 256-wide column blocks and
// no core ever holds a full row at once. It does, however, *see* every column of
// every row: the k loop below walks all K_BLOCKS, every round. A sum of squares
// is a reduction, so it does not need residency -- only the ordering has to work
// out, and that is the one real obstacle:
//
//     s[t] = 1/sqrt(mean(x[t]^2) + eps)
//
// is not known until the k loop has already consumed x. It does not have to be.
// RMSNorm splits into a per-column factor and a per-token scalar,
//
//     logits[t][m] = SUM_i (x[t][i] * s[t] * nw[i]) * W[m][i]
//                  = s[t] * SUM_i (x[t][i] * nw[i]) * W[m][i]
//
// and the scalar commutes straight through the matmul. So nw is folded into x
// inside _Q80_x_bf16_L, where x is loaded anyway, and s is applied to the
// finished accumulator in copy_y_scaled. Round 0's sum of squares completes at
// exactly the moment round 0's y_acc does, so even the first round scales
// correctly; rounds 1..ROUNDS-1 reuse the cached s.
//
// Note what this does NOT add: no extra pass over x, no resident buffer, and no
// prologue BD. The norm weight rides in as an extra lane of the x stream, so
// every round still looks identical to every other round and the design stays
// re-dispatchable.
// ---------------------------------------------------------------------------

/// Lane-wise partial sums of x^2 for the L token rows of one 256-wide chunk.
/// Called on round 0 only; 16 MACs per token per chunk, once per dispatch.
template<int L_>
void accumulate_ss(aie::accum<accfloat, 16>* ss, const bf16* x) {
    constexpr int F = QXNX_COL_BLOCK_SIZE / 16;
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L_; t++) {
        const bf16* it = x + t * QXNX_COL_BLOCK_SIZE;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(F, F)
        for (int i = 0; i < F; i++)
            ss[t] = aie::mac_square(ss[t], aie::load_v<16>(it + i * 16));
    }
}

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

/// Write the round's L accumulators out, each scaled by its token's
/// reciprocal RMS. This is where the norm's scalar half lands: y_acc was
/// accumulated from un-scaled x, so `scale[t]` has to multiply all M of token
/// t's logits. It costs one extra multiply per 16 values, on a store that
/// already existed.
template<int L_, int m>
void copy_y_scaled(bf16 * y, const float * y_acc, const float * scale) {
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L_; t++) {
        for (int i = 0; i < m / 16; i++){
            aie::vector<float, 16> y_vec = aie::load_v<16>(y_acc + t * m + i * 16);
            aie::vector<bf16, 16> o_vec = aie::mul(y_vec, scale[t]);
            aie::store_v(y + t * m + i * 16, o_vec);
        }
    }
}


extern "C" {
void qmm(
    bf16* y_ping, bf16* y_pong,
    q80_block_t* w_ping, q80_block_t* w_pong,
    bf16* x_ping, bf16* x_pong
){
    // Peano's CDO loader does not zero .bss; keep ping/pong selectors in .data.
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_y_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_w_ping = false;
#if !defined(__chess__)
    __attribute__((section(".data")))
#endif
    static bool is_x_ping = false;

    // L accumulators per 32-row weight block, [token][32].
    alignas(aie::vector_decl_align) float  y_acc[L * QXNX_ROW_BLOCK_SIZE];

    // The norm's scalar half. Accumulated over round 0's k loop -- which is the
    // one pass that sees all MODEL_DIM columns of x before any result leaves the
    // core -- and reused unchanged for the remaining rounds, since every round
    // is fed the same x.
    //
    // These are per dispatch, not static: qmm is re-entered once per dispatch
    // and each dispatch brings its own x.
    alignas(aie::vector_decl_align) aie::accum<accfloat, 16> ss[L];
    float scale[L];
    AIE_LOOP_UNROLL_FULL
    for (int t = 0; t < L; t++) {
        ss[t].from_vector(aie::broadcast<float, 16>(0.0f));
        scale[t] = 1.0f;
    }

    constexpr float ONE_OVER_D = 1.0f / (float)MODEL_DIM;
    constexpr float RMS_EPS    = 1e-6f;

    for (int r = 0; r < ROUNDS; r++) {

        zero_vectorized<float, L * QXNX_ROW_BLOCK_SIZE>(y_acc);

        for (int k = 0; k < K_BLOCKS; k++) {

            // The memtile streams one x slot per round, and a slot is exactly
            // K_BLOCKS of these X_LANES * 256 element chunks, so x advances in
            // lockstep with w. Both are ping/pong from their own DMA channel.
            //
            // Nothing here distinguishes the first round of a dispatch from any
            // other, which is what makes the design re-dispatchable: the core
            // holds no per-dispatch state beyond the ping/pong parity, and that
            // is carried across dispatches consistently because every dispatch
            // runs the same ROUNDS * K_BLOCKS count.
            _lock_acquire(x_cons_lock);
            is_x_ping = !is_x_ping;
            bf16* x_using = is_x_ping ? x_ping : x_pong;

            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            q80_block_t* w_using = is_w_ping ? w_ping : w_pong;

            // Round 0 is the pass that measures x. Only the core's arithmetic
            // branches here -- the DMA traffic is identical in every round, so
            // this does not reintroduce a first-round special case at the level
            // where the second-dispatch bug lived.
            if (r == 0) accumulate_ss<L>(ss, x_using);

            _Q80_x_bf16_L<L>(y_acc, x_using, w_using);

            _lock_release(w_prod_lock);
            _lock_release(x_prod_lock);
        }

        // ...and the reciprocal is ready exactly when round 0's y_acc is, so
        // round 0 needs no correction pass.
        if (r == 0) {
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < L; t++) {
#if defined(__chess__)
                const float ms = aie::reduce_add(ss[t].template to_vector<float>())
                                 * ONE_OVER_D;
                scale[t] = flm_rsqrt(ms + RMS_EPS);
#else
                const float ms = AIE_FMUL(aie::reduce_add(ss[t].template to_vector<float>()),
                                          ONE_OVER_D);
                scale[t] = flm_rsqrt(AIE_FADD(ms, RMS_EPS));
#endif
            }
        }

        _lock_acquire(y_prod_lock);

        is_y_ping = !is_y_ping;
        bf16* y_using = is_y_ping ? y_ping : y_pong;
        // [token][32]: the L rows of one 32-row weight block travel together, so
        // the memtile assembles whole per-round packets and the host sees
        // [round][col][core][token][32].
        copy_y_scaled<L, QXNX_ROW_BLOCK_SIZE>(y_using, y_acc, scale);

        _lock_release(y_cons_lock);
    }
}
}
