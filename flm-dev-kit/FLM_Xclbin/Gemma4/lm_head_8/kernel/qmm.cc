// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.
// lm_head_8: Gemma4 final vocabulary projection for 8 token rows per dispatch (verify phase of DFlash speculative decoding).
// Same topology, weight layout (Q4NX q4k_block_t, reordered 4 blocks per column) and softcap convention as ../lm_head_npu_bin (L=1);
// the differences are the activation (8 rows resident per core) and the core (q4_v4 of Gemma4/verify_decoding/toys/q4_mvm_L8).
//
// x buffer (per core, bf16), sent once per dispatch by the host in exactly this order:
//   [block 0..K_BLOCKS)[kt 0..32)[t 0..8)[k8 0..8)   8 raw (pre-norm) hidden rows, tokens interleaved per 8 columns (LT*MODEL_DIM elements)
//   nw[MODEL_DIM]                                      model.norm.weight (plain x*w convention of the L=1 kernel)
// The final RMSNorm is applied in place per token, exactly as the L=1 kernel does (x*w, then * 1/sqrt(mean+eps)).
// y per round per core: [t 0..8)[row 0..32) bf16 (after tanh(y/fs)*fs), memtile concatenates the 4 cores: [core][t][32].
#include <aie_api/aie.hpp>
#include "common.h"
#include "aie_kernel_utils.h"
#include "flm_rsqrt.h"

#define GEMMA4_E2B       0
#define GEMMA4_E4B       1

#ifndef MODEL_TYPE
#define MODEL_TYPE GEMMA4_E4B
#endif

constexpr int VOCAB_SIZE = 262144;
constexpr int NUM_CTS    = 32;

#if MODEL_TYPE == GEMMA4_E2B
constexpr int MODEL_DIM  = 1536;
#elif MODEL_TYPE == GEMMA4_E4B
constexpr int MODEL_DIM  = 2560;
#else
#error "Unsupported MODEL_TYPE"
#endif

constexpr int LT = 8;                       // token rows per dispatch
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE     = 32;

constexpr int M_PER_ROUND = NUM_CTS * Q4NX_ROW_BLOCK_SIZE;
constexpr int VOCAB_SIZE_PADDED = ((VOCAB_SIZE + M_PER_ROUND - 1) / M_PER_ROUND) * M_PER_ROUND;
constexpr int MODEL_DIM_PADDED  = ((MODEL_DIM + Q4NX_COL_BLOCK_SIZE - 1) / Q4NX_COL_BLOCK_SIZE) * Q4NX_COL_BLOCK_SIZE;
static_assert(MODEL_DIM == MODEL_DIM_PADDED, "x layout assumes MODEL_DIM is a multiple of 256");

constexpr int ROUNDS   = VOCAB_SIZE_PADDED / M_PER_ROUND;
constexpr int K_BLOCKS = MODEL_DIM_PADDED  / Q4NX_COL_BLOCK_SIZE;
constexpr int X_SLICE  = LT * Q4NX_COL_BLOCK_SIZE;          // 2048 bf16 per K block
constexpr int Y_ELEMS  = LT * Q4NX_ROW_BLOCK_SIZE;          // 256 bf16 per round per core

constexpr int x_prod_lock = 0;
constexpr int x_cons_lock = 1;
constexpr int w_prod_lock = 2;
constexpr int w_cons_lock = 3;
constexpr int y_prod_lock = 4;
constexpr int y_cons_lock = 5;

typedef struct {
    bf16 scales[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    bf16 mins[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE / Q4NX_GROUP_SIZE];
    uint4 qs[Q4NX_ROW_BLOCK_SIZE * Q4NX_COL_BLOCK_SIZE];
} q4k_block_t;

// accumulators [sl 0..4)[t 0..8)[n 0..8) f32, sl = rh*2+nh  <=> rows sl*8 .. sl*8+7 of the 32-row block
alignas(aie::vector_decl_align) static float g_acc[LT * Q4NX_ROW_BLOCK_SIZE] AIE_PERSIST_ZERO;

// final RMSNorm in place on the 8 resident rows (layout [block][kt][t][k8]); same arithmetic as the L=1 kernel.
// Works on 16-lane vectors = two tokens (t=2p, 2p+1) of one (block,kt) group; the per-token sums are folded in scalar code.
static void rms_norm_rows(bf16* x, const bf16* nw) {
    constexpr int NV = MODEL_DIM / 8;                         // number of (block,kt) groups of 8 columns
    // sum of squares per token in scalar fp32 (once per dispatch: ~20k MACs, negligible against 8 ms)
    float ssq[LT];
    for (int t = 0; t < LT; t++) ssq[t] = 0.0f;
    for (int i = 0; i < NV; i++) {
        for (int t = 0; t < LT; t++) {
            float acc = 0.0f;
            for (int j = 0; j < 8; j++) { float v = (float)x[i * (LT * 8) + t * 8 + j]; acc += v * v; }
            ssq[t] += acc;
        }
    }
    alignas(aie::vector_decl_align) float d16[LT / 2][16];   // per pair: lanes 0..7 = 1/rms of token 2p, lanes 8..15 = token 2p+1
    constexpr float one_over_D = 1.0f / (float)MODEL_DIM;
    const float epsilon = 1e-6;
    for (int p = 0; p < LT / 2; p++) {
        for (int h = 0; h < 2; h++) {
            float sum = ssq[p * 2 + h];
            sum = sum * one_over_D;
            sum = sum + epsilon;
            const float d = flm_rsqrt(sum);
            for (int j = 0; j < 8; j++) d16[p][h * 8 + j] = d;
        }
    }
    for (int i = 0; i < NV; i++) {
        aie::vector<bf16, 8> wv8 = aie::load_v<8>(nw + i * 8);
        aie::vector<bf16, 16> wv = aie::concat(wv8, wv8);
        AIE_LOOP_UNROLL_FULL
        for (int p = 0; p < LT / 2; p++) {
            aie::vector<bf16, 16> xv = aie::load_v<16>(x + i * (LT * 8) + p * 16);
            aie::vector<float, 16> wx = aie::mul(xv, wv);
            aie::vector<float, 16> dv = aie::load_v<16>(d16[p]);
            aie::accum<accfloat, 16> o = aie::mul(wx, dv);
            aie::store_v(x + i * (LT * 8) + p * 16, o.template to_vector<bf16>());
        }
    }
}

// one 32x256 Q4_1 block against the 8 x 256 slice of x: dequantise (q*scale+min) 128 lanes at a time, filter_even/odd(.,8) gives the
// two 8-row B tiles, aie::mmul<8,8,8> with tokens as M.  Accumulates into g_acc.  (q4_v4 of toys/q4_mvm_L8, measured 1.9 us/block)
static void q4_block_l8(const q4k_block_t* w, const bf16* x) {
    const uint4* qs_ptr = w->qs;
    aie::mmul<8, 8, 8, bf16, bf16, accfloat> mm[4];
    for (int rh = 0; rh < 2; rh++) {
        for (int s = 0; s < 8; s++) {
            aie::vector<bf16, 16> sc16 = aie::load_v<16>(w->scales + s * 32 + rh * 16);
            aie::vector<bf16, 16> mn16 = aie::load_v<16>(w->mins + s * 32 + rh * 16);
            aie::vector<bf16, 32> sc32 = aie::concat(sc16, sc16);
            aie::vector<bf16, 32> mn32 = aie::concat(mn16, mn16);
            aie::vector<bf16, 64> sc64 = aie::concat(sc32, sc32);
            aie::vector<bf16, 64> mn64 = aie::concat(mn32, mn32);
            aie::vector<bf16, 128> sc128 = aie::concat(sc64, sc64);
            aie::vector<bf16, 128> mn128 = aie::concat(mn64, mn64);
            AIE_LOOP_UNROLL_FULL
            for (int j = 0; j < 4; j++) {
                aie::vector<uint4, 128> q = aie::load_v<128>(qs_ptr);
                qs_ptr += 64;
                aie::vector<float, 128> f = aie::to_float(q, 0);
                aie::accum<accfloat, 128> fa;
                fa.from_vector(f);
                aie::vector<bf16, 128> b = fa.template to_vector<bf16>();
                aie::accum<accfloat, 128> d = aie::mul(b, sc128);
                d = aie::mac(d, mn128, (bf16)1.0f);
                aie::vector<bf16, 128> wd = d.template to_vector<bf16>();
                aie::vector<bf16, 64> B0 = aie::filter_even(wd, 8);
                aie::vector<bf16, 64> B1 = aie::filter_odd(wd, 8);
                aie::vector<bf16, 64> A = aie::load_v<64>(x + (s * 4 + j) * 64);
                if (s == 0 && j == 0) { mm[rh * 2 + 0].mul(A, B0); mm[rh * 2 + 1].mul(A, B1); }
                else { mm[rh * 2 + 0].mac(A, B0); mm[rh * 2 + 1].mac(A, B1); }
            }
        }
    }
    AIE_LOOP_UNROLL_FULL
    for (int sl = 0; sl < 4; sl++) {
        aie::vector<float, 64> cv = mm[sl].template to_vector<float>();
        aie::vector<float, 64> prev = aie::load_v<64>(g_acc + sl * 64);
        aie::store_v(g_acc + sl * 64, aie::add(prev, cv));
    }
}

extern "C" {
void qmm(
    bf16* y_ping, bf16* y_pong,
    q4k_block_t* w_ping, q4k_block_t* w_pong,
    bf16* x,
    float* final_scale
){
    // see ../lm_head_npu_bin/kernel/qmm.cc: persistent toggles must live in .data under Peano
    AIE_PERSIST_ZERO static bool is_y_ping = false;
    AIE_PERSIST_ZERO static bool is_w_ping = false;

    _lock_acquire(x_cons_lock);

    rms_norm_rows(x, x + LT * MODEL_DIM);

    for (int r = 0; r < ROUNDS; r++) {
        AIE_LOOP_UNROLL_FULL
        for (int i = 0; i < LT * Q4NX_ROW_BLOCK_SIZE; i += 16) aie::store_v(g_acc + i, aie::zeros<float, 16>());
        static_assert(K_BLOCKS % 2 == 0, "K_BLOCKS should be even so the ping/pong toggle parity is the same every round");
        for (int k = 0; k < K_BLOCKS; k++) {
            _lock_acquire(w_cons_lock);
            is_w_ping = !is_w_ping;
            q4k_block_t* w_using = is_w_ping ? w_ping : w_pong;
            q4_block_l8(w_using, x + k * X_SLICE);
            _lock_release(w_prod_lock);
        }

        _lock_acquire(y_prod_lock);
        is_y_ping = !is_y_ping;
        bf16* y_using = is_y_ping ? y_ping : y_pong;

        // [sl][t][n8] f32 -> [t][32] bf16
        AIE_LOOP_UNROLL_FULL
        for (int sl = 0; sl < 4; sl++) {
            aie::accum<accfloat, 64> a;
            a.from_vector(aie::load_v<64>(g_acc + sl * 64));
            aie::vector<bf16, 64> v = a.template to_vector<bf16>();
            AIE_LOOP_UNROLL_FULL
            for (int t = 0; t < LT; t++) aie::store_v(y_using + t * Q4NX_ROW_BLOCK_SIZE + sl * 8, v.extract<8>(t));
        }

        // softcap exactly as the L=1 kernel: y = tanh(y / fs) * fs, in bf16 with the global scale applied before and after
        bf16 final_scale_bf16 = (bf16)(*final_scale);
        bf16 final_scale_inv  = aie::inv(final_scale_bf16);
        for (int i = 0; i < Y_ELEMS; i += 16) {
            aie::vector<bf16, 16> y_vec = aie::load_v<16>(y_using + i);
            aie::accum<accfloat, 16> y_accum = aie::mul(y_vec, final_scale_inv);
            y_vec = aie::tanh(y_accum.template to_vector<float>());
            y_vec = aie::mul(y_vec, final_scale_bf16);
            aie::store_v(y_using + i, y_vec);
        }

        _lock_release(y_cons_lock);
    }
    _lock_release(x_prod_lock);
}
}
