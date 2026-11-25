// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// bfp16 GEMM core: 64 rows x 128 columns of one round, k in chunks of 128 (see iron/gemm_aie.py).
// Include with GEMM_EVEN 1 (the pair's lower core, holds the lo weight halves) or 0 (upper core, hi halves).
#pragma once
#include "aie_kernel_utils.h"
#include "lut_based_ops.h"

using MMUL = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, accauto>;

// core locks, as in iron/gemm_tile.py
constexpr int W_PROD0 = 0, W_CONS0 = 1, W_PROD1 = 2, W_CONS1 = 3;
constexpr int X_PROD = 4, X_CONS = 5, Y_PROD = 6, Y_CONS = 7;

constexpr unsigned KT = 16;          // 8-k tiles per 128 k chunk
constexpr unsigned UNR = 2, II = 8;
constexpr int W_TILES_BYTES = 9216;  // 64 columns x 128 k; the 64 bf16 biases of those columns follow
constexpr int STRIP_FLOATS = 2048;   // 16 rows x 128 columns

// parity of the next weight buffer; a call may take an odd number of chunks, so it carries over
AIE_PING_FLAG w_pong = false;

// one 2x2 block of 8x8 tiles over the KT k tiles of a chunk; acc = 4 x 64 floats, contiguous.
// x: [KT][rows 0-7 | rows 8-15] tiles; w: [jp 4][KT][2 column tiles], w_tile0 selects jp.
template <bool FIRST>
static inline __attribute__((always_inline)) void pass2x2(float *__restrict acc, const bfp16ebs8 *__restrict px, const bfp16ebs8 *__restrict pw, unsigned w_tile0) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> xs(px);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ws(pw);
    ws.seek(w_tile0);
    MMUL Y00, Y01, Y10, Y11;
    if constexpr (FIRST) {
        Y00 = MMUL(aie::zeros<accfloat, 64>()); Y01 = MMUL(aie::zeros<accfloat, 64>());
        Y10 = MMUL(aie::zeros<accfloat, 64>()); Y11 = MMUL(aie::zeros<accfloat, 64>());
    } else {
        Y00 = MMUL(aie::load_v<64>(acc));
        Y01 = MMUL(aie::load_v<64>(acc + 64));
        Y10 = MMUL(aie::load_v<64>(acc + 128));
        Y11 = MMUL(aie::load_v<64>(acc + 192));
    }
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(KT / UNR, KT / UNR)
    AIE_TRY_INITIATION_INTERVAL(II)
    for (unsigned i = 0; i < KT / UNR; ++i) {
#pragma unroll
        for (unsigned u = 0; u < UNR; ++u) {
            auto X0 = xs.pop(); auto X1 = xs.pop();
            auto W0 = ws.pop(); auto W1 = ws.pop();
            Y00.mac(X0, aie::op_transpose(W0));
            Y01.mac(X0, aie::op_transpose(W1));
            Y10.mac(X1, aie::op_transpose(W0));
            Y11.mac(X1, aie::op_transpose(W1));
        }
    }
    aie::store_v(acc, Y00.template to_vector<float>());
    aie::store_v(acc + 64, Y01.template to_vector<float>());
    aie::store_v(acc + 128, Y10.template to_vector<float>());
    aie::store_v(acc + 192, Y11.template to_vector<float>());
}

// one 16-row strip against the 128 columns of the chunk: 8 blocks of 16 x 16 (4 lo, 4 hi), 256 floats each
template <bool FIRST>
static inline void strip_pass(float *__restrict acc, const uint8 *__restrict x, const uint8 *__restrict w_lo, const uint8 *__restrict w_hi) {
    for (unsigned jp = 0; jp < 4; jp++) pass2x2<FIRST>(acc + jp * 256, (const bfp16ebs8 *)x, (const bfp16ebs8 *)w_lo, jp * 2 * KT);
    for (unsigned jp = 0; jp < 4; jp++) pass2x2<FIRST>(acc + (4 + jp) * 256, (const bfp16ebs8 *)x, (const bfp16ebs8 *)w_hi, jp * 2 * KT);
}

static inline __attribute__((always_inline)) aie::vector<bfloat16, 64> to_bf16(const aie::vector<float, 64> &v) {
    aie::accum<accfloat, 64> a;
    a.from_vector(v);
    return a.to_vector<bfloat16>();
}

// fp32 strip -> bf16, + bias, optional GELU, written in place as eight row-major 16 x 16 blocks at 512 B steps
// (block p of the strip's floats starts at 1024 p, so every block overwrites only blocks already done)
static inline void strip_epilogue(float *acc, const bfloat16 *bias_lo, const bfloat16 *bias_hi, int act) {
    bfloat16 *out = (bfloat16 *)acc;
    for (int p = 0; p < 8; p++) {
        const aie::vector<bfloat16, 16> b16 = aie::load_v<16>((p < 4 ? bias_lo : bias_hi) + 16 * (p & 3));
        const aie::vector<bfloat16, 8> b0 = b16.extract<8>(0), b1 = b16.extract<8>(1);
        // tile element (m, n) sits at 8 m + n, so the 8 column biases repeat 8 times
        const aie::vector<float, 64> bias0 = aie::accum<accfloat, 64>(aie::concat(b0, b0, b0, b0, b0, b0, b0, b0)).to_vector<float>();
        const aie::vector<float, 64> bias1 = aie::accum<accfloat, 64>(aie::concat(b1, b1, b1, b1, b1, b1, b1, b1)).to_vector<float>();
        const float *src = acc + 256 * p;
        aie::vector<bfloat16, 64> v00 = to_bf16(aie::add(aie::load_v<64>(src), bias0));
        aie::vector<bfloat16, 64> v01 = to_bf16(aie::add(aie::load_v<64>(src + 64), bias1));
        aie::vector<bfloat16, 64> v10 = to_bf16(aie::add(aie::load_v<64>(src + 128), bias0));
        aie::vector<bfloat16, 64> v11 = to_bf16(aie::add(aie::load_v<64>(src + 192), bias1));
        if (act == 1) {
            v00 = getGeluBf16_nonLUT<64>(v00); v01 = getGeluBf16_nonLUT<64>(v01);
            v10 = getGeluBf16_nonLUT<64>(v10); v11 = getGeluBf16_nonLUT<64>(v11);
        }
        // rows 0-7: [y00 row | y01 row], rows 8-15: [y10 row | y11 row]
        const auto [z0, z1] = aie::interleave_zip(v00, v01, 8);
        const auto [z2, z3] = aie::interleave_zip(v10, v11, 8);
        bfloat16 *o = out + 256 * p;
        aie::store_v(o, z0);
        aie::store_v(o + 64, z1);
        aie::store_v(o + 128, z2);
        aie::store_v(o + 192, z3);
    }
}

// weight buffer b of both halves: the lo half lives in the even core, the hi half in the odd core
static inline __attribute__((always_inline)) void w_acquire(bool b) {
#if GEMM_EVEN
    if (b) { _lock_acquire(W_CONS1); _up_lock_acquire(W_CONS1); } else { _lock_acquire(W_CONS0); _up_lock_acquire(W_CONS0); }
#else
    if (b) { _down_lock_acquire(W_CONS1); _lock_acquire(W_CONS1); } else { _down_lock_acquire(W_CONS0); _lock_acquire(W_CONS0); }
#endif
}

static inline __attribute__((always_inline)) void w_release(bool b) {
#if GEMM_EVEN
    if (b) { _lock_release(W_PROD1); _up_lock_release(W_PROD1); } else { _lock_release(W_PROD0); _up_lock_release(W_PROD0); }
#else
    if (b) { _down_lock_release(W_PROD1); _lock_release(W_PROD1); } else { _down_lock_release(W_PROD0); _lock_release(W_PROD0); }
#endif
}

// RTP: [0] k chunks of 128 (KC), [1] rounds of 256 rows in this call (all N blocks), [2] activation (0 none, 1 GELU)
static inline void gemm_body(float *acc, uint8 *x0, uint8 *x1, uint8 *w_lo0, uint8 *w_lo1, uint8 *w_hi0, uint8 *w_hi1, int *rtp) {
    const aie::rounding_mode old_rnd = aie::swap_rounding(aie::rounding_mode::conv_even);
    bool b = w_pong;
    // the first weight chunk arrives only after the host sequence has written the RTPs
    w_acquire(b);
    const int kc_n = ((volatile int *)rtp)[0];
    const int rounds = ((volatile int *)rtp)[1];
    const int act = ((volatile int *)rtp)[2];
    for (int t = 0; t < rounds; t++) {
        for (int kc = 0; kc < kc_n; kc++) {
            if (t != 0 || kc != 0) w_acquire(b);
            const uint8 *w_lo = b ? w_lo1 : w_lo0;
            const uint8 *w_hi = b ? w_hi1 : w_hi0;
            for (int s = 0; s < 4; s++) {
                float *a = acc + STRIP_FLOATS * s;
                _lock_acquire(X_CONS);
                const uint8 *x = (s & 1) ? x1 : x0;
                if (kc == 0) {
                    _lock_acquire(Y_PROD);
                    strip_pass<true>(a, x, w_lo, w_hi);
                } else {
                    strip_pass<false>(a, x, w_lo, w_hi);
                }
                _lock_release(X_PROD);
                if (kc == kc_n - 1) {
                    strip_epilogue(a, (const bfloat16 *)(w_lo + W_TILES_BYTES), (const bfloat16 *)(w_hi + W_TILES_BYTES), act);
                    _lock_release(Y_CONS);
                }
            }
            w_release(b);
            b = !b;
        }
    }
    w_pong = b;
    aie::set_rounding(old_rnd);
}
