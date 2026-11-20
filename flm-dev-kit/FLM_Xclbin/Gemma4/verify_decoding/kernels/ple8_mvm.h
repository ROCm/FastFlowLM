// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __PLE8_MVM_H__
#define __PLE8_MVM_H__
// 8 token bf16 matrix-vector block for the PLE stage: y_acc[t][0..32) += W[32 rows][K cols] . x[t][0..K) for t = 0..7, with W stored as in
// kernels/bf16_proj.h ([k][32 rows], rows contiguous).  Same per token arithmetic and accumulation order as _mvm_bf16_bf16<32, K> (so the
// result is bit-identical to running it once per token), but PLE8_NT tokens run interleaved inside the k loop: one weight column load feeds
// NT vector x scalar MACs and the MAC latency is hidden by the other tokens.  Measured on hardware (gate stage, W_gate 1.31 MB, stream-only
// floor 172 us): NT=1 406 us, NT=2 249 us, NT=4 2160 us (accumulators spill), 8 tokens at once 640 us.
#include "aie_kernel_utils.h"
#include "model_spec.h"

// NT tokens interleaved (NT = 1, 2, 4): the 8 token block is run as 8 / NT passes over the weights block.
template<int K, int NT>
void ple8_mvm_nt(const bf16* __restrict w, const bf16* __restrict x, float* __restrict y_acc) {
    constexpr int M = 32, XV = 32;
    static_assert(NT == 1 || NT == 2 || NT == 4, "NT");
    aie::accum<accfloat, M> acc[NT];
    for (int t = 0; t < NT; t++) acc[t].from_vector(aie::load_v<M>(y_acc + t * M));
    const bf16* w_ptr = w;
    for (int i = 0; i < K / XV; i++) {
        aie::vector<bf16, XV> xv[NT];
        for (int t = 0; t < NT; t++) xv[t] = aie::load_v<XV>(x + t * K + i * XV);
        for (int j = 0; j < XV; j++) {
            aie::vector<bf16, M> w_col = aie::load_v<M>(w_ptr);
            w_ptr += M;
            for (int t = 0; t < NT; t++) acc[t] = aie::mac(acc[t], w_col, xv[t][j]);
        }
    }
    for (int t = 0; t < NT; t++) aie::store_v(y_acc + t * M, acc[t].template to_vector<float>());
}


// 64-lane accumulators holding two tokens each ([t even: 32 rows | t odd: 32 rows]): per column k one weight column (duplicated to 64
// lanes) and PT/2 MACs against [x(t)[k] x32 | x(t+1)[k] x32], column loop unrolled so the x lanes are extracted statically.  PT/2
// independent accumulation chains hide the MAC latency; per output the order is still k ascending, as in _mvm_bf16_bf16 (bit-identical).
// Toy cycle counts (one [128 k][32 rows] block x 8 tokens, Peano, slope of repeat counts): NT=2 1.10 us, PT=4 0.99 us, PT=8 0.91 us.
template<int K, int PT>
void ple8_mvm_pair_u(const bf16* __restrict w, const bf16* __restrict x, float* __restrict y_acc) {
    static_assert(PT == 4 || PT == 8, "PT");
    for (int t0 = 0; t0 < 8; t0 += PT) {
        aie::accum<accfloat, 64> acc[PT / 2];
        for (int p = 0; p < PT / 2; p++) acc[p].from_vector(aie::load_v<64>(y_acc + (t0 / 2 + p) * 64));
        const bf16* w_ptr = w;
        for (int i = 0; i < K / 32; i++) {
            aie::vector<bf16, 32> xv[PT];
            for (int t = 0; t < PT; t++) xv[t] = aie::load_v<32>(x + (t0 + t) * K + i * 32);
            AIE_LOOP_UNROLL_FULL
            for (int j = 0; j < 32; j++) {
                aie::vector<bf16, 32> wc = aie::load_v<32>(w_ptr + j * 32);
                aie::vector<bf16, 64> w64 = aie::concat(wc, wc);
                AIE_LOOP_UNROLL_FULL
                for (int p = 0; p < PT / 2; p++) {
                    aie::vector<bf16, 64> xb = aie::concat(aie::broadcast<bf16, 32>(xv[2 * p][j]), aie::broadcast<bf16, 32>(xv[2 * p + 1][j]));
                    acc[p] = aie::mac(acc[p], w64, xb);
                }
            }
            w_ptr += 32 * 32;
        }
        for (int p = 0; p < PT / 2; p++) aie::store_v(y_acc + (t0 / 2 + p) * 64, acc[p].template to_vector<float>());
    }
}

#ifndef PLE8_NT
#define PLE8_NT 2
#endif
/// \brief the 8 token block as 8 / PLE8_NT passes
template<int K>
void ple8_mvm_block(const bf16* __restrict w, const bf16* __restrict x, float* __restrict y_acc) {
#ifdef PLE8_MVM_NT
    for (int t0 = 0; t0 < 8; t0 += PLE8_NT) ple8_mvm_nt<K, PLE8_NT>(w, x + t0 * K, y_acc + t0 * 32);
#else
    ple8_mvm_pair_u<K, 8>(w, x, y_acc);
#endif
}
#endif
