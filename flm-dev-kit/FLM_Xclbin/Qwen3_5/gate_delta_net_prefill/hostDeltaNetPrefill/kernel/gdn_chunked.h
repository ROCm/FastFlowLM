// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __GDN_CHUNKED_H__
#define __GDN_CHUNKED_H__

// Chunked GateDeltaNet prefill: one CT block of C = 16 tokens at a time instead
// of one token at a time. Same unit as the per-token kernel (one v-head, one
// 32-wide V slice), same buffers, locks and DMA; only the core function differs.
//
// H = state^T (DK x 32, row c = S_in_chunk[c*32 .. c*32+31]). Per block, with
// D_ij = alpha_{j+1} .. alpha_i (j <= i), E_i = alpha_0 .. alpha_i,
// dC_i = D_{C-1,i}, eC = E_{C-1}:
//
//     A  = strict_lower(beta_i (K K^T)_ij D_ij)
//     Y  = beta V - (beta E) * (K H)
//     Vn = (I + A)^-1 Y                      forward substitution, in place on Y
//     O  = E * (Q H) + lower_incl(Q K^T o D) Vn
//     H <- eC H + K^T (dC * Vn)
//
// which is the per-token recurrence S_t = alpha_t S_{t-1} + beta_t (v_t -
// alpha_t S_{t-1} k_t) k_t^T, o_t = S_t q_t, regrouped. Decay factors are fp32
// running products of alpha, the same products the per-token kernel forms.
//
// Accuracy ("acc" mode, FAST = false): Q, K, V are exact bf16 operands; every
// fp32 operand (H, Vn, A, P and the per-token scalars) enters the bf16 MACs as a
// hi + lo bf16 pair, fp32 accumulate, lo*lo dropped. H is kept between blocks as
// hi/lo bf16 rows in place of the fp32 rows (same 128 B per row) and converted
// back to fp32 before S is sent out. Error ~1e-5 relative, ~200x below the bf16
// rounding of the output. FAST = true uses the hi parts only (one pass).
//
// aie2p has no exact dense bf16 mmul (only bfp16), so everything below runs on
// 32-lane elementwise MACs.

#include <aie_api/aie.hpp>
#include "bf16_limb_math.hpp"
#include "aie_kernel_utils.h"

// Buffers below are reused in place as float and as bf16 (state, Y/Vn, A, P), so
// this file is compiled with -fno-strict-aliasing (makefiles/kernel.mk). Peano
// cannot lower an asm memory barrier, so there is no other fence.

namespace gdn_chunk {

constexpr int C  = 16;   // tokens per block (= CT_L_size)
constexpr int VS = 32;   // V slice per core (= V_chunk)
constexpr int DK = 128;
constexpr int DV = 128;
// Token stride of the Q/K/V CT buffers (padded_QKV_size): DK + 2 (the alpha/beta
// float) rounded up to 32, a constant() in iron/GateDeltaNet_prefill_aie.py.
constexpr int ROW = 160;

// Scratch, 4.8 KB (the core has ~4.9 KB of data memory left).
// Y rows (fp32), then Vn / Vn' rows as [hi 32 | lo 32] bf16, in place.
alignas(64) float gdn_y[C * VS];
// K K^T -> A (fp32, 16 x 16), then split in place: per 32 floats [hi 32 | lo 32].
alignas(64) float gdn_a[C * C];
// Q K^T -> P, same layout as gdn_a.
alignas(64) float gdn_p[C * C];
// Per-token fp32 scalars, one 16-lane row each.
alignas(64) float gdn_sc[5 * C];
// Their bf16 limbs, one 16-lane row each.
alignas(64) bfloat16 gdn_lb[13 * C];

constexpr int SC_ALPHA = 0, SC_BETA = 16, SC_E = 32, SC_BE = 48, SC_DC = 64;
// limb rows of gdn_lb
constexpr int LB_BETA = 0;   // 3 limbs
constexpr int LB_BE   = 3;   // 3 limbs
constexpr int LB_E    = 6;   // 3 limbs (the 3rd only for eC)
constexpr int LB_DC   = 9;   // 2 limbs
constexpr int LB_ES   = 11;  // 2 limbs, E / sqrt(DK)

using acc32 = aie::accum<accfloat, 32>;
using vbf   = aie::vector<bfloat16, 32>;
using vf    = aie::vector<float, 32>;

inline vbf bc(bfloat16 x) { return aie::broadcast<bfloat16, 32>(x); }

inline vbf limb(int row, int t) { return bc(gdn_lb[row * C + t]); }

// Pairwise sum of chunks: out = even chunks of [a | b] + odd chunks of [a | b].
// With chunk = 16, 8, 4, 2 in turn this folds 16 vectors into one, sums of
// vector j in lanes 2j, 2j+1; a last fold(r, r, 1) puts them in lanes 0..15.
inline vf fold(vf a, vf b, unsigned chunk)
{
    auto [x, y] = aie::interleave_unzip(a, b, chunk);
    return aie::add(x, y);
}

// hi/lo split of a 32-float vector, stored as [hi 32 | lo 32] at dst.
inline void store_hilo(bfloat16* dst, vf x)
{
    auto l = bf16_limb::split_f32(x);
    aie::store_v(dst, l.first);
    aie::store_v(dst + 32, l.second);
}

// A / P element (i, j) after the in-place split: hi at the returned index, lo 32 after.
inline int mat_hi(int i, int j) { return 64 * (i >> 1) + 16 * (i & 1) + j; }

// fp32 rows -> [hi | lo] bf16 rows, in place.
inline void state_to_hilo(float* S)
{
    bfloat16* Sb = reinterpret_cast<bfloat16*>(S);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK, DK)
    for (int c = 0; c < DK; c++) {
        vf x = aie::load_v<32>(S + c * VS);
        store_hilo(Sb + c * 64, x);
    }
}

inline void state_from_hilo(float* S)
{
    const bfloat16* Sb = reinterpret_cast<const bfloat16*>(S);
    const vbf one = bf16_limb::ones<32>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(DK, DK)
    for (int c = 0; c < DK; c++) {
        vbf h = aie::load_v<32>(Sb + c * 64);
        vbf l = aie::load_v<32>(Sb + c * 64 + 32);
        acc32 acc = aie::mul(l, one);
        acc = aie::mac(acc, h, one);
        aie::store_v(S + c * VS, acc.template to_vector<float>());
    }
}

// Rows n..C-1 of a partial block become no-op tokens: q = k = v = 0, alpha = 1,
// beta = 0. They then add nothing to O or H, and the decay products stop at n-1.
inline void pad_tail(bfloat16* Q, bfloat16* K, bfloat16* V, int row, int n)
{
    const vbf z = aie::zeros<bfloat16, 32>();
    for (int t = n; t < C; t++) {
        for (int k = 0; k < DK / 32; k++) {
            aie::store_v(Q + t * row + 32 * k, z);
            aie::store_v(K + t * row + 32 * k, z);
            aie::store_v(V + t * row + 32 * k, z);
        }
        *reinterpret_cast<float*>(K + t * row + DK) = 1.0f;
        *reinterpret_cast<float*>(V + t * row + DV) = 0.0f;
    }
}

// One call site; inlined so no call/stack frame sits on the per-block path.
template <bool FAST>
inline __attribute__((always_inline)) void block(const bfloat16* __restrict Q, const bfloat16* __restrict K, const bfloat16* __restrict V,
                  int row, int v_off, float* S, bfloat16* __restrict out, float inv_sqrt_DH)
{
    bfloat16* Sb = reinterpret_cast<bfloat16*>(S);
    bfloat16* yb = reinterpret_cast<bfloat16*>(gdn_y);
    const bfloat16* Ab = reinterpret_cast<const bfloat16*>(gdn_a);
    const bfloat16* Pb = reinterpret_cast<const bfloat16*>(gdn_p);

    // q is not scaled in place (the Q buffer is left as the DMA wrote it): the
    // 1/sqrt(DK) factor goes into P and into the E limbs of pass 2 instead.

    // ---- alpha, beta ----
    for (int t = 0; t < C; t++) {
        gdn_sc[SC_ALPHA + t] = *reinterpret_cast<const float*>(K + t * row + DK);
        gdn_sc[SC_BETA + t]  = *reinterpret_cast<const float*>(V + t * row + DV);
    }

    // ---- K K^T, Q K^T rows, exact bf16 products ----
    // Row i: the 16 dot products (32-lane partial sums each) are folded into one
    // 16-lane vector by a shuffle-add tree instead of 16 separate reductions.
    // Entries j > i come out 0 or are masked by the decay rows below.
    for (int i = 0; i < C; i++) {
        const bfloat16* ki = K + i * row;
        const bfloat16* qi = Q + i * row;
        vf gk[4], gq[4];
        AIE_LOOP_UNROLL_FULL
        for (int jj = 0; jj < 4; jj++) {
            if (4 * jj > i) {
                gk[jj] = aie::zeros<float, 32>();
                gq[jj] = aie::zeros<float, 32>();
                continue;
            }
            vf vk[4], vq[4];
            AIE_LOOP_UNROLL_FULL
            for (int u = 0; u < 4; u++) {
                const bfloat16* kj = K + (4 * jj + u) * row;
                vbf kj0 = aie::load_v<32>(kj);
                acc32 ak = aie::mul(aie::load_v<32>(ki), kj0);
                acc32 aq = aie::mul(aie::load_v<32>(qi), kj0);
                AIE_LOOP_UNROLL_FULL
                for (int k = 1; k < DK / 32; k++) {
                    vbf kjk = aie::load_v<32>(kj + 32 * k);
                    ak = aie::mac(ak, aie::load_v<32>(ki + 32 * k), kjk);
                    aq = aie::mac(aq, aie::load_v<32>(qi + 32 * k), kjk);
                }
                vk[u] = ak.template to_vector<float>();
                vq[u] = aq.template to_vector<float>();
            }
            gk[jj] = fold(fold(vk[0], vk[1], 16), fold(vk[2], vk[3], 16), 8);
            gq[jj] = fold(fold(vq[0], vq[1], 16), fold(vq[2], vq[3], 16), 8);
        }
        vf rk = fold(fold(gk[0], gk[1], 4), fold(gk[2], gk[3], 4), 2);
        vf rq = fold(fold(gq[0], gq[1], 4), fold(gq[2], gq[3], 4), 2);
        aie::store_v(gdn_a + 16 * i, fold(rk, rk, 1).template extract<16>(0));
        aie::store_v(gdn_p + 16 * i, fold(rq, rq, 1).template extract<16>(0));
    }

    // ---- decay rows D_i, A = beta (KK o D), P = QK o D, E, dC ----
    {
        const aie::vector<float, 16> one16 = aie::broadcast<float, 16>(1.0f);
        aie::vector<float, 16> d = aie::zeros<float, 16>();
        alignas(64) float e_raw[16];
        for (int i = 0; i < C; i++) {
            d = aie::mul(d, gdn_sc[SC_ALPHA + i]).template to_vector<float>();
            d = aie::select(d, one16, aie::mask<16>::from_uint32(1u << i));
            e_raw[i] = d[0];                                   // alpha_1 .. alpha_i
            aie::vector<float, 16> kk = aie::load_v<16>(gdn_a + 16 * i);
            aie::vector<float, 16> qk = aie::load_v<16>(gdn_p + 16 * i);
            aie::vector<float, 16> kd = aie::mul(kk, d).template to_vector<float>();
            aie::store_v(gdn_a + 16 * i, aie::mul(kd, gdn_sc[SC_BETA + i]).template to_vector<float>());
            aie::store_v(gdn_p + 16 * i, aie::mul(aie::mul(qk, d).template to_vector<float>(), inv_sqrt_DH).template to_vector<float>());
        }
        aie::store_v(gdn_sc + SC_DC, d);                       // dC_i = D_{C-1,i}
        aie::vector<float, 16> e = aie::mul(aie::load_v<16>(e_raw), gdn_sc[SC_ALPHA]).template to_vector<float>();
        aie::store_v(gdn_sc + SC_E, e);
        aie::store_v(gdn_sc + SC_BE, aie::mul(e, aie::load_v<16>(gdn_sc + SC_BETA)).template to_vector<float>());
    }

    // ---- splits: A, P in place; scalar limbs ----
    {
        AIE_LOOP_UNROLL_FULL
        for (int k = 0; k < C * C / 32; k++) {
            vf a = aie::load_v<32>(gdn_a + 32 * k);
            store_hilo(reinterpret_cast<bfloat16*>(gdn_a) + 64 * k, a);
            vf p = aie::load_v<32>(gdn_p + 32 * k);
            store_hilo(reinterpret_cast<bfloat16*>(gdn_p) + 64 * k, p);
        }
        auto l3 = [](int sc, int lb, int nl) {
            bf16_limb::limbs3<C> l = bf16_limb::split_f32_3(aie::load_v<C>(gdn_sc + sc));
            aie::store_v(gdn_lb + lb * C, l.l0);
            aie::store_v(gdn_lb + (lb + 1) * C, l.l1);
            if (nl > 2) aie::store_v(gdn_lb + (lb + 2) * C, l.l2);
        };
        l3(SC_BETA, LB_BETA, 3);
        l3(SC_BE, LB_BE, 3);
        l3(SC_E, LB_E, 3);
        l3(SC_DC, LB_DC, 2);
        // E / sqrt(DK) for pass 2
        bf16_limb::limbs3<C> es = bf16_limb::split_f32_3(
            aie::mul(aie::load_v<C>(gdn_sc + SC_E), inv_sqrt_DH).template to_vector<float>());
        aie::store_v(gdn_lb + LB_ES * C, es.l0);
        aie::store_v(gdn_lb + (LB_ES + 1) * C, es.l1);
    }

    // ---- pass 1: K H, Y = beta V - (beta E) K H ----
    // One accumulator per token (hi and lo products summed in it): 4 live
    // accumulators fit the register file, 8 spill.
    for (int tg = 0; tg < C; tg += 4) {
        acc32 ah[4];
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++) ah[u] = aie::zeros<accfloat, 32>();
        const bfloat16* k0 = K + tg * row;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(DK, DK)
        for (int c = 0; c < DK; c++) {
            vbf hh = aie::load_v<32>(Sb + c * 64);
            vbf hl;
            if constexpr (!FAST) hl = aie::load_v<32>(Sb + c * 64 + 32);
            AIE_LOOP_UNROLL_FULL
            for (int u = 0; u < 4; u++) {
                vbf kb = bc(k0[u * row + c]);
                if constexpr (!FAST) ah[u] = aie::mac(ah[u], kb, hl);
                ah[u] = aie::mac(ah[u], kb, hh);
            }
        }
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++) {
            const int t = tg + u;
            vbf v = aie::load_v<32>(V + t * row + v_off);
            acc32 y;
            if constexpr (FAST) {
                y = aie::mul(limb(LB_BETA, t), v);
                y = aie::msc(y, limb(LB_BE, t), ah[u].template to_vector<bfloat16>());
            } else {
                vf kh = ah[u].template to_vector<float>();
                bf16_limb::limbs3<32> b{limb(LB_BETA, t), limb(LB_BETA + 1, t), limb(LB_BETA + 2, t)};
                bf16_limb::limbs3<32> be{limb(LB_BE, t), limb(LB_BE + 1, t), limb(LB_BE + 2, t)};
                y = aie::zeros<accfloat, 32>();
                y = bf16_limb::mac_limbs_bf16(y, b, v);
                y = bf16_limb::msc_limbs3_f32_tri(y, be, kh);
            }
            aie::store_v(gdn_y + t * VS, y.template to_vector<float>());
        }
    }

    // ---- Vn = (I + A)^-1 Y, row by row, in place as [hi | lo] ----
    for (int i = 0; i < C; i++) {
        acc32 acc;
        acc.from_vector(aie::load_v<32>(gdn_y + i * VS));
        acc32 acl = aie::zeros<accfloat, 32>();
        for (int j = 0; j < i; j++) {
            const int f = mat_hi(i, j);
            vbf ah = bc(Ab[f]);
            vbf vh = aie::load_v<32>(yb + j * 64);
            if constexpr (!FAST) {
                vbf al = bc(Ab[f + 32]);
                vbf vl = aie::load_v<32>(yb + j * 64 + 32);
                acl = aie::msc(acl, al, vh);
                acl = aie::msc(acl, ah, vl);
            }
            acc = aie::msc(acc, ah, vh);
        }
        vf x = acc.template to_vector<float>();
        if constexpr (!FAST) x = aie::add(acl.template to_vector<float>(), x);
        store_hilo(yb + i * 64, x);
    }

    // ---- pass 2: O = E (Q H) + P Vn ----
    for (int tg = 0; tg < C; tg += 4) {
        acc32 ah[4];
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++) ah[u] = aie::zeros<accfloat, 32>();
        const bfloat16* q0 = Q + tg * row;
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(DK, DK)
        for (int c = 0; c < DK; c++) {
            vbf hh = aie::load_v<32>(Sb + c * 64);
            vbf hl;
            if constexpr (!FAST) hl = aie::load_v<32>(Sb + c * 64 + 32);
            AIE_LOOP_UNROLL_FULL
            for (int u = 0; u < 4; u++) {
                vbf qb = bc(q0[u * row + c]);
                if constexpr (!FAST) ah[u] = aie::mac(ah[u], qb, hl);
                ah[u] = aie::mac(ah[u], qb, hh);
            }
        }
        AIE_LOOP_UNROLL_FULL
        for (int u = 0; u < 4; u++) {
            const int t = tg + u;
            acc32 o, ol = aie::zeros<accfloat, 32>();
            if constexpr (FAST) {
                o = aie::mul(limb(LB_ES, t), ah[u].template to_vector<bfloat16>());
            } else {
                vf qh = ah[u].template to_vector<float>();
                auto q = bf16_limb::split_f32(qh);
                o = aie::mul(limb(LB_ES + 1, t), q.first);
                o = aie::mac(o, limb(LB_ES, t), q.second);
                o = aie::mac(o, limb(LB_ES, t), q.first);
            }
            for (int j = 0; j <= t; j++) {
                const int f = mat_hi(t, j);
                vbf ph = bc(Pb[f]);
                vbf vh = aie::load_v<32>(yb + j * 64);
                if constexpr (!FAST) {
                    vbf pl = bc(Pb[f + 32]);
                    vbf vl = aie::load_v<32>(yb + j * 64 + 32);
                    ol = aie::mac(ol, pl, vh);
                    ol = aie::mac(ol, ph, vl);
                }
                o = aie::mac(o, ph, vh);
            }
            vf of = o.template to_vector<float>();
            if constexpr (!FAST) of = aie::add(ol.template to_vector<float>(), of);
            acc32 oa;
            oa.from_vector(of);
            aie::store_v(out + t * VS, oa.template to_vector<bfloat16>());
        }
    }

    // ---- Vn' = dC * Vn, in place ----
    for (int t = 0; t < C; t++) {
        vbf vh = aie::load_v<32>(yb + t * 64);
        acc32 acc;
        if constexpr (FAST) {
            acc = aie::mul(limb(LB_DC, t), vh);
        } else {
            vbf vl = aie::load_v<32>(yb + t * 64 + 32);
            acc = aie::mul(limb(LB_DC + 1, t), vh);
            acc = aie::mac(acc, limb(LB_DC, t), vl);
            acc = aie::mac(acc, limb(LB_DC, t), vh);
        }
        store_hilo(yb + t * 64, acc.template to_vector<float>());
    }

    // ---- pass 3: H <- eC H + K^T Vn', 4 rows at a time (4 independent MAC chains) ----
    {
        const vbf e0 = limb(LB_E, C - 1), e1 = limb(LB_E + 1, C - 1), e2 = limb(LB_E + 2, C - 1);
        AIE_LOOP_RANGE(DK / 4, DK / 4)
        for (int c = 0; c < DK; c += 4) {
            acc32 a[4];
            AIE_LOOP_UNROLL_FULL
            for (int r = 0; r < 4; r++) {
                vbf hh = aie::load_v<32>(Sb + (c + r) * 64);
                if constexpr (FAST) {
                    a[r] = aie::mul(e0, hh);
                } else {
                    vbf hl = aie::load_v<32>(Sb + (c + r) * 64 + 32);
                    a[r] = aie::mul(e2, hh);
                    a[r] = aie::mac(a[r], e1, hl);
                    a[r] = aie::mac(a[r], e1, hh);
                    a[r] = aie::mac(a[r], e0, hl);
                    a[r] = aie::mac(a[r], e0, hh);
                }
            }
            const bfloat16* kc = K + c;
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_RANGE(C, C)
            for (int t = 0; t < C; t++) {
                vbf vh = aie::load_v<32>(yb + t * 64);
                vbf vl;
                if constexpr (!FAST) vl = aie::load_v<32>(yb + t * 64 + 32);
                AIE_LOOP_UNROLL_FULL
                for (int r = 0; r < 4; r++) {
                    vbf kb = bc(kc[t * row + r]);
                    if constexpr (!FAST) a[r] = aie::mac(a[r], kb, vl);
                    a[r] = aie::mac(a[r], kb, vh);
                }
            }
            AIE_LOOP_UNROLL_FULL
            for (int r = 0; r < 4; r++) store_hilo(Sb + (c + r) * 64, a[r].template to_vector<float>());
        }
    }
}

// Same as determine_ping_pong_buffer in the kernel .cc.
inline bfloat16* pick(bfloat16* ping, bfloat16* pong, int* flag)
{
    if (*flag == 0) {
        *flag = 1;
        return ping;
    }
    *flag = 0;
    return pong;
}

// Same contract as the per-token kernel body in
// gate_delta_net_prefill_128_128_128_32_16_128.cc.
template <bool FAST, int CT_L_size, int MT_L_size>
inline __attribute__((always_inline)) void run(float* S_in_chunk,
                bfloat16* Q_ping, bfloat16* Q_pong, bfloat16* K_ping, bfloat16* K_pong,
                bfloat16* V_ping, bfloat16* V_pong, bfloat16* out_ping, bfloat16* out_pong,
                int row, int chunk_idx, int* ping_pong_flag, int* rtp_buffer,
                int S_IN_prod, int S_IN_con, int Q_prod, int Q_con, int K_prod, int K_con,
                int V_prod, int V_con, int O_prod, int O_con,
                float inv_sqrt_DH)
{
    static_assert(CT_L_size == C, "block size must equal the CT buffer length");
    aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);

    constexpr int iter_for_send_S = (VS * DK * sizeof(float)) / (CT_L_size * VS * sizeof(bfloat16));
    constexpr int S_elements_per_iteration = (CT_L_size * VS * sizeof(bfloat16)) / sizeof(float);

    for (int head_idx = 0; head_idx < rtp_buffer[0]; head_idx++) {
        acquire_greater_equal(S_IN_con, 1);
        state_to_hilo(S_in_chunk);

        for (int MT_L_idx = 0; MT_L_idx < rtp_buffer[1]; MT_L_idx += MT_L_size) {
            for (int ct = 0; ct < MT_L_size; ct += CT_L_size) {
                const int L_idx = MT_L_idx + ct;
                bfloat16* q = pick(Q_ping, Q_pong, &ping_pong_flag[1]);
                bfloat16* k = pick(K_ping, K_pong, &ping_pong_flag[2]);
                bfloat16* v = pick(V_ping, V_pong, &ping_pong_flag[3]);
                bfloat16* o = pick(out_ping, out_pong, &ping_pong_flag[4]);
                const int n = (L_idx + CT_L_size <= rtp_buffer[1]) ? CT_L_size : (rtp_buffer[1] - L_idx);

                acquire_greater_equal(Q_con, 1);
                acquire_greater_equal(K_con, 1);
                acquire_greater_equal(V_con, 1);
                acquire_greater_equal(O_prod, 1);
                if (n > 0) {
                    // ROW, not row: a constant stride turns the strided scalar loads into fixed offsets
                    if (n < C) pad_tail(q, k, v, ROW, n);
                    block<FAST>(q, k, v, ROW, chunk_idx * VS, S_in_chunk, o, inv_sqrt_DH);
                }
                release(O_con, 1);
                release(V_prod, 1);
                release(K_prod, 1);
                release(Q_prod, 1);
            }
        }

        state_from_hilo(S_in_chunk);
        for (int i = 0; i < iter_for_send_S; i++) {
            bfloat16* o = pick(out_ping, out_pong, &ping_pong_flag[4]);
            acquire_greater_equal(O_prod, 1);
            copy_float_to_float<S_elements_per_iteration, 32>((float*)o, S_in_chunk + i * S_elements_per_iteration);
            release(O_con, 1);
        }
        release(S_IN_prod, 1);
    }

    aie::set_rounding(saved);
}

} // namespace gdn_chunk

#endif // __GDN_CHUNKED_H__
