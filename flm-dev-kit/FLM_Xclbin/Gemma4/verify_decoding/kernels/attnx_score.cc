// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// attnx score tile: q.K^T, masked online softmax, P to the two PV tiles of this tile's tokens.  Layouts: kernels/attnx_common.h.
#include "lut_based_ops.h"
#include "attnx_common.h"

using namespace attnx;

constexpr int Q_PROD = 0;
constexpr int Q_CONS = 1;
constexpr int K_PROD = 2;
constexpr int K_CONS = 3;
constexpr int PA_PROD = 4;
constexpr int PA_CONS = 5;
constexpr int PB_PROD = 6;
constexpr int PB_CONS = 7;

using MMQ = aie::mmul<8, 8, 8, bfloat16, bfp16ebs8>;

static inline aie::block_vector<bfp16ebs8, 64> to_bfp(const aie::vector<bf16, 64>& v) {
    aie::accum<accfloat, 64> a(v);
    return a.template to_vector<bfp16ebs8>();
}

/// exp(x) of 64 lanes, clamped like the decode kernels, bf16 out
static inline aie::vector<bf16, 64> exp64(const aie::vector<float, 64>& x) {
    aie::vector<float, 64> d = aie::max(aie::min(x, 88.0f), -87.0f);
    aie::accum<accfloat, 64> a;
    a.from_vector(d);
    aie::vector<bf16, 64> b = a.template to_vector<bf16>();
    aie::accum<accfloat, 16> e0 = getExpBf16(b.extract<16>(0));
    aie::accum<accfloat, 16> e1 = getExpBf16(b.extract<16>(1));
    aie::accum<accfloat, 16> e2 = getExpBf16(b.extract<16>(2));
    aie::accum<accfloat, 16> e3 = getExpBf16(b.extract<16>(3));
    return aie::concat(e0.template to_vector<bf16>(), e1.template to_vector<bf16>(), e2.template to_vector<bf16>(), e3.template to_vector<bf16>());
}

/// exp(x) of 8 lanes (the row corrections), float out
static inline aie::vector<float, 8> exp8(const aie::vector<float, 8>& x) {
    aie::vector<float, 16> d = aie::concat(x, aie::zeros<float, 8>());
    d = aie::max(aie::min(d, 88.0f), -87.0f);
    aie::accum<accfloat, 16> a;
    a.from_vector(d);
    aie::accum<accfloat, 16> e = getExpBf16(a.template to_vector<bf16>());
    return e.template to_vector<float>().extract<8>(0);
}

/// one row set (8 rows) with NB score blocks [8 rows][8 keys]: masked online softmax.  m: running max of the 8 rows (updated);
/// P: exp(S - m_new) (masked lanes 0); returns c = exp(m_old - m_new).
template <int NB>
static inline aie::vector<float, 8> row_softmax(aie::vector<float, 64>* S, float* m, aie::vector<bf16, 64>* P, const aie::mask<64>* msk, bool masked) {
    const aie::vector<float, 64> neg = aie::broadcast<float, 64>(NEG);
    if (masked) {
        for (int b = 0; b < NB; b++) S[b] = aie::select(neg, S[b], msk[b]);
    }
    aie::vector<float, 8> rm = rowmax8(S[0]);
    if constexpr (NB == 2) rm = aie::max(rm, rowmax8(S[1]));
    // Lazy running max: the max a row normalises with only moves when a new score exceeds it by more than LAZY (P values then stay below
    // e^LAZY, harmless in bf16/f32), so the correction c is exactly 1 for nearly every sub-chunk and the PV tiles skip the y rescale (an
    // fp32 x fp32 multiply is emulated with bf16 limbs on AIE2P: it was the most expensive op of the PV loop).
    constexpr float LAZY = 8.0f;
    aie::vector<float, 8> mo = aie::load_v<8>(m);
    const aie::mask<8> up = aie::gt(rm, aie::add(mo, LAZY));
    aie::vector<float, 8> mn = aie::select(mo, rm, up);
    aie::store_v(m, mn);
    aie::vector<float, 64> M = rows64(mn);
    for (int b = 0; b < NB; b++) {
        aie::vector<bf16, 64> e = exp64(aie::sub(S[b], M));
        P[b] = masked ? aie::select(aie::zeros<bf16, 64>(), e, msk[b]) : e;
    }
    if (up.empty()) return aie::broadcast<float, 8>(1.0f);
    return aie::select(aie::broadcast<float, 8>(1.0f), exp8(aie::sub(mo, mn)), up);
}

/// 64-lane int vector, lane r*8 + j = v[r]
static inline aie::vector<int32, 64> irows64(const int* v) {
    return aie::concat(aie::broadcast<int32, 8>(v[0]), aie::broadcast<int32, 8>(v[1]), aie::broadcast<int32, 8>(v[2]), aie::broadcast<int32, 8>(v[3]),
                       aie::broadcast<int32, 8>(v[4]), aie::broadcast<int32, 8>(v[5]), aie::broadcast<int32, 8>(v[6]), aie::broadcast<int32, 8>(v[7]));
}

alignas(32) static const int g_iota8[8] = {0, 1, 2, 3, 4, 5, 6, 7};

/// valid lanes of a score block [8 rows][8 keys]: key of lane (r, j) = kbase + j, valid iff lo[r] <= key <= hi[r]
static inline aie::mask<64> key_mask(int kbase, const int* lo, const int* hi) {
    const aie::vector<int32, 8> j8 = aie::load_v<8>(g_iota8);
    aie::vector<int32, 64> k = aie::concat(j8, j8, j8, j8, j8, j8, j8, j8);
    k = aie::add(k, kbase);
    return aie::ge(k, irows64(lo)) & aie::le(k, irows64(hi));
}

// per-dispatch row tables: [0] global rows r8 -> tl = r8 % 4; [1 + e] SWA row set e: rows i -> tl = 2 (i % 2) + e
alignas(32) static int g_lo[3][8] AIE_PERSIST_ZERO;
alignas(32) static int g_hi[3][8] AIE_PERSIST_ZERO;
alignas(64) static float g_m[2][8] AIE_PERSIST_ZERO;

extern "C" {

void attnx_score(bf16* __restrict q, bf16* __restrict k, bf16* __restrict pa, bf16* __restrict pb, int* rtp) {
    AIE_PERSIST_ZERO static int k_ping = 0;
    AIE_PERSIST_ZERO static int pa_ping = 0;
    AIE_PERSIST_ZERO static int pb_ping = 0;

    _lock_acquire(Q_CONS);
    const int n_sub = rtp[0];      // K sub-chunks per head
    const int hi0 = rtp[1];        // token 0's last key (stream row index)
    const int u0 = rtp[2];         // token 0's first key = max(0, u0) (SWA window), very negative for global
    const int is_swa = rtp[3];
    const int tb = rtp[4];         // first token of this tile (4 s)

    for (int r = 0; r < 8; r++) {
        for (int st = 0; st < 3; st++) {
            const int tl = (st == 0) ? (r & 3) : (2 * (r & 1) + (st - 1));
            const int t = tb + tl;
            const int lo = u0 + t;
            g_hi[st][r] = hi0 + t;
            g_lo[st][r] = lo > 0 ? lo : 0;
        }
    }
    const int hi_min = hi0 + tb;
    const int lo_max = (u0 + tb + 3) > 0 ? (u0 + tb + 3) : 0;
    const int kps = is_swa ? 16 : 8;

    for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 16; i++) g_m[i >> 3][i & 7] = NEG;
        for (AIE_LOCK_LOOP_CTR sub = 0; sub < n_sub; sub++) {
            _lock_acquire(K_CONS);
            const bf16* kb = k + k_ping * SUB;
            k_ping ^= 1;
            const int kfirst = sub * kps, klast = kfirst + kps - 1;
            const bool masked = !(klast <= hi_min && kfirst >= lo_max);

            aie::vector<bf16, 64> Pa0, Pa1, Pb0, Pb1;
            aie::vector<float, 8> ca, cb;
            if (!is_swa) {
                // rows of head group h: Q blocks rb = 2h, 2h+1
                const bf16* q0 = q + (2 * h) * QBD;
                const bf16* q1 = q + (2 * h + 1) * QBD;
                MMQ C0, C1;
                {
                    aie::block_vector<bfp16ebs8, 64> B = to_bfp(gather8(kb, DHG));
                    C0.mul(aie::load_v<64>(q0), aie::op_transpose(B));
                    C1.mul(aie::load_v<64>(q1), aie::op_transpose(B));
                }
                for (int kd = 1; kd < DHG / 8; kd++) {
                    aie::block_vector<bfp16ebs8, 64> B = to_bfp(gather8(kb + kd * 8, DHG));
                    C0.mac(aie::load_v<64>(q0 + kd * 64), aie::op_transpose(B));
                    C1.mac(aie::load_v<64>(q1 + kd * 64), aie::op_transpose(B));
                }
                aie::vector<float, 64> S0[1] = {C0.template to_vector<float>()};
                aie::vector<float, 64> S1[1] = {C1.template to_vector<float>()};
                aie::mask<64> msk[1];
                if (masked) msk[0] = key_mask(kfirst, g_lo[0], g_hi[0]);
                aie::vector<bf16, 64> P0[1], P1[1];
                aie::vector<float, 8> c0 = row_softmax<1>(S0, g_m[0], P0, msk, masked);
                aie::vector<float, 8> c1 = row_softmax<1>(S1, g_m[1], P1, msk, masked);
                // PV a: tokens tl 0, 1 = rows {0,1,4,5} of each block; PV b: rows {2,3,6,7}.  Result rows (hql, tl2).
                Pa0 = aie::concat(aie::filter_even(P0[0], 16), aie::filter_even(P1[0], 16));
                Pb0 = aie::concat(aie::filter_odd(P0[0], 16), aie::filter_odd(P1[0], 16));
                Pa1 = aie::zeros<bf16, 64>();
                Pb1 = aie::zeros<bf16, 64>();
                ca = aie::concat(aie::filter_even(c0, 2), aie::filter_even(c1, 2));
                cb = aie::concat(aie::filter_odd(c0, 2), aie::filter_odd(c1, 2));
            } else {
                // head group h = Q BD h; A blocks (e, kd) at h*QBD + (e*32 + kd)*64; K blocks (kb, kd): keys kb*8..+8 of the linear [16][256]
                const bf16* qh = q + h * QBD;
                MMQ C00, C01, C10, C11;   // C[e][kb]
                {
                    aie::block_vector<bfp16ebs8, 64> B0 = to_bfp(gather8(kb, DHS));
                    aie::block_vector<bfp16ebs8, 64> B1 = to_bfp(gather8(kb + 8 * DHS, DHS));
                    aie::vector<bf16, 64> A0 = aie::load_v<64>(qh);
                    aie::vector<bf16, 64> A1 = aie::load_v<64>(qh + 32 * 64);
                    C00.mul(A0, aie::op_transpose(B0)); C01.mul(A0, aie::op_transpose(B1)); C10.mul(A1, aie::op_transpose(B0)); C11.mul(A1, aie::op_transpose(B1));
                }
                for (int kd = 1; kd < DHS / 8; kd++) {
                    aie::block_vector<bfp16ebs8, 64> B0 = to_bfp(gather8(kb + kd * 8, DHS));
                    aie::block_vector<bfp16ebs8, 64> B1 = to_bfp(gather8(kb + 8 * DHS + kd * 8, DHS));
                    aie::vector<bf16, 64> A0 = aie::load_v<64>(qh + kd * 64);
                    aie::vector<bf16, 64> A1 = aie::load_v<64>(qh + (32 + kd) * 64);
                    C00.mac(A0, aie::op_transpose(B0)); C01.mac(A0, aie::op_transpose(B1)); C10.mac(A1, aie::op_transpose(B0)); C11.mac(A1, aie::op_transpose(B1));
                }
                aie::vector<float, 64> S0[2] = {C00.template to_vector<float>(), C01.template to_vector<float>()};
                aie::vector<float, 64> S1[2] = {C10.template to_vector<float>(), C11.template to_vector<float>()};
                aie::mask<64> m0[2], m1[2];
                if (masked) {
                    m0[0] = key_mask(kfirst, g_lo[1], g_hi[1]);
                    m0[1] = key_mask(kfirst + 8, g_lo[1], g_hi[1]);
                    m1[0] = key_mask(kfirst, g_lo[2], g_hi[2]);
                    m1[1] = key_mask(kfirst + 8, g_lo[2], g_hi[2]);
                }
                aie::vector<bf16, 64> P0[2], P1[2];
                aie::vector<float, 8> c0 = row_softmax<2>(S0, g_m[0], P0, m0, masked);
                aie::vector<float, 8> c1 = row_softmax<2>(S1, g_m[1], P1, m1, masked);
                // rows i: tl = 2 (i%2) + e.  PV a (tl 0,1): i even, rows (hql, e); PV b (tl 2,3): i odd.  [1] = key block kb 1 (keys 8..15)
                auto za0 = aie::interleave_zip(aie::filter_even(P0[0], 8), aie::filter_even(P1[0], 8), 8);
                auto za1 = aie::interleave_zip(aie::filter_even(P0[1], 8), aie::filter_even(P1[1], 8), 8);
                auto zb0 = aie::interleave_zip(aie::filter_odd(P0[0], 8), aie::filter_odd(P1[0], 8), 8);
                auto zb1 = aie::interleave_zip(aie::filter_odd(P0[1], 8), aie::filter_odd(P1[1], 8), 8);
                Pa0 = aie::concat(za0.first, za0.second);
                Pa1 = aie::concat(za1.first, za1.second);
                Pb0 = aie::concat(zb0.first, zb0.second);
                Pb1 = aie::concat(zb1.first, zb1.second);
                auto zca = aie::interleave_zip(aie::filter_even(c0, 1), aie::filter_even(c1, 1), 1);
                auto zcb = aie::interleave_zip(aie::filter_odd(c0, 1), aie::filter_odd(c1, 1), 1);
                ca = aie::concat(zca.first, zca.second);
                cb = aie::concat(zcb.first, zcb.second);
            }
            _lock_release(K_PROD);

            // global: P_a -> PV tile 2s (channel A), P_b -> PV tile 2s+1 (channel B); SWA: both halves to PV tile 2s (channel A only)
            _lock_acquire(PA_PROD);
            pmsg* ma = (pmsg*)(pa + pa_ping * MSG);
            pa_ping ^= 1;
            aie::store_v(ma->p[0], Pa0);
            aie::store_v(ma->p[1], Pa1);
            aie::store_v(ma->c, ca);
            ma->pad[0] = (aie::reduce_min(ca) < 1.0f) ? 1.0f : 0.0f;    // rescale flag (some row's max moved)
            if (is_swa) {
                aie::store_v(ma->p[2], Pb0);
                aie::store_v(ma->p[3], Pb1);
                aie::store_v(ma->c + 8, cb);
                ma->pad[0] = (aie::reduce_min(aie::min(ca, cb)) < 1.0f) ? 1.0f : 0.0f;
            }
            _lock_release(PA_CONS);
            if (!is_swa) {
                _lock_acquire(PB_PROD);
                pmsg* mb = (pmsg*)(pb + pb_ping * MSG);
                pb_ping ^= 1;
                aie::store_v(mb->p[0], Pb0);
                aie::store_v(mb->c, cb);
                mb->pad[0] = (aie::reduce_min(cb) < 1.0f) ? 1.0f : 0.0f;
                _lock_release(PB_CONS);
            }
        }
    }
    _lock_release(Q_PROD);
}

}
