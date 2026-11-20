// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// attnx PV tile: y += P.V per V sub-chunk, o = y / l (layouts: kernels/attnx_common.h).
//
// global (all 4 tiles, tokens 2j, 2j+1): y f32 blocks (h, kd) [8 rows][8 dims], rows (hql, tl2) = 2 hql + tl2, 2 x 64 blocks (32 KB).
// SWA    (tiles 0, 2: tokens 4s..4s+3; tiles 1, 3 drain the V multicast only): blocks (h, rbp, kd), rows (hql, tl%2), rbp = tl / 2,
//        2 x 2 x 32 blocks (32 KB).
// At the end every block is scaled by 1/l and converted to bf16 IN PLACE (a block keeps its slot of 256 B, the bf16 data in its first 128 B);
// the o MM2S BDs then read token rows: global (tl2, h): (hql 4, stride 16), (kd 64, stride 128), (d8 8, 1), offset h*64*128 + tl2*8;
// SWA (tl, h): (hql 4, stride 16), (kd 32, stride 128), (d8 8, 1), offset (h*2 + tl/2)*32*128 + (tl%2)*8.
#include "attnx_common.h"

using namespace attnx;

constexpr int V_PROD = 0;
constexpr int V_CONS = 1;
constexpr int P_PROD = 2;
constexpr int P_CONS = 3;
constexpr int OG_CONS = 4;
constexpr int OS_CONS = 5;
constexpr int O_SENT = 6;

using MMV = aie::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8>;

static inline aie::block_vector<bfp16ebs8, 64> to_bfp(const aie::vector<bf16, 64>& v) {
    aie::accum<accfloat, 64> a(v);
    return a.template to_vector<bfp16ebs8>();
}

/// V block [8 keys][8 dims] of the linear sub-chunk (row stride `stride`) -> bfp16 of its transpose [8 dims][8 keys] (the B^T operand of P.V)
static inline aie::block_vector<bfp16ebs8, 64> vt_bfp(const bf16* p, int stride) {
    aie::vector<bf16, 64> t = aie::transpose(gather8(p, stride), 8, 8);
    aie::accum<accfloat, 64> a(t);
    return a.template to_vector<bfp16ebs8>();
}

/// row sums of a P block [8 rows][8 keys] (bf16) as float
static inline aie::vector<float, 8> prowsum(const aie::vector<bf16, 64>& p) {
    aie::accum<accfloat, 64> a;
    a.from_vector(p);
    return rowsum8(a.template to_vector<float>());
}

/// l = l * c + rowsum (c applied only when the message says some row's max moved: fp32 multiplies are emulated on AIE2P)
static inline void l_update(float* l, bool rescale, const aie::vector<float, 8>& c8, const aie::vector<float, 8>& rs) {
    aie::vector<float, 8> lv = aie::load_v<8>(l);
    if (rescale) lv = aie::mul(lv, c8).template to_vector<float>();
    aie::store_v(l, aie::add(lv, rs));
}

static inline aie::accum<accfloat, 64> y_load(const float* yb, bool rescale, const aie::vector<float, 64>& C64) {
    if (rescale) return aie::mul(aie::load_v<64>(yb), C64);
    aie::accum<accfloat, 64> acc;
    acc.from_vector(aie::load_v<64>(yb));
    return acc;
}

/// y block = y block (* c) + A0 . B0^T
static inline void y_update(float* yb, bool rescale, const aie::vector<float, 64>& C64, const aie::block_vector<bfp16ebs8, 64>& A0,
                            const aie::block_vector<bfp16ebs8, 64>& B0) {
    MMV Y(y_load(yb, rescale, C64));
    Y.mac(A0, aie::op_transpose(B0));
    aie::store_v(yb, Y.template to_vector<float>());
}

/// y block = y block (* c) + A0 . B0^T + A1 . B1^T
static inline void y_update2(float* yb, bool rescale, const aie::vector<float, 64>& C64, const aie::block_vector<bfp16ebs8, 64>& A0,
                             const aie::block_vector<bfp16ebs8, 64>& B0, const aie::block_vector<bfp16ebs8, 64>& A1,
                             const aie::block_vector<bfp16ebs8, 64>& B1) {
    MMV Y(y_load(yb, rescale, C64));
    Y.mac(A0, aie::op_transpose(B0));
    Y.mac(A1, aie::op_transpose(B1));
    aie::store_v(yb, Y.template to_vector<float>());
}

alignas(64) static float g_l[4][8] AIE_PERSIST_ZERO;   // row sets: global h -> [2h]; SWA (h, rbp) -> [2h + rbp]

extern "C" {

void attnx_pv(bf16* __restrict v, bf16* __restrict p, bf16* __restrict ybuf, int* rtp) {
    AIE_PERSIST_ZERO static int v_ping = 0;
    AIE_PERSIST_ZERO static int p_ping = 0;
    float* __restrict y = (float*)ybuf;

    _lock_acquire(V_CONS);           // the first V sub-chunk also orders the RTP read after the host's writes
    const int n_sub = rtp[0];
    const int is_swa = rtp[1];
    const int j = rtp[2];

    if (is_swa && (j & 1)) {
        // SWA layer: tiles 1 and 3 only drain the V multicast
        for (AIE_LOCK_LOOP_CTR i = 0; i < 2 * n_sub; i++) {
            if (i != 0) _lock_acquire(V_CONS);
            v_ping ^= 1;
            _lock_release(V_PROD);
        }
        return;
    }

    for (int h = 0; h < 2; h++) {
        float* yh = y + h * 64 * 64;    // 64 blocks per head in both kinds
        const aie::vector<float, 16> z16 = aie::zeros<float, 16>();
        for (int i = 0; i < 64 * 64; i += 16) aie::store_v(yh + i, z16);
        aie::store_v(g_l[2 * h], aie::zeros<float, 8>());
        aie::store_v(g_l[2 * h + 1], aie::zeros<float, 8>());

        for (AIE_LOCK_LOOP_CTR sub = 0; sub < n_sub; sub++) {
            if (h != 0 || sub != 0) _lock_acquire(V_CONS);
            const bf16* vb = v + v_ping * SUB;
            v_ping ^= 1;
            _lock_acquire(P_CONS);
            const pmsg* msg = (const pmsg*)(p + p_ping * MSG);
            p_ping ^= 1;
            if (!is_swa) {
                const aie::vector<float, 8> c8 = aie::load_v<8>(msg->c);
                const aie::vector<float, 64> C64 = rows64(c8);
                const aie::vector<bf16, 64> P0 = aie::load_v<64>(msg->p[0]);
                const bool rescale = msg->pad[0] != 0.0f;
                l_update(g_l[2 * h], rescale, c8, prowsum(P0));
                const aie::block_vector<bfp16ebs8, 64> A0 = to_bfp(P0);
                _lock_release(P_PROD);
                if (rescale) { for (int kd = 0; kd < DHG / 8; kd++) y_update(yh + kd * 64, true, C64, A0, vt_bfp(vb + kd * 8, DHG)); }
                else         { for (int kd = 0; kd < DHG / 8; kd++) y_update(yh + kd * 64, false, C64, A0, vt_bfp(vb + kd * 8, DHG)); }
            } else {
                const aie::vector<float, 8> c0 = aie::load_v<8>(msg->c);
                const aie::vector<float, 8> c1 = aie::load_v<8>(msg->c + 8);
                const aie::vector<float, 64> C0 = rows64(c0), C1 = rows64(c1);
                const aie::vector<bf16, 64> P00 = aie::load_v<64>(msg->p[0]), P01 = aie::load_v<64>(msg->p[1]);
                const aie::vector<bf16, 64> P10 = aie::load_v<64>(msg->p[2]), P11 = aie::load_v<64>(msg->p[3]);
                const bool rescale = msg->pad[0] != 0.0f;
                l_update(g_l[2 * h], rescale, c0, aie::add(prowsum(P00), prowsum(P01)));
                l_update(g_l[2 * h + 1], rescale, c1, aie::add(prowsum(P10), prowsum(P11)));
                const aie::block_vector<bfp16ebs8, 64> A00 = to_bfp(P00), A01 = to_bfp(P01), A10 = to_bfp(P10), A11 = to_bfp(P11);
                _lock_release(P_PROD);
                if (rescale) {
                    for (int kd = 0; kd < DHS / 8; kd++) {
                        const aie::block_vector<bfp16ebs8, 64> B0 = vt_bfp(vb + kd * 8, DHS);
                        const aie::block_vector<bfp16ebs8, 64> B1 = vt_bfp(vb + 8 * DHS + kd * 8, DHS);
                        y_update2(yh + kd * 64, true, C0, A00, B0, A01, B1);
                        y_update2(yh + (32 + kd) * 64, true, C1, A10, B0, A11, B1);
                    }
                } else {
                    for (int kd = 0; kd < DHS / 8; kd++) {
                        const aie::block_vector<bfp16ebs8, 64> B0 = vt_bfp(vb + kd * 8, DHS);
                        const aie::block_vector<bfp16ebs8, 64> B1 = vt_bfp(vb + 8 * DHS + kd * 8, DHS);
                        y_update2(yh + kd * 64, false, C0, A00, B0, A01, B1);
                        y_update2(yh + (32 + kd) * 64, false, C1, A10, B0, A11, B1);
                    }
                }
            }
            _lock_release(V_PROD);
        }
    }

    // o = y / l, bf16 in place (block slot keeps its 256 B, data in the first 128 B).  Row set rs covers 32 blocks: global (h = rs / 2,
    // both halves of the head's 64 blocks use g_l[2h]), SWA (h, rbp) = rs.
    for (int rs = 0; rs < 4; rs++) {
        const float* lr = g_l[is_swa ? rs : (rs & ~1)];
        alignas(32) float inv[8];
        for (int r = 0; r < 8; r++) inv[r] = 1.0f / lr[r];
        const aie::vector<float, 64> IV = rows64(aie::load_v<8>(inv));
        float* yb0 = y + rs * 32 * 64;
        for (int kd = 0; kd < 32; kd++) {
            float* yb = yb0 + kd * 64;
            aie::accum<accfloat, 64> a = aie::mul(aie::load_v<64>(yb), IV);
            aie::store_v((bf16*)yb, a.template to_vector<bf16>());
        }
    }
    _lock_release(is_swa ? OS_CONS : OG_CONS);
    _lock_acquire(O_SENT);
}

}
