// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// attnx: 8-token verify attention that reads every K/V row ONCE per verify (plan stream A, notes/attn_longctx.md).
//
// Work split (both layer kinds, one set of tiles; the kind is an RTP):
//   score tile s (2 tiles)  : tokens 4s..4s+3, all 8 q heads (32 q rows); per K sub-chunk of ONE kv head it computes the scores of the 16
//                             rows of that head's group, the masked online softmax and sends P + the per-row correction c to the two PV
//                             tiles of its tokens.
//   PV tile j    (4 tiles)  : global: tokens 2j, 2j+1, all 8 q heads (16 rows of 512); SWA: tiles 0 and 2 take tokens 4s..4s+3 (32 rows of
//                             256), tiles 1 and 3 only drain the V multicast.  y += P.V per V sub-chunk, o = y / l at the end.
//   o assembly (memtile)    : every PV tile sends its o rows on a circuit of its own, global on MM2S0 (2 tokens x 4096), SWA on MM2S1
//                             (4 tokens x 2048); the memtile lands them token-major and sends 8 token rows per kind: no packet merges,
//                             token order by construction.
// K and V are streamed head-major ONCE per verify: [head][rows][DH], sub-chunks of 4096 elements (global: 8 keys x 512, SWA: 16 keys x 256),
// multicast to the score tiles (K) and the PV tiles (V).  Sub-chunks land LINEARLY (row-major [keys][DH]): a tile S2MM with 16-byte runs
// ran at about a quarter of the stream rate; the kernels gather the 8x8 blocks (8 row loads of 16 B) instead.  Block (kb, kd) = keys
// kb*8..+8 (global kb = 0 only, SWA kb = 0, 1), dims kd*8..+8.
// Q (per score tile 4 x 4096 elements, the same S2MM pattern per 8 rows): global block rb (rows hq 2rb, 2rb+1 x tokens 4) at rb*4096 + kd*64,
// row r8 -> hq = 2rb + r8/4, tl = r8%4; SWA: Q BD b = head group b (rows hq 4b..4b+3 x tokens 4, 16 rows x 256 viewed as 8 x 512),
// block (e, kd) at b*4096 + (e*32 + kd)*64, row i -> hq = 4b + i/2, tl = 2(i%2) + e; Q BDs 2, 3 carry filler for SWA.
#ifndef __ATTNX_COMMON_H__
#define __ATTNX_COMMON_H__

#include "aie_kernel_utils.h"
#include "typedef.hpp"

namespace attnx {
constexpr int SUB = 4096;            // elements per K/V sub-chunk (one kv head)
constexpr int QBD = 4096;            // elements per Q BD
constexpr int NQBD = 4;              // Q BDs per score tile
constexpr int DHG = 512, DHS = 256;  // head dims
constexpr int MSG = 320;             // bf16 slots per P message (64-byte aligned)
constexpr float NEG = -0x1.FEp127f;  // running-max sentinel (as the decode kernels)

// P message layout: global (one per PV tile): p[0] = [8 rows (hql, tl2)][8 keys], c[0..7];
//                   SWA (one to PV 2s, carries both row halves): p[2 rbp + f] = [8 rows (hql, tl%2)][8 keys of parity f] for tokens
//                   tl = 2 rbp + tl%2, c[8 rbp + r]
struct pmsg {
    bf16 p[4][64];
    float c[16];
    float pad[16];
};
static_assert(sizeof(pmsg) == MSG * 2, "P message size");
}

/// 64-lane row-scale vector: lane r*8 + j = v[r]
static inline aie::vector<float, 64> rows64(const aie::vector<float, 8>& v) {
    aie::vector<float, 8> b0 = aie::broadcast<float, 8>(v[0]);
    aie::vector<float, 8> b1 = aie::broadcast<float, 8>(v[1]);
    aie::vector<float, 8> b2 = aie::broadcast<float, 8>(v[2]);
    aie::vector<float, 8> b3 = aie::broadcast<float, 8>(v[3]);
    aie::vector<float, 8> b4 = aie::broadcast<float, 8>(v[4]);
    aie::vector<float, 8> b5 = aie::broadcast<float, 8>(v[5]);
    aie::vector<float, 8> b6 = aie::broadcast<float, 8>(v[6]);
    aie::vector<float, 8> b7 = aie::broadcast<float, 8>(v[7]);
    return aie::concat(b0, b1, b2, b3, b4, b5, b6, b7);
}

/// 8x8 block of a row-major bf16 matrix: rows p, p + stride, ..., 8 elements each (16-byte loads)
static inline aie::vector<bf16, 64> gather8(const bf16* p, int stride) {
    return aie::concat(aie::concat(aie::load_v<8>(p), aie::load_v<8>(p + stride)), aie::concat(aie::load_v<8>(p + 2 * stride), aie::load_v<8>(p + 3 * stride)),
                       aie::concat(aie::load_v<8>(p + 4 * stride), aie::load_v<8>(p + 5 * stride)), aie::concat(aie::load_v<8>(p + 6 * stride), aie::load_v<8>(p + 7 * stride)));
}

/// per-row max of a [8 rows][8] float block
static inline aie::vector<float, 8> rowmax8(const aie::vector<float, 64>& x) {
    aie::vector<float, 32> m4 = aie::max(aie::filter_even(x, 4), aie::filter_odd(x, 4));
    aie::vector<float, 16> m2 = aie::max(aie::filter_even(m4, 2), aie::filter_odd(m4, 2));
    return aie::max(aie::filter_even(m2, 1), aie::filter_odd(m2, 1));
}

/// per-row sum of a [8 rows][8] float block
static inline aie::vector<float, 8> rowsum8(const aie::vector<float, 64>& x) {
    aie::vector<float, 32> s4 = aie::add(aie::filter_even(x, 4), aie::filter_odd(x, 4));
    aie::vector<float, 16> s2 = aie::add(aie::filter_even(s4, 2), aie::filter_odd(s4, 2));
    return aie::add(aie::filter_even(s2, 1), aie::filter_odd(s2, 1));
}

#endif
