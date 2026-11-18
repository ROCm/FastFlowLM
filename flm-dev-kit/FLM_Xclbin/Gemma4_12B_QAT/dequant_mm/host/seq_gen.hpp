// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SEQ_GEN_HPP__
#define __SEQ_GEN_HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weights_desc.hpp"

// y[L, Dout] = dequant(w) * x[L, Din].  The overlay is two independent halves of 4
// shims; one half computes one "unit" of 512 x-rows by 256 weight columns.  Within a
// half `lane` is the shim index: the core column for weights, the core row for x/y.
//
// `glu_slab` = 0 is one contiguous Dout-row matrix (q_proj, o_proj, down_proj, ...).
// `glu_slab` = S is the fused layout the decoding runtime builds, alternating S-row
// slabs up[0:S] | gate[0:S] | up[S:2S] | ... with Dout = 2 * intermediate_size.
// Weights stream in physical order either way; only the y scatter changes, so the
// fused case lands de-interleaved as [ up | gate ], each Dout/2 wide.
void gen_dequant_mm(npu_sequence* seq, const uint32_t L, const uint32_t Dout, const uint32_t Din, const size_t weight_offset, const int rtp_act, const uint32_t glu_slab = 0){
    static constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    static constexpr uint32_t desired_m = 128;  // L rows per core row
    static constexpr uint32_t desired_k = 256;  // Din per k-tile
    static constexpr uint32_t desired_n = 64;   // Dout columns per core column

    static constexpr uint32_t half_lanes = 4;   // shims per half: 4 core columns == 4 core rows
    static constexpr uint32_t unit_m = desired_m * half_lanes;  // 512 L rows per unit
    static constexpr uint32_t unit_n = desired_n * half_lanes;  // 256 Dout columns per unit

    assert(Din  % desired_k == 0);
    assert(Dout % unit_n    == 0);
    assert(L    % unit_m    == 0);
    assert(L > 0 && Dout > 0);
    if (glu_slab != 0){
        // a unit is the smallest thing the y scatter can redirect, so a seam must
        // never fall inside one, and the slabs must pair up
        assert(glu_slab % unit_n == 0);
        assert(Dout % (2 * glu_slab) == 0);
    }

    const uint32_t K_div_k = Din / desired_k;  // also the q4_0 blocks per weight row
    // K_div_k rides the BD iteration dimension, which the hardware caps at 6 bits
    assert(K_div_k < 64 && "Din too wide for the weight gather's iteration dimension");

    // one 16x256 q4_0 block: 2048 B of nibbles + 256 B of bf16 scales.  Half a
    // QXNX_ROW_BLOCK_SIZE chunk, so get_quantization_byte_size() cannot supply it.
    static constexpr uint32_t blk = 16 * 256 * 9 / 16;  // 2304 B

    const uint32_t T      = Dout / unit_n;   // Dout tiles
    const uint32_t C      = L / unit_m;      // L chunks
    const uint32_t units  = T * C;
    const uint32_t rounds = (units + 1) / 2;

    seq->clear_cmds();

    for (int row = 2; row < 6; row += 2){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 54528, K_div_k);
            seq->rtp_write(tile, 54592, rtp_act);
        }
    }
    for (int row = 3; row < 6; row += 2){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, 61696, K_div_k);
            seq->rtp_write(tile, 61760, rtp_act);
        }
    }

    // byte offset of the 64-row slab this lane owns; the slab base is the same in
    // either block ordering, only the order inside it differs (see send_w)
    auto w_byte_off = [&](uint32_t t, uint32_t lane) -> uint32_t {
        return (uint32_t)(weight_offset +
            get_quantization_byte_size((size_t)(t * unit_n + lane * desired_n) * Din, flm_q40));
    };
    auto x_elem_off = [&](uint32_t c, uint32_t lane) -> uint32_t {
        return (c * unit_m + lane * desired_m) * Din;
    };
    // physical Dout base of tile `t` -> its column base in y. Identity for a plain
    // matrix; for fused up/gate it undoes the slab interleave into [ up | gate ].
    auto y_dout_base = [&](uint32_t t) -> uint32_t {
        const uint32_t phys = t * unit_n;
        if (glu_slab == 0) {
            return phys;
        }
        const uint32_t slab = phys / glu_slab;   // even -> up, odd -> gate
        return (slab % 2) * (Dout / 2)           // which half of y
             + (slab / 2) * glu_slab             // which slab within that half
             + phys % glu_slab;                  // offset inside the slab
    };
    auto y_elem_off = [&](uint32_t c, uint32_t t, uint32_t lane) -> uint32_t {
        return (c * unit_m + lane * desired_m) * Dout + y_dout_base(t);
    };

    auto send_w = [&](uint32_t shim, npu_bd_id bd, uint32_t t, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            sizeof(char), 2, MM2S, IT[shim], bd, it_channel_1,
            {0, 0, 0, w_byte_off(t, lane)},
            {K_div_k, 2, 2, blk},
            {2 * blk, 2 * K_div_k * blk, blk, 1},
            -1, 0, false, aggressive_cache
        );
    };
    auto recv_y = [&](uint32_t shim, npu_bd_id bd, uint32_t c, uint32_t t, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            2, 0, S2MM, IT[shim], bd, it_channel_0,
            {0, 0, 0, y_elem_off(c, t, lane)},
            {1, 1, desired_m, unit_n},
            {0, 0, Dout, 1},
            -1, 0, true
        );
    };
    auto send_x = [&](uint32_t shim, npu_bd_id bd, uint32_t c, uint32_t lane){
        seq->npu_dma_memcpy_nd(
            2, 1, MM2S, IT[shim], bd, it_channel_0,
            {0, 0, 0, x_elem_off(c, lane)},
            {1, K_div_k, desired_m, desired_k},
            {0, desired_k, Din, 1},
            -1, 0, false
        );
    };

    bool prev_valid = false;   // is there a round in flight to drain?
    bool prev_right = false;   // did it program the right half?

    for (uint32_t r = 0; r < rounds; r++){
        const uint32_t u_l = 2 * r;
        const uint32_t u_r = 2 * r + 1;

        // Only the very last round can come up short, and only when T * C is odd.
        const bool right_active = (u_r < units);

        const uint32_t cl = u_l / T, tl = u_l % T;
        const uint32_t cr = right_active ? u_r / T : 0;
        const uint32_t tr = right_active ? u_r % T : 0;

        const int bd_offset = (int)(r % 2) * 8;
        npu_bd_id y_bd_0 = npu_bd_id(0 + bd_offset);
        npu_bd_id y_bd_1 = npu_bd_id(1 + bd_offset);
        npu_bd_id x_bd_0 = npu_bd_id(2 + bd_offset);
        npu_bd_id x_bd_1 = npu_bd_id(3 + bd_offset);
        npu_bd_id w_bd_0 = npu_bd_id(4 + bd_offset);
        npu_bd_id w_bd_1 = npu_bd_id(5 + bd_offset);

        for (uint32_t lane = 0; lane < half_lanes; lane++){
            send_w(lane, w_bd_0, tl, lane);
            if (right_active) send_w(lane + 4, w_bd_1, tr, lane);

            recv_y(lane, y_bd_0, cl, tl, lane);
            if (right_active) recv_y(lane + 4, y_bd_1, cr, tr, lane);

            send_x(lane, x_bd_0, cl, lane);
            if (right_active) send_x(lane + 4, x_bd_1, cr, lane);
        }

        if (prev_valid){
            for (uint32_t s = 0; s < half_lanes; s++){
                seq->npu_dma_wait(IT[s], S2MM, it_channel_0);
                if (prev_right) seq->npu_dma_wait(IT[s + 4], S2MM, it_channel_0);
            }
        }
        prev_valid = true;
        prev_right = right_active;
    }

    if (prev_valid){
        for (uint32_t s = 0; s < half_lanes; s++){
            seq->npu_dma_wait(IT[s], S2MM, it_channel_0);
            if (prev_right) seq->npu_dma_wait(IT[s + 4], S2MM, it_channel_0);
        }
    }

    seq->cmds2seq();
}

#endif
