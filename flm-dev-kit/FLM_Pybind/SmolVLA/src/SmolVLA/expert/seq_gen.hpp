// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__EXPERT__SEQ__GEN__HPP__
#define __SMOLVLA__EXPERT__SEQ__GEN__HPP__
#pragma once
#include "npu_utils/npu_utils.hpp"
#include "weight_desc.hpp"
#include "expert_mode.hpp"

namespace smolvla_expert_seq{

// ---------------------------------------------------------------------------
// Redundant-RTP elimination
//
// RTPs are written into the compute tiles' data memory and stay there until
// something overwrites them, so a parameter only has to be re-sent when it
// differs from whatever the *previously executing* sequence left in the tile.
// Re-sending an unchanged value just costs write commands in every launch.
//
// The trackers below are the value each generator last emitted. They are only
// valid because generation order matches steady-state execution order, and the
// expert's execution order is a fixed cycle over one self + one cross layer:
//
//   q_s, k_s, v_s, [mha], o_s, gate, up, down,
//   q_c, k_c, v_c, [mha], o_c, gate, up, down   -> loop
//
// Two consequences the callers have to respect:
//
//  * The mm sequences are generated ONCE (they never change), so they must be
//    generated in that cycle order -- self attn, then mlp, then cross attn.
//    That is not the block construction order, which is why the blocks expose a
//    separate gen_seqs() that SmolVLA_expert_exec calls in the right sequence.
//  * The mlp block is shared by both layer kinds, so its single set of
//    sequences has two possible predecessors: o_s and o_c. That is sound only
//    because both are [pDQ -> pD] with act=0 and therefore leave identical mm
//    RTP state. Changing either o_proj's shape breaks this.
//
// The cycle also has to close: the value left by down_proj (last in the cycle)
// must match what q_s (first) assumes, which is why q_s's writes are always
// emitted -- it is the first sequence generated, when every tracker is still
// -1. That also makes q_s self-sufficient after an xclbin switch, which wipes
// tile memory: layer 0 is always a self layer, so the first mm to execute on a
// freshly loaded expert.xclbin re-sends everything.
//
// The attn generators do NOT share RTP state with the mm generators: attn
// targets tile columns 0,1,6,7 while mm targets columns 2-5, so an mha running
// between v_proj and o_proj cannot clobber the mm parameters.
//
// The two attn sequences are also generated once each (they depend on the
// shapes only through the prefix length P, so the caller rebuilds them only
// when P changes). That leaves a two-element cycle, self then cross, and the
// two are asymmetric:
//
//   * the self sequence is built with force_rtp, so it re-sends all four slots.
//     It has to: a self layer is always layer 0, so it is the first thing to
//     run after the switch to expert.xclbin wipes the tiles.
//   * the cross sequence is built right after, so the tracker has it emitting
//     only the is_self flip -- the one slot the self sequence gets wrong for it.
//
// Every subsequent layer replays those two, which alternates is_self correctly
// and re-establishes the other three slots on each self layer.
//
// Caveat: the trackers are function-local statics, so they describe one NPU's
// tile state. A second concurrently-generated expert instance would interleave
// into them and skip writes it actually needs -- pass force_rtp=true to opt out
// of the tracking if that ever happens.
// ---------------------------------------------------------------------------

/// @param force_rtp emit every RTP write even if the tracked value is unchanged.
///                  Required on the first launch after an xclbin switch, which
///                  drops whatever the tiles were holding.
inline void gen_expert_attn_seq(
    npu_sequence* seq,
    const int DH,
    const int DQ,
    const int DK,
    const int number_of_blocks,
    const int effective_number,
    const int is_self,
    bool force_rtp = false
){
    npu_tiles IT[4] = {IT0, IT1, IT7, IT6};
    seq->clear_cmds();

    // rtp_buffer is int[16]: consecutive slots are 4 bytes apart.
    //   [0]=blocks  [1]=text_chunk_id  [2]=effective_number  [3]=is_self
    const int rtp_address_1 = 12544;
    const int rtp_address_0 = 57472;

    int text_chunk_id = number_of_blocks - 1;

    // last value this generator emitted for each slot; -1 forces the first pass
    // to write everything.
    static int last_number_of_blocks = -1;
    static int last_text_chunk_id    = -1;
    static int last_effective_number = -1;
    static int last_is_self          = -1;

    // broadcast one rtp slot to every tile the attention kernel runs on.
    auto write_slot = [&](int slot_offset, int value) {
        for (int row = 2; row < 6; row++){
            for (int col = 0; col < 2; col++){
                seq->rtp_write(get_tile(row, col), rtp_address_1 + slot_offset, value);
            }
            seq->rtp_write(get_tile(row, 7), rtp_address_1 + slot_offset, value);
        }
        for (int row = 2; row < 5; row++){
            seq->rtp_write(get_tile(row, 6), rtp_address_0 + slot_offset, value);
        }
    };

    // blocks / chunk_id / effective_number are functions of the VLM prefix
    // length, so they hold across every layer and every denoise step of a run;
    // only is_self actually alternates.
    if (force_rtp || last_number_of_blocks != number_of_blocks){
        write_slot(0, number_of_blocks);
        last_number_of_blocks = number_of_blocks;
    }
    if (force_rtp || last_text_chunk_id != text_chunk_id){
        write_slot(4, text_chunk_id);
        last_text_chunk_id = text_chunk_id;
    }
    if (force_rtp || last_effective_number != effective_number){
        write_slot(8, effective_number);
        last_effective_number = effective_number;
    }
    if (force_rtp || last_is_self != is_self){
        write_slot(12, is_self);
        last_is_self = is_self;
    }

    // send k / v, one transfer per chunk from its chunk offset in the packed buffer
    const int total_blocks = number_of_blocks + (is_self == 1 ? 1 : 0);

    for (int round = 0; round < total_blocks; round++){
        int bd_offset = (round % 2) * 8;
        for (int col = 0; col < 4; col++){
            if (round == 0 && col < 3){
                // send q
                size_t q_offset = (size_t)col * 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, false
                );
                // receive y
                size_t y_offset = (size_t)col * 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(0), it_channel_0,
                    {0, 0, 0, (uint32_t)y_offset},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, true
                );
            }
            else if(round == 0 && col == 3){
                // send q
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(1), it_channel_0,
                    {0, 0, 0, (uint32_t)12 * DH},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, false
                );
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(0), it_channel_0,
                    {0, 0, 0, (uint32_t)12 * DH},
                    {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                    {0, 0, (uint32_t)DQ, 1},
                    -1, 0, true
                );
            }

            if (col < 3){
                size_t kv_offset = (size_t)round * 64 * DK + (size_t)col * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                    {0, 0, (uint32_t)DK, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 3,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                    {0, 0, (uint32_t)DK, 1},
                    -1, 0, true
                );
            }
            else if (col == 3){
                size_t kv_offset = (size_t)round * 64 * DK + 4 * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH},
                    {0, 0, (uint32_t)DK, 1},
                    -1, 0, true
                );
                seq->npu_dma_memcpy_nd(
                    2, 3,
                    MM2S, IT[col],
                    (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset},
                    {1, 1, (uint32_t)64, (uint32_t)DH},
                    {0, 0, (uint32_t)DK, 1},
                    -1, 0, true
                );
            }
        }
        if (round > 0){
            for (int col = 0; col < 4; col++){
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
                seq->npu_dma_wait(
                    IT[col],
                    MM2S,
                    it_channel_1
                );
            } // end col
        } // end round > 0
    }
    for (int col = 0; col < 4; col++){
        seq->npu_dma_wait(
            IT[col],
            S2MM,
            it_channel_0
        );
        seq->npu_dma_wait(
            IT[col],
            MM2S,
            it_channel_1
        );
        seq->npu_dma_wait(
            IT[col],
            MM2S,
            it_channel_1
        );
    }
    seq->cmds2seq();
}


/// attention sequence for the rope-in-kernel expert.xclbin: q (and, on self layers, the expert's own K/V tail rows)
/// are read straight out of the projection output buffer QB (mha arg 1), so no host split / copy / sync sits between
/// the projection and the mha. The kernel ropes q and the tail K with the cos/sin table that follows q in QB.
///
///  * QB rows are QS elements apart; q is at column 0, tail K at column k_col, tail V at column v_col (self only)
///  * the table is 64 rows x 64 (tile layout, see attn_kernel.h rope_tile) at element offset table_off of QB
///  * self layers send the tail first (round 0), then the prefix blocks from k_cat/v_cat (args 2/3)
///  * RTP tracking is shared with gen_expert_attn_seq (same statics); see its notes
inline void gen_expert_attn_seq_rope(
    npu_sequence* seq,
    const int DH,
    const int DQ,
    const int DK,
    const int QS,
    const size_t table_off,
    const int k_col,
    const int v_col,
    const int number_of_blocks,
    const int effective_number,
    const int is_self,
    bool force_rtp = false
){
    npu_tiles IT[4] = {IT0, IT1, IT7, IT6};
    seq->clear_cmds();

    const int rtp_address_1 = 12544;
    const int rtp_address_0 = 57472;
    int text_chunk_id = number_of_blocks - 1;

    static int last_number_of_blocks = -1;
    static int last_text_chunk_id    = -1;
    static int last_effective_number = -1;
    static int last_is_self          = -1;

    auto write_slot = [&](int slot_offset, int value) {
        for (int row = 2; row < 6; row++){
            for (int col = 0; col < 2; col++){
                seq->rtp_write(get_tile(row, col), rtp_address_1 + slot_offset, value);
            }
            seq->rtp_write(get_tile(row, 7), rtp_address_1 + slot_offset, value);
        }
        for (int row = 2; row < 5; row++){
            seq->rtp_write(get_tile(row, 6), rtp_address_0 + slot_offset, value);
        }
    };
    if (force_rtp || last_number_of_blocks != number_of_blocks){
        write_slot(0, number_of_blocks);
        last_number_of_blocks = number_of_blocks;
    }
    if (force_rtp || last_text_chunk_id != text_chunk_id){
        write_slot(4, text_chunk_id);
        last_text_chunk_id = text_chunk_id;
    }
    if (force_rtp || last_effective_number != effective_number){
        write_slot(8, effective_number);
        last_effective_number = effective_number;
    }
    if (force_rtp || last_is_self != is_self){
        write_slot(12, is_self);
        last_is_self = is_self;
    }

    const int total_blocks = number_of_blocks + (is_self == 1 ? 1 : 0);
    const int tail_round = (is_self == 1) ? 0 : -1;

    for (int round = 0; round < total_blocks; round++){
        int bd_offset = (round % 2) * 8;
        const bool tail = (round == tail_round);
        const int prefix_round = round - (is_self == 1 ? 1 : 0);
        for (int col = 0; col < 4; col++){
            if (round == 0){
                // q: 4 heads (3 on the last column) of 64 rows, then the rope table
                const uint32_t q_off = (col < 3) ? (uint32_t)col * 4 * DH : (uint32_t)12 * DH;
                const uint32_t q_w   = (col < 3) ? (uint32_t)DH * 4 : (uint32_t)DH * 3;
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[col], (npu_bd_id)(1), it_channel_0,
                    {0, 0, 0, q_off}, {1, 1, (uint32_t)64, q_w}, {0, 0, (uint32_t)QS, 1},
                    -1, 0, false);
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[col], (npu_bd_id)(4), it_channel_0,
                    {0, 0, 0, (uint32_t)table_off}, {1, 1, (uint32_t)64, (uint32_t)64}, {0, 0, (uint32_t)64, 1},
                    -1, 0, false);
                // y
                seq->npu_dma_memcpy_nd(
                    2, 0, S2MM, IT[col], (npu_bd_id)(0), it_channel_0,
                    {0, 0, 0, q_off}, {1, 1, (uint32_t)64, q_w}, {0, 0, (uint32_t)DQ, 1},
                    -1, 0, true);
            }
            const uint32_t kv_col = (col < 3) ? (uint32_t)col * DH : (uint32_t)4 * DH;
            const uint32_t kv_w   = (col < 3) ? (uint32_t)DH * 2 : (uint32_t)DH;
            if (tail){
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[col], (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)k_col + kv_col}, {1, 1, (uint32_t)64, kv_w}, {0, 0, (uint32_t)QS, 1},
                    -1, 0, true);
                seq->npu_dma_memcpy_nd(
                    2, 1, MM2S, IT[col], (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)v_col + kv_col}, {1, 1, (uint32_t)64, kv_w}, {0, 0, (uint32_t)QS, 1},
                    -1, 0, true);
            } else {
                const size_t kv_offset = (size_t)prefix_round * 64 * DK + kv_col;
                seq->npu_dma_memcpy_nd(
                    2, 2, MM2S, IT[col], (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset}, {1, 1, (uint32_t)64, kv_w}, {0, 0, (uint32_t)DK, 1},
                    -1, 0, true);
                seq->npu_dma_memcpy_nd(
                    2, 3, MM2S, IT[col], (npu_bd_id)(bd_offset + 3), it_channel_1,
                    {0, 0, 0, (uint32_t)kv_offset}, {1, 1, (uint32_t)64, kv_w}, {0, 0, (uint32_t)DK, 1},
                    -1, 0, true);
            }
        }
        if (round > 0){
            for (int col = 0; col < 4; col++){
                seq->npu_dma_wait(IT[col], MM2S, it_channel_1);
                seq->npu_dma_wait(IT[col], MM2S, it_channel_1);
            }
        }
    }
    for (int col = 0; col < 4; col++){
        seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        seq->npu_dma_wait(IT[col], MM2S, it_channel_1);
        seq->npu_dma_wait(IT[col], MM2S, it_channel_1);
    }
    seq->cmds2seq();
}

/// @param force_rtp emit every RTP write even if the tracked value is unchanged
/// @param up_weight non-null: SwiGLU-fused gate+up. `weight` is the gate projection and rtp_act must be 3. Every n
///                  block streams its gate strip, then its up strip, through the same tiles; the kernel keeps
///                  silu(gate) in L1 and emits gate*up on the up pass, so the output is [M, N] hidden and nothing
///                  else leaves the array. bfp16 weights only.
inline void gen_expert_mm_seq(npu_sequence* seq, const uint32_t M, const weight_desc_t& weight, int32_t rtp_act, bool force_rtp = false,
                              const weight_desc_t* up_weight = nullptr){
    const bool bf16_w = smolvla_expert_mode::bf16_weights();
    const bool fused = up_weight != nullptr;
    assert(!fused || (!bf16_w && rtp_act == 3));
    // bytes: the weight is bfp16ebs8 packed (see SmolVLA_expert_def.hpp) unless SMOLVLA_EXPERT_BF16
    const uint32_t w_base = bf16_w ? (uint32_t)weight.offset / sizeof(bf16) : (uint32_t)weight.offset;
    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id = 6;
    const uint32_t K = weight.shape[0];
    const uint32_t N = weight.shape[1];
    constexpr int m = 16;
    constexpr int k = 256;
    constexpr int n = 64;
    assert(N % 256 == 0);
    seq->clear_cmds();

    // mm tiles; read from input_with_addresses.mlir after any L1 buffer change
    const int rtp_address = bf16_w ? 51200 : 23040;
    int K_div_k = K / k;
    const int M_div_m = M / m / total_rows;
    const int N_div_n  = N / n / total_cols;

    // last value this generator emitted for each slot; -1 forces the first pass
    // to write everything (and closes the cycle -- see the note at the top).
    static int32_t last_K_div_k = -1;
    static int32_t last_rtp_act = -1;

    // broadcast one rtp slot to every tile of the mm array.
    auto write_slot = [&](int slot_offset, int32_t value) {
        for (int row = 2; row < 6; row++){
            for (int col = 2; col < total_cols + 2; col++){
                seq->rtp_write(get_tile(row, col), rtp_address + slot_offset, value);
            }
        }
    };

    if (force_rtp || last_K_div_k != K_div_k){
        write_slot(0, K_div_k);
        last_K_div_k = K_div_k;
    }
    if (force_rtp || last_rtp_act != rtp_act){
        write_slot(4, rtp_act);
        last_rtp_act = rtp_act;
    }

    if (fused){
        // 2 rounds per (n block, m block): gate, then up. Four rotating BD sets: y is issued on the up rounds only, so
        // the y wait that normally gates BD reuse trails by two rounds, not one.
        const uint32_t up_base = (uint32_t)up_weight->offset;
        const uint32_t strip_bytes = (uint32_t)n * K / 8 * 9;
        int y_rounds = 0;
        for (int j = 0; j < N_div_n; j++){
            for (int i = 0; i < M_div_m; i++){
                int y_offset = i * m * N * total_rows + j * n * total_cols;
                int x_offset = i * m * K * total_rows;
                for (int t = 0; t < 2; t++){
                    const int round = (j * M_div_m + i) * 2 + t;
                    const int bd_base = (round % 4) * 3;
                    npu_bd_id y_bd = npu_bd_id(bd_base);
                    npu_bd_id x_bd = npu_bd_id(bd_base + 1);
                    npu_bd_id w_bd = npu_bd_id(bd_base + 2);
                    const uint32_t w_offset = (t ? up_base : w_base) + j * total_cols * strip_bytes;
                    for (int c = 0; c < total_cols; c++){
                        seq->npu_dma_memcpy_nd(
                            1, 2,
                            MM2S, IT[c],
                            w_bd, it_channel_1,
                            {0, 0, 0, w_offset + c * strip_bytes},
                            {1, 1, strip_bytes / 2304, 2304},
                            {0, 0, 2304, 1},
                            -1, 0, false
                        );
                        if (t == 1){
                            seq->npu_dma_memcpy_nd(
                                2, 0,
                                S2MM, IT[c],
                                y_bd, it_channel_0,
                                {0, 0, 0, y_offset + c * m * N},
                                {1, 1, m, n * 4},
                                {0, 0, N, 1},
                                -1, 0, true
                            );
                        }
                        seq->npu_dma_memcpy_nd(
                            2, 1,
                            MM2S, IT[c],
                            x_bd, it_channel_0,
                            {0, 0, 0, x_offset + c * m * K},
                            {1, (uint32_t)K_div_k, m, k},
                            {0, k, K, 1},
                            -1, 0, false
                        );
                    }
                    if (t == 1){
                        if (y_rounds > 0){
                            for (int r = 0; r < total_rows; r++){
                                seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                            }
                        }
                        y_rounds++;
                    }
                }
            }
        }
    } else
    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            // W: strip s = j*total_cols + c (64 n rows) is contiguous, 72*K bytes
            const uint32_t strip_bytes = (uint32_t)n * K / 8 * 9;
            uint32_t w_offset = bf16_w ? j * n * K * total_cols + w_base : j * total_cols * strip_bytes + w_base;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                if (bf16_w){
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[c],
                        w_bd, it_channel_1,
                        {0, 0, 0, w_offset + c * n * K},
                        {1, K_div_k, n, (uint32_t)k},
                        {0, k, K, 1},
                        -1, 0, false
                    );
                } else {
                    seq->npu_dma_memcpy_nd(
                        1, 2,
                        MM2S, IT[c],
                        w_bd, it_channel_1,
                        {0, 0, 0, w_offset + c * strip_bytes},
                        {1, 1, strip_bytes / 2304, 2304},
                        {0, 0, 2304, 1},
                        -1, 0, false
                    );
                }
                // receive y
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset + c * m * N},
                    {1, 1, m, n * 4},
                    {0, 0, N, 1},
                    -1, 0, true
                );
                // send x
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[c],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + c * m * K},
                    {1, (uint32_t)K_div_k, m, k},
                    {0, k, K, 1},
                    -1, 0, false
                );
            }
            if (round > 0){
                for (int r = 0; r < total_rows; r++){
                    seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
                }
            }
        }
    }
    for (int r = 0; r < total_rows; r++){
        seq->npu_dma_wait(IT[r], S2MM, it_channel_0);
    }
    seq->cmds2seq();
}
} // namespace smolvla_expert_seq
#endif

