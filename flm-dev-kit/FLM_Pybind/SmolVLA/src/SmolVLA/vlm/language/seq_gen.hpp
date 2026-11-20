// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#ifndef __SMOLVLA__LANGUAGE__SEQ__GEN__HPP__
#define __SMOLVLA__LANGUAGE__SEQ__GEN__HPP__
#include "npu_utils/npu_utils.hpp"
#include "weight_desc.hpp"

namespace smolvla_vlm_text_seq{

// ---------------------------------------------------------------------------
// Redundant-RTP elimination
//
// RTPs are written into the compute tiles' data memory and stay there until
// something overwrites them, while the sequences built here are generated once
// and then re-executed unchanged on every layer. So a parameter only has to be
// re-sent when it differs from whatever the *previously executing* sequence
// left in the tile; re-sending an unchanged value just costs write commands in
// every launch.
//
// The trackers below are the value each generator last emitted. This is only
// valid because generation order matches steady-state execution order and that
// order is a fixed cycle:
//
//   q_proj, k_proj, v_proj, [mha], o_proj, gate_proj, up_proj, down_proj  -> loop
//
// which is exactly the order the attn-block then mlp-block constructors
// generate in. The cycle also has to close: the value left by down_proj (the
// last sequence in the cycle) must match what q_proj (the first) assumes, which
// is why q_proj's writes are always emitted on the first generation pass.
//
// The mm and attn generators do not share RTP state: they target disjoint
// tile columns (mm uses cols 2-5, attn uses cols 0,1,6,7) at disjoint
// addresses, so mha running between v_proj and o_proj cannot clobber the mm
// parameters.
//
// Caveat: the trackers are function-local statics, so they describe one NPU's
// tile state. A second concurrently-generated model instance would interleave
// into them and skip writes it actually needs -- pass force_rtp=true to opt out
// of the tracking if that ever happens.
// ---------------------------------------------------------------------------

/// @param force_rtp emit every RTP write even if the tracked value is unchanged
inline void gen_text_attn_seq(
    npu_sequence* seq,
    const int DH,
    const int Heads,
    const int kv_heads,
    const int number_of_blocks,
    const int text_chunk_id,
    const int effective_number,
    bool force_rtp = false
){
    npu_tiles IT[4] = {IT0, IT1, IT7, IT6};
    seq->clear_cmds();

    // rtp_buffer is int[16]: consecutive slots are 4 bytes apart.
    //   [0]=blocks  [1]=text_chunk_id  [2]=effective_number  [3]=is_self
    const int rtp_address_1 = 13568;
    const int rtp_address_0 = 57472;

    // last value this generator emitted for each slot; -1 forces the first pass
    // to write everything.
    static int last_number_of_blocks = -1;
    static int last_text_chunk_id    = -1;
    static int last_effective_number = -1;

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

    for (int q_round = 0; q_round < number_of_blocks; q_round++){
        for (int round = 0; round < number_of_blocks; round++){
            int bd_offset = (round % 2) * 8;
            for (int col = 0; col < 4; col++){
                if (round == 0 && col < 3){
                    // send q
                    size_t q_offset = q_round * 64 * DH * Heads + col * 4 * DH;
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[col],
                        (npu_bd_id)(1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_offset},
                        {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // receive y
                    size_t y_offset = q_round * 64 * DH * Heads + col * 4 * DH;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[col],
                        (npu_bd_id)(0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)64, (uint32_t)(DH * 4)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                else if(round == 0 && col == 3){
                    // send q
                    seq->npu_dma_memcpy_nd(
                        2, 1,
                        MM2S, IT[col],
                        (npu_bd_id)(1), it_channel_0,
                        {0, 0, 0, (uint32_t)q_round * 64 * DH * Heads + 12 * DH},
                        {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, false
                    );
                    // receive y
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[col],
                        (npu_bd_id)(0), it_channel_0,
                        {0, 0, 0, (uint32_t)q_round * 64 * DH * Heads + 12 * DH},
                        {1, 1, (uint32_t)64, (uint32_t)(DH * 3)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }

                if (col < 3){
                    size_t kv_offset = (size_t)round * 64 * DH * kv_heads + (size_t)col * DH;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[col],
                        (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset},
                        {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                        {0, 0, (uint32_t)DH * kv_heads, 1},
                        -1, 0, true
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 3,
                        MM2S, IT[col],
                        (npu_bd_id)(bd_offset + 3), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset},
                        {1, 1, (uint32_t)64, (uint32_t)DH * 2},
                        {0, 0, (uint32_t)DH * kv_heads, 1},
                        -1, 0, true
                    );
                }
                else if (col == 3){
                    size_t kv_offset = (size_t)round * 64 * DH * kv_heads + 4 * DH;
                    seq->npu_dma_memcpy_nd(
                        2, 2,
                        MM2S, IT[col],
                        (npu_bd_id)(bd_offset + 2), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset},
                        {1, 1, (uint32_t)64, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * kv_heads, 1},
                        -1, 0, true
                    );
                    seq->npu_dma_memcpy_nd(
                        2, 3,
                        MM2S, IT[col],
                        (npu_bd_id)(bd_offset + 3), it_channel_1,
                        {0, 0, 0, (uint32_t)kv_offset},
                        {1, 1, (uint32_t)64, (uint32_t)DH},
                        {0, 0, (uint32_t)DH * kv_heads, 1},
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
    }
    seq->cmds2seq();
}

/// @param force_rtp emit every RTP write even if the tracked value is unchanged
inline void gen_text_mm_seq(npu_sequence* seq, const uint32_t M, const weight_desc_t& weight, int32_t rtp_act, bool force_rtp = false){
    const uint32_t w_base = (uint32_t)weight.offset / sizeof(bf16);
    constexpr npu_tiles IT[4] = {IT2, IT3, IT4, IT5};
    constexpr int total_cols = 4;
    constexpr int total_rows = 4;
    const int32_t K = weight.shape[0];
    const int32_t N = weight.shape[1];
    constexpr int m = 64;
    constexpr int k = 320;
    constexpr int n = 80;
    assert(N % k == 0);
    seq->clear_cmds();

    constexpr int rtp_address = 49408;
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

    for (int j = 0; j < N_div_n; j++){
        for (int i = 0; i < M_div_m; i++){
            int round = j * M_div_m + i;
            int bd_offset = (round % 2) * 8;
            int y_offset = i * m * N * total_rows + j * n * total_cols;
            int x_offset = i * m * K * total_rows;
            uint32_t w_offset = j * n * K * total_cols + w_base;
            npu_bd_id y_bd = npu_bd_id(0 + bd_offset);
            npu_bd_id x_bd = npu_bd_id(1 + bd_offset);
            npu_bd_id w_bd = npu_bd_id(2 + bd_offset);
            // send w
            for (int c = 0; c < total_cols; c++){
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset + c * n * K},
                    {1, K_div_k, n, (uint32_t)k},
                    {0, k, K, 1},
                    -1, 0, false
                );
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

}
#endif