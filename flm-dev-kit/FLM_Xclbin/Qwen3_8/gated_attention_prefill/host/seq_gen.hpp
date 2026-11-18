// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once
#include "npu_utils/npu_utils.hpp"

// -----------------------------------------------------------------------------
// Qwen3.8-27B prefill attention sequence.
//
// Array: 8 columns x 4 rows = 4 CUs, one CU per column pair.
//   CU0 = cols 0,1   CU1 = cols 2,3   CU2 = cols 4,5   CU3 = cols 6,7
// The 8 CTs of a CU split one q head's 128-token chunk 16 tokens apiece.
//
// k/v is broadcast from two shims (see iron/attn_aie.py):
//   IT3 -> MT3 -> columns 0..3  (CU0, CU1)
//   IT7 -> MT7 -> columns 4..7  (CU2, CU3)
// so exactly two kv heads are resident per round and each one feeds two q
// heads -- the 1:2 GQA mapping.  24 q heads / 4 kv heads (group 6) are covered
// in 6 passes:
//
//   pass p:  pair = p / 3,  pp = p % 3
//   CU c  :  kvh  = 2*pair + c/2
//            head = kvh * 6 + 2*pp + (c & 1)
//
// which enumerates heads 0..23 exactly once and always satisfies head/6 == kvh.
//
// KV cache layout -- two contiguous planes, k then v:
//
//   [0            .. MAX_L*DK)          k, token-major, kv heads side by side
//   [MAX_L*DK .. MAX_L*DK + MAX_L*DV)   v, same shape
//
// so k_at(t) = t*DK and v_at(t) = MAX_L*DK + t*DV, matching
// ../MTP_decoding, ../gated_attention_verify and the DLL's
// q38npu::attn_layout.  One cache can therefore be handed to any of them.
//
// Nothing here scales a *stride* with MAX_L.  The k->v hop is a base offset,
// which npu_dma_memcpy_nd lowers to a DDR address patch and not to one of the
// BD's 20-bit step fields, so it costs nothing to make it large -- unlike the
// append side of the decode designs, where that same hop as a stride masked
// silently past MAX_L = 2048.  The only MAX_L-dependent BD field left is the
// key-block count `kv_length / 128` on a 10-bit wrap, i.e. MAX_L <= 131072.
// -----------------------------------------------------------------------------

void gen_seq_1_4(
    npu_sequence* seq,
    const uint32_t L_begin,
    const uint32_t L_wanted,
    const int MAX_L,
    const int WINDOW_SIZE,
    const int lc,
    const int DH,
    const int Heads,                 // q heads (24)
    const int Num_of_d_in_KV_cache,  // kv heads per cache row (4)
    bool is_sliding_window = false,
    const int l_begin_mha_address = 61568,
    const int l_end_mha_address   = 11904,
    const int window_size_address = 61600
){
    npu_tiles IT[4][2] = {{IT0, IT1}, {IT2, IT3}, {IT4, IT5}, {IT6, IT7}};
    // the two kv broadcast sources, indexed by column group (0 -> cols 0..3)
    npu_tiles KV_IT[2] = {IT3, IT7};

    const int num_cu = 4;
    const int KV_Heads = Num_of_d_in_KV_cache;
    const int gqa_group = Heads / KV_Heads;          // 6

    // The layout, in one place.  Every kv descriptor below is these two plus a
    // head offset; nothing else knows how the cache is arranged.
    const size_t kv_row  = (size_t)DH * Num_of_d_in_KV_cache;   // DK == DV
    const size_t v_plane = (size_t)MAX_L * kv_row;              // v_at(0)

    assert((L_wanted) % (lc * 8) == 0);
    assert(Heads % num_cu == 0);
    assert(KV_Heads % 2 == 0);                        // two kv streams
    assert(gqa_group % 2 == 0);                       // two q heads per stream
    assert((KV_Heads / 2) * (gqa_group / 2) == Heads / num_cu);

    const int using_window_size = is_sliding_window ? WINDOW_SIZE : MAX_L;
    const int Q_chunks = L_wanted / (lc * 8);
    seq->clear_cmds();

    for (int row = 2; row < 6; row++){
        for (int col = 0; col < 8; col++){
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, l_begin_mha_address, L_begin);
            seq->rtp_write(tile, l_end_mha_address, Q_chunks);
            seq->rtp_write(tile, window_size_address, using_window_size);
        }
    }

    const int data_per_round = lc * 8;                // 128 tokens
    const int passes = Heads / num_cu;                // 6
    const int passes_per_pair = gqa_group / 2;        // 3

    for (int pass = 0; pass < passes; pass++){
        const int pair = pass / passes_per_pair;
        const int pp   = pass % passes_per_pair;

        for (int round = 0; round < Q_chunks; round++){
            int Lq_current = L_begin + round * data_per_round;
            int kv_begin = ((Lq_current - using_window_size) > 0) ? (Lq_current - using_window_size) : 0;
            int kv_end = (Lq_current + 2 * data_per_round - 1) / data_per_round * data_per_round;
            int kv_length = kv_end - kv_begin;

            int bd_offset = (round % 2) * 8;

            for (int cu = 0; cu < num_cu; cu++){
                const int kv_head_offset = 2 * pair + cu / 2;
                const int head_offset = kv_head_offset * gqa_group + 2 * pp + (cu & 1);

                for (int col = 0; col < 2; col++){
                    // receive y
                    size_t y_offset = (size_t)head_offset * DH
                                    + (size_t)(round * data_per_round + col * lc * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        2, 0,
                        S2MM, IT[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)4 * lc, (uint32_t)(DH)},
                        {0, 0, (uint32_t)DH * Heads, 1},
                        -1, 0, true
                    );
                }
                // send q
                size_t q_offset = (size_t)head_offset * DH
                                + (size_t)round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, (uint32_t)(q_offset + (size_t)lc * 4 * DH * Heads)},
                    {1, 1, (uint32_t)lc * 4, (uint32_t)(DH)},
                    {0, 0, (uint32_t)DH * Heads, 1},
                    -1, 0, false
                );
            } // cu

            // Two kv broadcasts per round, one per column group.
            for (int grp = 0; grp < 2; grp++){
                const int kv_head_offset = 2 * pair + grp;
                // 2*pair+grp with pair,grp in {0,1} never reaches KV_Heads, so
                // the cache is never chunked across kv heads here; an earlier
                // version carried a kv_chunk_offset term that was always zero.
                assert(kv_head_offset < Num_of_d_in_KV_cache);

                size_t k_offset = (size_t)kv_begin * kv_row + (size_t)kv_head_offset * DH;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, KV_IT[grp],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)(128 * kv_row), (uint32_t)kv_row, 1},
                    -1, 0, false
                );
                // The whole k plane in, not half the cache: that form assumed
                // DK == DV, which is true for this model and is not a property
                // of the layout.
                size_t v_offset = v_plane + k_offset;
                seq->npu_dma_memcpy_nd(
                    2, 2,
                    MM2S, KV_IT[grp],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)kv_length / 128, (uint32_t)128, (uint32_t)(DH)},
                    {0, (uint32_t)(128 * DH * Num_of_d_in_KV_cache), (uint32_t)(DH * Num_of_d_in_KV_cache), 1},
                    -1, 0, false
                );
            } // grp

            if (round > 0){
                for (int cu = 0; cu < num_cu; cu++){
                    for (int col = 0; col < 2; col++){
                        seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
                    }
                }
            }
        } // round

        for (int cu = 0; cu < num_cu; cu++){
            for (int col = 0; col < 2; col++){
                seq->npu_dma_wait(IT[cu][col], S2MM, it_channel_0);
            }
        }
    } // pass
    seq->cmds2seq();
}
