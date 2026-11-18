// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/npu/vision_attn_sequence.cpp
/// \brief The vision attention core's pass, descriptor for descriptor.
///
/// Every descriptor below is `gen_attn_seq()`'s from
/// FLM_Xclbin/Qwen3_8/vision_attn/host/seq_gen.hpp, with the loose parameters
/// replaced by reads of the arguments and the k/v head made permutable. The
/// comments on why a descriptor is shaped the way it is are kept: they are not
/// recoverable from the numbers alone.
#include "image/npu/vision_attn_sequence.hpp"

#include <stdexcept>
#include <string>

namespace mcvm_npu {

using namespace vision_attn_tiles;

void gen_vision_attn_seq(npu_sequence* seq,
                         int L_pad, int L_valid, int DH, int Heads,
                         const vision_attn_rtp_addresses& rtp,
                         const int* head_perm, int n_seg) {
    if (DH != DH_BAKED)
        throw std::runtime_error(
            "gen_vision_attn_seq: head_dim " + std::to_string(DH) +
            " but vision_attn.xclbin is compiled for " +
            std::to_string(DH_BAKED) + " (kernel/common.h's `dh`, which also "
            "fixes d_blocks and the 2-wide column loop's odd tail)");
    if (Heads <= 0 || Heads % HEADS_PER_PASS != 0)
        throw std::runtime_error(
            "gen_vision_attn_seq: " + std::to_string(Heads) +
            " heads is not a positive multiple of " +
            std::to_string(HEADS_PER_PASS) + " -- one pass covers one head per "
            "CU and there are four CUs");
    if (L_pad <= 0 || L_pad % ROWS_PER_ROUND != 0)
        throw std::runtime_error(
            "gen_vision_attn_seq: L_pad " + std::to_string(L_pad) +
            " is not a positive multiple of " + std::to_string(ROWS_PER_ROUND) +
            " -- the caller pads the image, the kernel cannot");
    if (n_seg <= 0)
        throw std::runtime_error("gen_vision_attn_seq: n_seg " + std::to_string(n_seg));
    if (L_valid <= 0 || L_valid > L_pad)
        throw std::runtime_error(
            "gen_vision_attn_seq: L_valid " + std::to_string(L_valid) +
            " is outside (0, " + std::to_string(L_pad) + "]");

    // Row stride of the q/k/v/o buffers. Every descriptor below is a strided
    // read of one head's DH-wide slice out of this, which is why the host
    // hands over head-major rows and never a per-head plane.
    const uint32_t D = (uint32_t)(DH * Heads);

    seq->clear_cmds();

    // ---- the two lengths, to all 32 cores ---------------------------------
    // attn.cc reads L_buffer[0] for its trip count and L_buffer[1] for the key
    // mask, every outer iteration. Rows 2..5 are the four CT rows; the shim is
    // row 0 and the memtile row 1, so the offset is not cosmetic.
    for (int row = 2; row < 6; row++) {
        for (int col = 0; col < 8; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp.l,     (uint32_t)L_pad);
            seq->rtp_write(tile, rtp.l + 4, (uint32_t)L_valid);
        }
    }

    const int rounds = L_pad / ROWS_PER_ROUND;

    // MiniCPM: one view per segment, each its own set of passes over its own
    // rows. `seg_base` is the segment's first element in every one of q/k/v/o.
    for (int seg = 0; seg < n_seg; seg++) {
    const uint32_t seg_base = (uint32_t)seg * (uint32_t)L_pad * D;
    for (int pass = 0; pass < Heads / HEADS_PER_PASS; pass++) {
        for (int round = 0; round < rounds; round++) {

            // Ping-pong the BD ids across rounds. A shim has 16 BDs and a round
            // uses five of them, so two banks of eight let round r+1's
            // descriptors be written while round r is still in flight
            // (memory: memtile-bd-ids-are-banked-by-parity -- a shim's budget
            // is the tighter one).
            const int bd_offset = (round % 2) * 8;

            for (int cu = 0; cu < 4; cu++) {
                const int head    = pass * HEADS_PER_PASS + cu;
                const int kv_head = head_perm ? head_perm[head] : head;

                // ---- y back, 256 rows per column of the pair --------------
                // The even column returns rows [0,256) of the round and the
                // odd one rows [256,512), which is the same split the q sends
                // below make. issue_token = true: these are what the waits at
                // the bottom of the round are waiting on.
                for (int col = 0; col < 2; col++) {
                    const uint32_t y_offset = seg_base +
                        (uint32_t)head * (uint32_t)DH +
                        (uint32_t)round * (uint32_t)ROWS_PER_ROUND * D +
                        (uint32_t)col * (uint32_t)(LQ * 4) * D;
                    seq->npu_dma_memcpy_nd(
                        2, o_arg_id,
                        S2MM, compute_tiles[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, y_offset},
                        {1, 1, (uint32_t)(4 * LQ), (uint32_t)DH},
                        {0, 0, D, 1},
                        -1, 0, true
                    );
                }

                // ---- q in, both channels of the EVEN shim ----------------
                // Both halves enter on compute_tiles[cu][0]: the memtile
                // broadcasts each half to all four CTs of both columns, and
                // attn.cc's `col` argument is what makes a core keep its own
                // half and drop the other. So a send here is one flow and two
                // consumers, not two flows.
                const uint32_t q_offset = seg_base +
                    (uint32_t)head * (uint32_t)DH +
                    (uint32_t)round * (uint32_t)ROWS_PER_ROUND * D;
                seq->npu_dma_memcpy_nd(
                    2, q_arg_id,
                    MM2S, compute_tiles[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, q_offset},
                    {1, 1, (uint32_t)(LQ * 4), (uint32_t)DH},
                    {0, 0, D, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, q_arg_id,
                    MM2S, compute_tiles[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0, q_offset + (uint32_t)(LQ * 4) * D},
                    {1, 1, (uint32_t)(LQ * 4), (uint32_t)DH},
                    {0, 0, D, 1},
                    -1, 0, false
                );

                // ---- k and v in, the whole padded sequence, every round ----
                // Both on the ODD shim of the pair, k on channel 0 and v on
                // channel 1. The design holds no k/v across rounds, so each
                // round re-reads all L_pad rows: the DDR traffic is
                // rounds * 2 * L_pad * DH * 2 bytes per CU, and that -- not the
                // MACs -- is why a 256-row image padded to 512 costs the same
                // as a real 512-row one.
                //
                // The chunking lives in dims 1..2 (`L_pad / KV_CHUNK` blocks of
                // KV_CHUNK rows), never in the iteration dim, because an
                // iteration count is not part of the transfer length and
                // splitting an axis across it truncates silently
                // (memory: shim-bd-iter-not-in-length).
                const uint32_t kv_offset = seg_base + (uint32_t)kv_head * (uint32_t)DH;
                seq->npu_dma_memcpy_nd(
                    2, k_arg_id,
                    MM2S, compute_tiles[cu][1],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, kv_offset},
                    {1, (uint32_t)(L_pad / KV_CHUNK), (uint32_t)KV_CHUNK, (uint32_t)DH},
                    {0, (uint32_t)KV_CHUNK * D, D, 1},
                    -1, 0, false
                );
                seq->npu_dma_memcpy_nd(
                    2, v_arg_id,
                    MM2S, compute_tiles[cu][1],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, kv_offset},
                    {1, (uint32_t)(L_pad / KV_CHUNK), (uint32_t)KV_CHUNK, (uint32_t)DH},
                    {0, (uint32_t)KV_CHUNK * D, D, 1},
                    -1, 0, false
                );
            } // cu

            // Round r's descriptors are queued before round r-1's output is
            // collected -- that overlap is the point of the BD ping-pong -- so
            // the wait trails by one and the last round's is picked up by the
            // drain below.
            if (round > 0) {
                for (int cu = 0; cu < 4; cu++)
                    for (int col = 0; col < 2; col++)
                        seq->npu_dma_wait(compute_tiles[cu][col], S2MM, it_channel_0);
            }
        } // round

        // Drain the pass. A head group must be complete before the next one
        // starts writing the same BD ids.
        for (int cu = 0; cu < 4; cu++)
            for (int col = 0; col < 2; col++)
                seq->npu_dma_wait(compute_tiles[cu][col], S2MM, it_channel_0);
    } // pass
    } // seg

    seq->cmds2seq();
}

} // namespace mcvm_npu
