// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/prefill_attn_sequence.cpp
/// \brief The prefill attention core's pass, descriptor for descriptor.
///
/// Every descriptor below is `gen_seq_1_4()`'s with the loose parameters
/// replaced by reads of `t`. The comments on why a descriptor is shaped the
/// way it is are kept: they are not recoverable from the numbers alone.
#include "prefill/npu/prefill_attn_sequence.hpp"

#include "speculate/npu/mtp_weights.hpp"    // q38npu::mtp_layout, for the
                                            // explicit instantiation at the end
#include <stdexcept>
#include <string>

namespace q38npu {

using namespace prefill_attn_tiles;

template <class Layout>
void gen_prefill_attn_seq(npu_sequence* seq, const Layout& t,
                          int L_begin, int L_wanted,
                          const prefill_attn_rtp_addresses& rtp) {
    const int DH       = t.DH;                    // 256
    const int Heads    = t.DQ / t.DH;             // 24 q heads
    const int KV_Heads = t.num_kv_heads;          // 4
    const int num_cu   = 4;
    const int gqa_group = Heads / KV_Heads;       // 6

    // The layout, in one place. Every kv descriptor below is these two plus a
    // head offset; nothing else here knows how the cache is arranged.
    //
    // The pitch by name, through plane_rows(), rather than reaching for MAX_L
    // directly: the accessor is the single place the layout is expressed (it
    // happens to be MAX_L now that the cache has no apron), and it is where
    // the v plane starts. Get it wrong and every v read lands in the k plane's
    // tail -- cosine on `o` falls but does not collapse, which is why the
    // driver scores the cache rows themselves and not only the output.
    const size_t kv_row  = (size_t)DH * KV_Heads;         // DK == DV
    const size_t v_plane = t.plane_rows() * kv_row;       // v_at(0)

    if (L_wanted <= 0 || L_wanted % Q_CHUNK != 0)
        throw std::runtime_error(
            "gen_prefill_attn_seq: L_wanted " + std::to_string(L_wanted) +
            " is not a positive multiple of " + std::to_string(Q_CHUNK) +
            " -- the caller pads the window, the kernel cannot");
    if (L_begin < 0)
        throw std::runtime_error("gen_prefill_attn_seq: L_begin is negative");
    if (Heads % num_cu || KV_Heads % 2 || gqa_group % 2 ||
        (KV_Heads / 2) * (gqa_group / 2) != Heads / num_cu)
        throw std::runtime_error(
            "gen_prefill_attn_seq: the head partition does not fit the 4-CU / "
            "2-kv-stream floorplan");
    if ((size_t)t.DK != kv_row || t.DK != t.DV)
        throw std::runtime_error(
            "gen_prefill_attn_seq: the cache row is not DH * num_kv_heads, or "
            "k and v differ in width -- both are assumed by the broadcast BDs");

    // Sliding window is off for these layers: handing the kernel the plane
    // pitch means `Lq - window <= 0` always, so kv_begin stays 0 and every
    // pass replays from the start of the cache.
    const int using_window_size = (int)t.plane_rows();

    const int Q_chunks        = L_wanted / Q_CHUNK;
    const int data_per_round  = Q_CHUNK;
    const int passes          = Heads / num_cu;       // 6
    const int passes_per_pair = gqa_group / 2;        // 3

    if (Q_chunks > 0xffff)
        throw std::runtime_error("gen_prefill_attn_seq: too many rounds");

    seq->clear_cmds();

    // All 32 cores get the same three words. The kernel derives its own token
    // positions from L_begin (attn.cc: pointer_block_q = L_begin + i*128,
    // pointer_block_inner_q = that + col*16 + row*32) and terminates on
    // Q_chunks, so these three are the whole of the per-dispatch state.
    for (int row = 2; row < 6; row++) {
        for (int col = 0; col < 8; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp.l_begin,     (uint32_t)L_begin);
            seq->rtp_write(tile, rtp.l_chunks,    (uint32_t)Q_chunks);
            seq->rtp_write(tile, rtp.window_size, (uint32_t)using_window_size);
        }
    }

    for (int pass = 0; pass < passes; pass++) {
        const int pair = pass / passes_per_pair;
        const int pp   = pass % passes_per_pair;

        for (int round = 0; round < Q_chunks; round++) {
            const int Lq_current = L_begin + round * data_per_round;
            const int kv_begin   = (Lq_current - using_window_size) > 0
                                 ? (Lq_current - using_window_size) : 0;
            // One whole block past the query chunk, then rounded up: the CTs
            // of a round cover tokens [Lq, Lq+128), so the deepest key any of
            // them needs is Lq+127, and the broadcast is in 128-row blocks.
            const int kv_end    = (Lq_current + 2 * data_per_round - 1)
                                / data_per_round * data_per_round;
            const int kv_length = kv_end - kv_begin;

            // The backstop on prefill_attn_op::run's guard. There is no apron
            // above MAX_L: if this ever fires, the device would be reading the
            // next layer's cache -- silently, a DDR read past a BO does not
            // fault. It cannot be clamped instead, because attn.cc computes
            // the same block count itself and a short BD would strand blocks.
            if ((size_t)kv_end > t.plane_rows())
                throw std::runtime_error(
                    "gen_prefill_attn_seq: the round at " +
                    std::to_string(Lq_current) + " streams keys through " +
                    std::to_string(kv_end) + ", past the plane's " +
                    std::to_string(t.plane_rows()) + " rows (MAX_L " +
                    std::to_string(t.MAX_L) + ") -- the caller must guard "
                    "L_begin + round_up(L, 128) <= MAX_L");

            // Ping/pong: a round's BDs must not overwrite the previous round's
            // while it is still in flight, and the wait below is one round
            // behind. Eight ids per bank, five used.
            const int bd_offset = (round % 2) * 8;

            for (int cu = 0; cu < num_cu; cu++) {
                const int kv_head_offset = 2 * pair + cu / 2;
                const int head_offset =
                    kv_head_offset * gqa_group + 2 * pp + (cu & 1);

                // receive y -- each of the CU's two columns returns half the
                // round's tokens, head-major inside a row of DQ.
                for (int col = 0; col < 2; col++) {
                    const size_t y_offset =
                        (size_t)head_offset * DH
                      + (size_t)(round * data_per_round
                                 + col * TOKENS_PER_CT * 4) * DH * Heads;
                    seq->npu_dma_memcpy_nd(
                        sizeof(bf16), o_arg_id,
                        S2MM, compute_tiles[cu][col],
                        (npu_bd_id)(bd_offset + 0), it_channel_0,
                        {0, 0, 0, (uint32_t)y_offset},
                        {1, 1, (uint32_t)(4 * TOKENS_PER_CT), (uint32_t)DH},
                        {0, 0, (uint32_t)(DH * Heads), 1},
                        -1, 0, true);
                }

                // send q -- both halves leave on the CU's first column, one
                // per channel. Row 0 of the BO is token L_begin, so the
                // offset carries `round` and not the absolute position.
                const size_t q_offset = (size_t)head_offset * DH
                                      + (size_t)round * data_per_round * DH * Heads;
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), q_arg_id,
                    MM2S, compute_tiles[cu][0],
                    (npu_bd_id)(bd_offset + 1), it_channel_0,
                    {0, 0, 0, (uint32_t)q_offset},
                    {1, 1, (uint32_t)(TOKENS_PER_CT * 4), (uint32_t)DH},
                    {0, 0, (uint32_t)(DH * Heads), 1},
                    -1, 0, false);
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), q_arg_id,
                    MM2S, compute_tiles[cu][0],
                    (npu_bd_id)(bd_offset + 2), it_channel_1,
                    {0, 0, 0,
                     (uint32_t)(q_offset
                                + (size_t)TOKENS_PER_CT * 4 * DH * Heads)},
                    {1, 1, (uint32_t)(TOKENS_PER_CT * 4), (uint32_t)DH},
                    {0, 0, (uint32_t)(DH * Heads), 1},
                    -1, 0, false);
            } // cu

            // Two kv broadcasts per round, one per column group.
            for (int grp = 0; grp < 2; grp++) {
                const int kv_head_offset = 2 * pair + grp;
                // 2*pair+grp with pair,grp in {0,1} never reaches KV_Heads, so
                // the cache is never chunked across kv heads here; an earlier
                // version upstream carried a kv_chunk_offset term that was
                // always zero.
                if (kv_head_offset >= KV_Heads)
                    throw std::runtime_error(
                        "gen_prefill_attn_seq: kv head out of range");

                const size_t k_offset = (size_t)kv_begin * kv_row
                                      + (size_t)kv_head_offset * DH;
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), kv_arg_id,
                    MM2S, kv_tiles[grp],
                    (npu_bd_id)(bd_offset + 3), it_channel_0,
                    {0, 0, 0, (uint32_t)k_offset},
                    {1, (uint32_t)(kv_length / Q_CHUNK), (uint32_t)Q_CHUNK,
                     (uint32_t)DH},
                    {0, (uint32_t)(Q_CHUNK * kv_row), (uint32_t)kv_row, 1},
                    -1, 0, false);
                // The whole k plane in, not half the cache: that form assumed
                // DK == DV, which is true for this model and is not a property
                // of the layout.
                const size_t v_offset = v_plane + k_offset;
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), kv_arg_id,
                    MM2S, kv_tiles[grp],
                    (npu_bd_id)(bd_offset + 4), it_channel_1,
                    {0, 0, 0, (uint32_t)v_offset},
                    {1, (uint32_t)(kv_length / Q_CHUNK), (uint32_t)Q_CHUNK,
                     (uint32_t)DH},
                    {0, (uint32_t)(Q_CHUNK * kv_row), (uint32_t)kv_row, 1},
                    -1, 0, false);
            } // grp

            // One round behind: round n's BDs reuse round n-2's ids, so it is
            // round n-1 that has to have landed before they are posted.
            if (round > 0) {
                for (int cu = 0; cu < num_cu; cu++)
                    for (int col = 0; col < 2; col++)
                        seq->npu_dma_wait(compute_tiles[cu][col], S2MM,
                                          it_channel_0);
            }
        } // round

        // Drain the pass before the next one re-posts the same ids with a
        // different head mapping.
        for (int cu = 0; cu < num_cu; cu++)
            for (int col = 0; col < 2; col++)
                seq->npu_dma_wait(compute_tiles[cu][col], S2MM, it_channel_0);
    } // pass

    seq->cmds2seq();
}

// The two cache owners. Explicit rather than header-visible because the body is
// ~200 descriptors of hardware detail that nothing else should be able to
// instantiate by accident -- a third layout has to be added here deliberately,
// and that is the moment to check that it agrees with `attn_layout` on
// every field the generator reads.
template void gen_prefill_attn_seq<attn_layout>(npu_sequence*, const attn_layout&,
                                                int, int,
                                                const prefill_attn_rtp_addresses&);
template void gen_prefill_attn_seq<mtp_layout>(npu_sequence*, const mtp_layout&,
                                               int, int,
                                               const prefill_attn_rtp_addresses&);

} // namespace q38npu
