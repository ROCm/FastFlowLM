// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file seq_gen.hpp
/// \brief Runtime sequence for the Qwen3.8-27B gated-delta-net conv1d prefill.
///
/// How the 8x4 array is cut up, since every offset below follows from it:
///
///   * The conv is depthwise, so channel c never meets any channel but itself.
///     That makes *channels* the free axis to shard on, and tokens the axis
///     that must stay in order on one core (the 4-tap window slides over them).
///   * A column owns COL_CHANNELS = 4 * d contiguous channels of one token.
///     The memtile takes that 4d-wide row into one buffer and hands slice r to
///     row r, so a core owns d channels and sees *every* token in order.
///     (This is why the core indexes w as `w + row*d + k*4d`: its tap rows are
///     4d wide, its own slice is d of them.)
///   * 8 columns x 4d = 32d = PASS_CHANNELS channels move per pass.  With
///     d = 256 that is 8192, and CONV_DIM = 10240 needs two passes: a full one
///     over channels 0..8191 and a 2-column one over 8192..10239.
///
/// Channel base for (pass p, column c) is therefore p*PASS_CHANNELS +
/// c*COL_CHANNELS, and q | k | v land on column boundaries because
/// COL_CHANNELS = 1024 divides DQ = DK = 2048.  That is what lets each core be
/// told "you are q" / "you are k" / "you are v" once per pass instead of
/// branching per channel.
#pragma once
#include <cassert>
#include "npu_utils/npu_utils.hpp"

namespace conv_seq {

/// Channels one core owns; must match `d` in iron/conv_aie.py.
constexpr uint32_t CORE_CHANNELS = 256;
/// Channels one column owns: its 4 rows.
constexpr uint32_t COL_CHANNELS  = 4 * CORE_CHANNELS;   // 1024
constexpr uint32_t NUM_COLS      = 8;
/// Channels moved per pass over the whole array.
constexpr uint32_t PASS_CHANNELS = NUM_COLS * COL_CHANNELS;  // 8192

/// L_buffer[1] values, mirroring kernel/conv.cc.
constexpr uint32_t NORM_NONE = 0;
constexpr uint32_t NORM_K    = 1;
constexpr uint32_t NORM_Q    = 2;

/// Tokens per iteration step of the x stream.  The BD's iteration stride is a
/// 20-bit field in *32-bit words*, so it must hold BLK*D/2; at D = 10240 that
/// caps BLK at 204, and 128 is the convenient power of two under it.
constexpr uint32_t X_BLK_TOKENS = 128;
/// Output tokens per S2MM descriptor.  The wrap field is 10 bits, so a single
/// descriptor cannot walk more than 1024 rows.
constexpr uint32_t Y_CHUNK_TOKENS = 1024;

} // namespace conv_seq

/// \param seq         sequence being built
/// \param L_in        input rows = L_OUT + CONV_K - 1 (the 3 history rows)
/// \param D           DDR row stride in elements == CONV_DIM
/// \param L_OUT       output tokens
/// \param DQ          width of q, in channels (k is the same width)
/// \param l_address   byte address of L_buffer on a core, from the MLIR
/// \param weight_offset element offset of the conv weights inside arg 2
inline void gen_seq(
    npu_sequence* seq,
    const uint32_t L_in,
    const uint32_t D,
    const uint32_t L_OUT,
    const uint32_t DQ,
    const uint32_t l_address,
    const uint32_t weight_offset = 0
){
    using namespace conv_seq;

    auto round_up_to_multiple = [](uint32_t x, uint32_t multiple) -> uint32_t {
        return multiple == 0 ? x : ((x + multiple - 1) / multiple) * multiple;
    };

    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    // L_buffer[0] = L_OUT, L_buffer[1] = norm mode.  i32 buffer, so +4 bytes.
    const uint32_t l_out_address    = l_address;
    const uint32_t norm_mode_address = l_address + 4;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id  = 6;

    const uint32_t num_passes = round_up_to_multiple(D, PASS_CHANNELS) / PASS_CHANNELS;

    for (uint32_t pass = 0; pass < num_passes; pass++){
        const uint32_t pass_base = pass * PASS_CHANNELS;
        // Columns with real channels left in this pass.  The tail pass drives
        // fewer; the cores of the idle columns simply never get their RTP
        // sync lock released and stay parked.
        const uint32_t active_cols =
            std::min<uint32_t>(NUM_COLS, (D - pass_base + COL_CHANNELS - 1) / COL_CHANNELS);

        // ---- RTPs: tell each core its length and which norm it owes -------
        for (uint32_t col = 0; col < active_cols; col++){
            const uint32_t col_base = pass_base + col * COL_CHANNELS;
            // A column never straddles the q/k or k/v split, because
            // COL_CHANNELS divides DQ.
            uint32_t norm_mode = NORM_NONE;
            if (col_base < DQ)          norm_mode = NORM_Q;
            else if (col_base < 2 * DQ) norm_mode = NORM_K;

            for (int row = 2; row < 6; row++){
                npu_tiles tile = get_tile(row, col);
                seq->rtp_write(tile, l_out_address, L_OUT);
                seq->rtp_write(tile, norm_mode_address, norm_mode);
                // Release last: the core blocks on this lock, so the two
                // values above are guaranteed visible before it reads them.
                seq->rtp_write(tile, CT_lock_address_base + 16 * CT_rtp_sync_lock_id, 1);
            }
        }

        // ---- w: 4 taps x COL_CHANNELS, tap-major with row stride D --------
        for (uint32_t col = 0; col < active_cols; col++){
            const uint32_t w_offset = pass_base + col * COL_CHANNELS + weight_offset;
            seq->npu_dma_memcpy_nd(
                2, 2,
                MM2S, IT[col],
                (npu_bd_id)(1), it_channel_1,
                {0, 0, 0, w_offset},
                {1, 1, 4, COL_CHANNELS},
                {0, 0, D, 1},
                -1, 0, false
            );
        }

        // ---- x: L_in tokens x COL_CHANNELS --------------------------------
        const uint32_t num_blocks = L_in / X_BLK_TOKENS;
        const uint32_t remainder  = L_in % X_BLK_TOKENS;
        // iter_size is a 6-bit field in the BD.
        assert(num_blocks < 64 && "L_in too large for one x descriptor");

        for (uint32_t col = 0; col < active_cols; col++){
            const uint32_t col_base = pass_base + col * COL_CHANNELS;
            if (num_blocks > 0){
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(2), it_channel_0,
                    {0, 0, 0, col_base},
                    {1, num_blocks, X_BLK_TOKENS, COL_CHANNELS},
                    {0, X_BLK_TOKENS * D, D, 1},
                    -1, 0, false
                );
            }
            if (remainder > 0){
                const uint32_t tail_offset = col_base + num_blocks * X_BLK_TOKENS * D;
                seq->npu_dma_memcpy_nd(
                    2, 1,
                    MM2S, IT[col],
                    (npu_bd_id)(10), it_channel_0,
                    {0, 0, 0, tail_offset},
                    {1, 1, remainder, COL_CHANNELS},
                    {0, 0, D, 1},
                    -1, 0, false
                );
            }
        }

        // ---- o: L_OUT tokens x COL_CHANNELS, in <=1024-token chunks -------
        const uint32_t receive_round =
            round_up_to_multiple(L_OUT, Y_CHUNK_TOKENS) / Y_CHUNK_TOKENS;
        for (uint32_t round = 0; round < receive_round; round++){
            const int bd_offset = (round % 2) * 8;
            const uint32_t rows_this_round =
                std::min<uint32_t>(Y_CHUNK_TOKENS, L_OUT - round * Y_CHUNK_TOKENS);
            for (uint32_t col = 0; col < active_cols; col++){
                const uint32_t o_offset =
                    pass_base + col * COL_CHANNELS + round * Y_CHUNK_TOKENS * D;
                seq->npu_dma_memcpy_nd(
                    2, 0,
                    S2MM, IT[col],
                    (npu_bd_id)(bd_offset + 0), it_channel_0,
                    {0, 0, 0, o_offset},
                    {1, 1, rows_this_round, COL_CHANNELS},
                    {0, 0, D, 1},
                    -1, 0, true
                );
            }
            // Two BD ids alternate, so round N+2 may not be issued until
            // round N has drained.
            if (round > 0){
                for (uint32_t col = 0; col < active_cols; col++){
                    seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
                }
            }
        }
        for (uint32_t col = 0; col < active_cols; col++){
            seq->npu_dma_wait(IT[col], S2MM, it_channel_0);
        }
    } // pass

    seq->cmds2seq();
}
