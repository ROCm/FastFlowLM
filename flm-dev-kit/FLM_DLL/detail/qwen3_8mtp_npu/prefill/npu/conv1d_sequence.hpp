// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/conv1d_sequence.hpp
/// \brief Runtime sequence for `conv.xclbin`, the delta-net prefill conv1d.
///
/// A port of `FLM_Xclbin/Qwen3_8/conv1d_prefill/host/seq_gen.hpp` into the
/// engine, the same relationship `prefill/npu/prefill_attn_sequence.hpp` has to
/// its own kernel project. The array geometry below is the design's, not a
/// choice made here -- if the bitstream is rebuilt with a different `d`, every
/// offset in this file moves.
///
/// How the 8x4 array is cut up, since every offset follows from it:
///
///   * The conv is depthwise, so channel c never meets any channel but itself.
///     That makes *channels* the free axis to shard on, and tokens the axis
///     that must stay in order on one core (the 4-tap window slides over them).
///   * A column owns COL_CHANNELS = 4 * d contiguous channels of one token.
///     The memtile takes that 4d-wide row into one buffer and hands slice r to
///     row r, so a core owns d channels and sees *every* token in order.
///   * 8 columns x 4d = 32d = PASS_CHANNELS channels move per pass. With
///     d = 256 that is 8192, and CONV_DIM = 10240 needs two passes: a full one
///     over channels 0..8191 and a 2-column one over 8192..10239.
///
/// Channel base for (pass p, column c) is p*PASS_CHANNELS + c*COL_CHANNELS, and
/// q | k | v land on column boundaries because COL_CHANNELS = 1024 divides
/// DQ = DK = 2048. That is what lets each core be told "you are q" / "you are k"
/// once per pass instead of branching per channel.
///
/// ## The q scale is NOT applied here, and that is deliberate
///
/// The kernel's `NORM_Q` mode folds `1/sqrt(head_dim)` into q's l2norm
/// (`conv1d_prefill/kernel/typedef.hpp:19`, LINEAR_ATTN_SCALE = 0.08838835).
/// The delta-net bitstream this feeds applies its own `inv_sqrt_DH` to q
/// (`gate_delta_net_prefill_128_128_128_32_16_128.cc:137`), and its own host
/// hands it q l2-normalised with no scale at all. Chaining the two designs with
/// `NORM_Q` therefore scales q by 1/128 instead of 1/sqrt(128).
///
/// So `q_norm_mode` defaults to NORM_K -- l2norm with scale 1.0, the same code
/// path k takes -- and the delta-net kernel remains the only place the scale is
/// applied. A cosine cannot see this error (a uniform scale on q survives the
/// gated RMSNorm downstream and mostly cancels), so the check that catches it is
/// the measured per-head L2 norm of the device's q band: 1.0, not 0.0884.
#ifndef __QWEN3_8MTP_NPU_CONV1D_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_CONV1D_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace q38npu {
namespace conv_seq {

/// Channels one core owns; must match `d` in the design's iron/conv_aie.py.
constexpr uint32_t CORE_CHANNELS = 256;
/// Channels one column owns: its 4 rows.
constexpr uint32_t COL_CHANNELS  = 4 * CORE_CHANNELS;   // 1024
constexpr uint32_t NUM_COLS      = 8;
/// Channels moved per pass over the whole array.
constexpr uint32_t PASS_CHANNELS = NUM_COLS * COL_CHANNELS;  // 8192

/// L_buffer[1] values, mirroring the design's kernel/conv.cc.
constexpr uint32_t NORM_NONE = 0;
constexpr uint32_t NORM_K    = 1;   ///< l2norm, scale 1.0
constexpr uint32_t NORM_Q    = 2;   ///< l2norm, times 1/sqrt(head_dim)

/// Tokens per iteration step of the x stream. The BD's iteration stride is a
/// 20-bit field in *32-bit words*, so it must hold BLK*D/2; at D = 10240 that
/// caps BLK at 204, and 128 is the convenient power of two under it.
constexpr uint32_t X_BLK_TOKENS = 128;
/// Output tokens per S2MM descriptor. The wrap field is 10 bits, so a single
/// descriptor cannot walk more than 1024 rows.
constexpr uint32_t Y_CHUNK_TOKENS = 1024;
/// The iteration count is a 6-bit field, so L_in / X_BLK_TOKENS must stay
/// under 64 -- 8192 input rows, 8189 output tokens.
constexpr uint32_t MAX_L_IN = 64 * X_BLK_TOKENS;

} // namespace conv_seq

/// Per-build core addresses, the way `dequant_mm_rtp_addresses` is.
///
/// Read out of the design's `input_with_addresses.mlir`, not out of anything
/// checked in here; `conv1d_prefill/host/host.cpp:112` carries the same number.
/// A stale one writes the length and the norm mode into whatever now lives at
/// that address, and the design hangs in `qds_device::wait()`.
struct conv1d_rtp_addresses {
    uint32_t l_buffer = 49664;
};

/// \param seq         sequence being built
/// \param L_in        input rows = L_OUT + CONV_K - 1 (the 3 history rows)
/// \param D           DDR row stride in elements == conv_dim
/// \param L_OUT       output tokens
/// \param DQ          width of q, in channels (k is the same width)
/// \param l_address   byte address of L_buffer on a core, per build
/// \param weight_offset element offset of the conv weights inside arg 2
/// \param q_norm_mode what the q columns are told to do -- see the file header
inline void gen_seq_conv1d(
    npu_sequence* seq,
    const uint32_t L_in,
    const uint32_t D,
    const uint32_t L_OUT,
    const uint32_t DQ,
    const uint32_t l_address,
    const uint32_t weight_offset = 0,
    const uint32_t q_norm_mode   = conv_seq::NORM_K
){
    using namespace conv_seq;

    if (L_in > MAX_L_IN)
        throw std::runtime_error(
            "gen_seq_conv1d: " + std::to_string(L_in) + " input rows, the x "
            "descriptor's 6-bit iteration field caps it at " +
            std::to_string(MAX_L_IN));
    if (DQ % COL_CHANNELS != 0)
        throw std::runtime_error(
            "gen_seq_conv1d: DQ " + std::to_string(DQ) + " is not a whole "
            "number of columns -- a column would straddle the q/k split and "
            "the per-column norm mode would be wrong for half its channels");

    auto round_up_to_multiple = [](uint32_t x, uint32_t multiple) -> uint32_t {
        return multiple == 0 ? x : ((x + multiple - 1) / multiple) * multiple;
    };

    npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    seq->clear_cmds();

    // L_buffer[0] = L_OUT, L_buffer[1] = norm mode. i32 buffer, so +4 bytes.
    const uint32_t l_out_address     = l_address;
    const uint32_t norm_mode_address = l_address + 4;
    constexpr int CT_lock_address_base = 0x000001F000;
    constexpr int CT_rtp_sync_lock_id  = 6;

    const uint32_t num_passes = round_up_to_multiple(D, PASS_CHANNELS) / PASS_CHANNELS;

    for (uint32_t pass = 0; pass < num_passes; pass++){
        const uint32_t pass_base = pass * PASS_CHANNELS;
        // Columns with real channels left in this pass. The tail pass drives
        // fewer; the cores of the idle columns simply never get their RTP sync
        // lock released and stay parked.
        const uint32_t active_cols =
            std::min<uint32_t>(NUM_COLS, (D - pass_base + COL_CHANNELS - 1) / COL_CHANNELS);

        // ---- RTPs: tell each core its length and which norm it owes -------
        for (uint32_t col = 0; col < active_cols; col++){
            const uint32_t col_base = pass_base + col * COL_CHANNELS;
            uint32_t norm_mode = NORM_NONE;
            if (col_base < DQ)          norm_mode = q_norm_mode;
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

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_CONV1D_SEQUENCE_HPP__
