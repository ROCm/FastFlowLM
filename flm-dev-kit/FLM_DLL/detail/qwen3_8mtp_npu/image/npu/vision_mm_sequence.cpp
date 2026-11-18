// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file image/npu/vision_mm_sequence.cpp
/// \brief The vision matmul array's pass, descriptor for descriptor.
///
/// Every descriptor below is `gen_mm_seq()`'s from
/// FLM_Xclbin/Qwen3_8/vision_mm/host/seq_gen.hpp, with the loose tile
/// parameters replaced by a `design_params` and the asserts replaced by throws
/// that say what the caller got wrong. The comments on why a descriptor is
/// shaped the way it is are kept: they are not recoverable from the numbers.
#include "image/npu/vision_mm_sequence.hpp"

#include <stdexcept>
#include <string>

namespace q38npu {

using namespace vision_mm_tiles;

void gen_vision_mm_seq(npu_sequence* seq,
                       uint32_t M_pad, uint32_t K, uint32_t N,
                       const design_params& dp,
                       uint32_t weight_offset, int32_t rtp_act,
                       uint32_t rtp_address) {

    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

    const uint32_t tile_M = dp.tile_M();
    const uint32_t tile_N = dp.tile_N();

    // The kernel's trip counts are M_pad/tile_M and K/k, and they are integers
    // it reads out of an RTP. A remainder is not a short last round, it is a
    // round the cores wait for and never get.
    if (M_pad == 0 || M_pad % tile_M != 0)
        throw std::runtime_error(
            "gen_vision_mm_seq: M_pad " + std::to_string(M_pad) +
            " is not a positive multiple of " + std::to_string(tile_M) +
            " (m " + std::to_string(dp.m) + " x " +
            std::to_string(TOTAL_ROWS) + " array rows) -- the caller pads M, "
            "the kernel cannot");
    if (N == 0 || N % tile_N != 0)
        throw std::runtime_error(
            "gen_vision_mm_seq: N " + std::to_string(N) +
            " is not a positive multiple of " + std::to_string(tile_N) +
            " (n " + std::to_string(dp.n) + " x " +
            std::to_string(TOTAL_COLS) + " array columns)");
    if (K == 0 || K % dp.k != 0)
        throw std::runtime_error(
            "gen_vision_mm_seq: K " + std::to_string(K) +
            " is not a positive multiple of the k tile " +
            std::to_string(dp.k));
    if (rtp_act < RTP_NONE || rtp_act > RTP_GELU)
        throw std::runtime_error(
            "gen_vision_mm_seq: rtp_act " + std::to_string(rtp_act) +
            " is not one of 0 none / 1 silu / 2 gelu_tanh. There is no erf "
            "gelu on this kernel; the merger's epilogue stays on the host");

    seq->clear_cmds();

    const uint32_t K_div_k = K / dp.k;
    const uint32_t M_div_m = M_pad / tile_M;
    const uint32_t N_div_n = N / tile_N;

    // ---- the three trip counts, to all 32 cores ---------------------------
    // mm.cc's outer loop runs rtp[0] rounds and its k loop rtp[1] chunks, and
    // it re-reads rtp[2] on every round -- the activation is per-ROUND, not per
    // dispatch, so it cannot be rewritten mid-sequence to give two bound
    // weights different epilogues.
    // That is why the seam puts `activation` in bind() rather than run().
    //
    // Rows 2..5 are the four CT rows; the shim is row 0 and the memtile row 1,
    // so the offset is not cosmetic.
    for (uint32_t row = 2; row < 2 + TOTAL_ROWS; row++) {
        for (uint32_t col = 0; col < TOTAL_COLS; col++) {
            npu_tiles tile = get_tile((int)row, (int)col);
            seq->rtp_write(tile, rtp_address,     M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, (uint32_t)rtp_act);
        }
    }

    for (uint32_t j = 0; j < N_div_n; j++) {
        // Bias prologue for this N strip. The kernel reads the bias out of the
        // leading n elements of the first w chunk, then drains the rest of the
        // k chunks, so this ships n*k elements even though only n are used.
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t c = 0; c < TOTAL_COLS; c++) {
            uint32_t b_offset = weight_offset + j * tile_N + c * dp.n;
            seq->npu_dma_memcpy_nd(
                2, w_arg_id,
                MM2S, IT[c],
                bias_bd, it_channel_1,
                {0, 0, 0, b_offset},
                {1, 1, 1, dp.n * dp.k},
                {0, 0, 0, 1},
                -1, 0, false
            );
        }

        for (uint32_t i = 0; i < M_div_m; i++) {
            const uint32_t round = j * M_div_m + i;
            const uint32_t p = (round % 2) * 8;
            const npu_bd_id y_bd = npu_bd_id(0 + p);
            const npu_bd_id x_bd = npu_bd_id(1 + p);
            const npu_bd_id w_bd = npu_bd_id(2 + p);

            // w: column c owns cols [j*tile_N + c*n, +n). Sent k-tile major so
            // the memtile receives one natural (n, k) block per buffer fill.
            for (uint32_t c = 0; c < TOTAL_COLS; c++) {
                uint32_t w_offset =
                    weight_offset + N + (j * tile_N + c * dp.n) * K;
                seq->npu_dma_memcpy_nd(
                    2, w_arg_id,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {1, K_div_k, dp.n, dp.k},
                    {0, dp.k, K, 1},
                    -1, 0, false
                );
            }

            // y: column c returns a (tile_M, n) chunk, contiguous rows of n
            // that land at DDR row stride N.
            for (uint32_t c = 0; c < TOTAL_COLS; c++) {
                uint32_t y_offset = i * tile_M * N + j * tile_N + c * dp.n;
                seq->npu_dma_memcpy_nd(
                    2, y_arg_id,
                    S2MM, IT[c],
                    y_bd, it_channel_0,
                    {0, 0, 0, y_offset},
                    {1, 1, tile_M, dp.n},
                    {0, 0, N, 1},
                    -1, 0, true
                );
            }

            // x: array row r owns rows [i*tile_M + r*m, +m). Only TOTAL_ROWS
            // shims carry x, unlike w and y which use all TOTAL_COLS.
            for (uint32_t r = 0; r < TOTAL_ROWS; r++) {
                uint32_t x_offset = (i * tile_M + r * dp.m) * K;
                seq->npu_dma_memcpy_nd(
                    2, x_arg_id,
                    MM2S, IT[r],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset},
                    {1, K_div_k, dp.m, dp.k},
                    {0, dp.k, K, 1},
                    -1, 0, false
                );
            }

            // The wait lags a round: round r's descriptors are written while
            // round r-1 is still in flight, which is what the two BD banks
            // above are for.
            if (round > 0) {
                for (uint32_t c = 0; c < TOTAL_COLS; c++)
                    seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }
    for (uint32_t c = 0; c < TOTAL_COLS; c++)
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);

    seq->cmds2seq();
}

} // namespace q38npu
