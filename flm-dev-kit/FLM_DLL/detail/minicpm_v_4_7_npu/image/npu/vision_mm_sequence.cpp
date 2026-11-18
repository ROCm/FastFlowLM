// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

namespace mcv_npu {

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
            std::to_string(dp.cols_split_m ? TOTAL_COLS : TOTAL_ROWS) +
            (dp.cols_split_m ? " array columns" : " array rows") + ") -- the caller pads M, "
            "the kernel cannot");
    if (N == 0 || N % tile_N != 0)
        throw std::runtime_error(
            "gen_vision_mm_seq: N " + std::to_string(N) +
            " is not a positive multiple of " + std::to_string(tile_N) +
            " (n " + std::to_string(dp.n) + " x " +
            std::to_string(dp.cols_split_m ? TOTAL_ROWS : TOTAL_COLS) +
            (dp.cols_split_m ? " array rows)" : " array columns)"));
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
    // How many shims carry each operand, by mapping: x follows M, w follows N.
    const uint32_t n_x = dp.cols_split_m ? TOTAL_COLS : TOTAL_ROWS;   // shims carrying x
    const uint32_t n_w = dp.cols_split_m ? TOTAL_ROWS : TOTAL_COLS;   // shims carrying w

    // ---- the three trip counts, to all 32 cores ---------------------------
    // mm.cc's outer loop runs rtp[0] rounds and its k loop rtp[1] chunks, and
    // it re-reads rtp[2] on every round -- the activation is per-ROUND, not per
    // dispatch, so it cannot be rewritten mid-sequence to give two bound
    // weights different epilogues (memory: rtp-act-is-per-round-not-per-dispatch).
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

    // One descriptor per strip and operand covers up to `kRepeat` consecutive
    // rounds: the first (d3) dimension of a shim BD is a repeat count, so w
    // (stride 0), x (stride tile_M rows) and y (stride tile_M rows) are each
    // programmed once instead of once per round. The per-round form issued ~28
    // commands per round (a ~30 us floor at K = 384, measured), and made the
    // instruction stream (and the ELF built from it) M_div_m times longer. The
    // cores are unchanged: they still run `M_div_m` rounds per strip in order,
    // and the repeat feeds them the same bytes in the same order.
    //
    // A shim BD's repeat count is a 6-bit field, so a strip with more rounds
    // is issued in groups; each group alternates the BD bank, and the wait for
    // a group's y lags by one group exactly like the per-round form did.
    // The repeat stride of x and y (tile_M rows of K resp. N elements) is a BD
    // step field of 20 bits in 32-bit words, i.e. 2^21 bf16 elements. A larger
    // stride is not an error: the BD wraps and the data lands in the wrong rows
    // (measured: the output is uncorrelated with the reference). Such operands
    // are issued per round (group of one round) instead. A one-round y passes
    // stride 0 for its unused repeat dimension: the BD ignores it either way,
    // but npu_dma_memcpy_nd range-checks it and logs "stride_1 out of range".
    constexpr uint64_t kMaxStrideElems = (1ull << 21) - 1;
    uint32_t kRepeat = 32;
    if ((uint64_t)tile_M * K > kMaxStrideElems || (uint64_t)tile_M * N > kMaxStrideElems)
        kRepeat = 1;
    uint32_t group = 0;
    for (uint32_t j = 0; j < N_div_n; j++) {
        // Bias prologue for this N strip. The kernel reads the bias out of the
        // leading n elements of the first w chunk, then drains the rest of the
        // k chunks, so this ships n*k elements even though only n are used.
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t c = 0; c < n_w; c++) {
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

        for (uint32_t i0 = 0; i0 < M_div_m; i0 += kRepeat, group++) {
            const uint32_t reps = (M_div_m - i0 < kRepeat) ? (M_div_m - i0) : kRepeat;
            const uint32_t p = (group % 2) * 8;
            const npu_bd_id y_bd = npu_bd_id(0 + p);
            const npu_bd_id x_bd = npu_bd_id(1 + p);
            const npu_bd_id w_bd = npu_bd_id(2 + p);
            const uint32_t y_rep_stride = reps > 1 ? tile_M * N : 0;

            // w: column c owns cols [j*tile_N + c*n, +n), the same bytes every
            // round (repeat stride 0). Sent k-tile major so the memtile
            // receives one natural (n, k) block per buffer fill.
            for (uint32_t c = 0; c < n_w; c++) {
                uint32_t w_offset =
                    weight_offset + N + (j * tile_N + c * dp.n) * K;
                seq->npu_dma_memcpy_nd(
                    2, w_arg_id,
                    MM2S, IT[c],
                    w_bd, it_channel_1,
                    {0, 0, 0, w_offset},
                    {reps, K_div_k, dp.n, dp.k},
                    {0, dp.k, K, 1},
                    -1, 0, false
                );
            }

            // y. A/B: column c returns a (tile_M, n) chunk per round, contiguous rows of n that
            // land at DDR row stride N; round r is tile_M rows further down. C: column c
            // returns the (m, tile_N) block of FULL token rows [i*tile_M + c*m, +m): rows of
            // tile_N contiguous elements (768 B), at the same row stride.
            for (uint32_t c = 0; c < TOTAL_COLS; c++) {
                if (dp.cols_split_m) {
                    uint32_t y_offset = (i0 * tile_M + c * dp.m) * N + j * tile_N;
                    seq->npu_dma_memcpy_nd(
                        2, y_arg_id,
                        S2MM, IT[c],
                        y_bd, it_channel_0,
                        {0, 0, 0, y_offset},
                        {1, reps, dp.m, tile_N},
                        {0, y_rep_stride, N, 1},
                        -1, 0, true
                    );
                } else {
                    uint32_t y_offset = i0 * tile_M * N + j * tile_N + c * dp.n;
                    seq->npu_dma_memcpy_nd(
                        2, y_arg_id,
                        S2MM, IT[c],
                        y_bd, it_channel_0,
                        {0, 0, 0, y_offset},
                        {1, reps, tile_M, dp.n},
                        {0, y_rep_stride, N, 1},
                        -1, 0, true
                    );
                }
            }

            // x: shim r owns rows [i*tile_M + r*m, +m) (A/B: array row r, four shims; C: array
            // column r, eight shims).
            for (uint32_t r = 0; r < n_x; r++) {
                uint32_t x_offset = (i0 * tile_M + r * dp.m) * K;
                seq->npu_dma_memcpy_nd(
                    2, x_arg_id,
                    MM2S, IT[r],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset},
                    {reps, K_div_k, dp.m, dp.k},
                    {tile_M * K, dp.k, K, 1},
                    -1, 0, false
                );
            }

            // The wait lags a group: this group's descriptors are written
            // while the previous group is still in flight, which is what the
            // two BD banks above are for.
            if (group > 0) {
                for (uint32_t c = 0; c < TOTAL_COLS; c++)
                    seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
            }
        }
    }
    for (uint32_t c = 0; c < TOTAL_COLS; c++)
        seq->npu_dma_wait(IT[c], S2MM, it_channel_0);

    seq->cmds2seq();
}

void gen_vision_mm_p_seq(npu_sequence* seq,
                         uint32_t M_pad, uint32_t K, uint32_t N,
                         const design_params& dp,
                         int32_t rtp_act, uint32_t rtp_address) {
    constexpr npu_tiles IT[8] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};
    constexpr uint32_t kPiece = 3456;   // bytes, 864 words: divides 72*k and 108*k for k = 384
    const uint32_t tile_M = dp.tile_M(), tile_N = dp.tile_N();
    if (!dp.packed || !dp.cols_split_m || dp.k != 384)
        throw std::runtime_error("gen_vision_mm_p_seq: not design P's params");
    if (M_pad == 0 || M_pad % tile_M != 0 || N == 0 || N % tile_N != 0 || K == 0 || K % dp.k != 0)
        throw std::runtime_error("gen_vision_mm_p_seq: M " + std::to_string(M_pad) + " K " +
                                 std::to_string(K) + " N " + std::to_string(N) +
                                 " is not padded to the " + std::to_string(tile_M) + " x " +
                                 std::to_string(dp.k) + " x " + std::to_string(tile_N) + " tile");
    if (rtp_act < RTP_NONE || rtp_act > RTP_GELU)
        throw std::runtime_error("gen_vision_mm_p_seq: rtp_act " + std::to_string(rtp_act));

    const uint32_t x_tile_bytes  = dp.m * K / 8 * 9;   // one column's m rows, all K
    const uint32_t w_strip_bytes = dp.n * K / 8 * 9;   // one row's n channels, all K
    const uint32_t bias_bytes    = N * 2;
    const uint32_t w_mem_bytes   = dp.n * dp.k / 8 * 9;   // one memtile w buffer
    if (x_tile_bytes % kPiece != 0 || w_strip_bytes % kPiece != 0)
        throw std::runtime_error("gen_vision_mm_p_seq: K " + std::to_string(K) + " does not split into pieces");

    seq->clear_cmds();
    const uint32_t K_div_k = K / dp.k, M_div_m = M_pad / tile_M, N_div_n = N / tile_N;
    for (uint32_t row = 2; row < 2 + TOTAL_ROWS; row++)
        for (uint32_t col = 0; col < TOTAL_COLS; col++) {
            npu_tiles tile = get_tile((int)row, (int)col);
            seq->rtp_write(tile, rtp_address, M_div_m);
            seq->rtp_write(tile, rtp_address + 4, K_div_k);
            seq->rtp_write(tile, rtp_address + 8, rtp_act);
        }
    // Repeat strides must fit a BD step field (2^20 words); a larger one wraps silently.
    constexpr uint64_t kMaxStrideWords = (1ull << 20) - 1;
    const uint32_t kRepeat = ((uint64_t)TOTAL_COLS * x_tile_bytes / 4 > kMaxStrideWords ||
                              (uint64_t)tile_M * N / 2 > kMaxStrideWords) ? 1 : 32;
    uint32_t group = 0;
    for (uint32_t j = 0; j < N_div_n; j++) {
        // bias prologue: one memtile w buffer read linearly from the strip's bias;
        // its first 2n bytes are the bias, the rest is drained unread
        const npu_bd_id bias_bd = npu_bd_id(6 + (j % 2));
        for (uint32_t r = 0; r < TOTAL_ROWS; r++)
            seq->npu_dma_memcpy_nd(1, w_arg_id, MM2S, IT[r], bias_bd, it_channel_1,
                {0, 0, 0, (j * tile_N + r * dp.n) * 2}, {1, 1, 1, w_mem_bytes}, {0, 0, 0, 1}, -1, 0, false);
        for (uint32_t i0 = 0; i0 < M_div_m; i0 += kRepeat, group++) {
            const uint32_t reps = (M_div_m - i0 < kRepeat) ? (M_div_m - i0) : kRepeat;
            const uint32_t p = (group % 2) * 8;
            const uint32_t y_rep_stride = reps > 1 ? tile_M * N : 0;   // see gen_vision_mm_seq
            for (uint32_t r = 0; r < TOTAL_ROWS; r++) {
                const uint32_t w_offset = bias_bytes + (j * TOTAL_ROWS + r) * w_strip_bytes;
                seq->npu_dma_memcpy_nd(1, w_arg_id, MM2S, IT[r], (npu_bd_id)(2 + p), it_channel_1,
                    {0, 0, 0, w_offset}, {reps, 1, w_strip_bytes / kPiece, kPiece}, {0, 0, kPiece, 1},
                    -1, 0, false);
            }
            for (uint32_t c = 0; c < TOTAL_COLS; c++) {
                const uint32_t y_offset = (i0 * tile_M + c * dp.m) * N + j * tile_N;
                seq->npu_dma_memcpy_nd(2, y_arg_id, S2MM, IT[c], (npu_bd_id)(0 + p), it_channel_0,
                    {0, 0, 0, y_offset}, {1, reps, dp.m, tile_N}, {0, y_rep_stride, N, 1}, -1, 0, true);
            }
            for (uint32_t c = 0; c < TOTAL_COLS; c++) {
                const uint32_t x_offset = (i0 * TOTAL_COLS + c) * x_tile_bytes;
                seq->npu_dma_memcpy_nd(1, x_arg_id, MM2S, IT[c], (npu_bd_id)(1 + p), it_channel_0,
                    {0, 0, 0, x_offset}, {reps, 1, x_tile_bytes / kPiece, kPiece},
                    {TOTAL_COLS * x_tile_bytes, 0, kPiece, 1}, -1, 0, false);
            }
            // the wait lags a group, as in gen_vision_mm_seq
            if (group > 0)
                for (uint32_t c = 0; c < TOTAL_COLS; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
        }
    }
    for (uint32_t c = 0; c < TOTAL_COLS; c++) seq->npu_dma_wait(IT[c], S2MM, it_channel_0);
    seq->cmds2seq();
}

} // namespace mcv_npu
