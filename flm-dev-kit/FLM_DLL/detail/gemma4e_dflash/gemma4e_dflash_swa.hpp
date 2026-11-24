// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file gemma4e_dflash_swa.hpp
/// \brief the sliding-window K/V buffer of the DFlash engine: a LINEAR window buffer with host compaction
///
/// Replaces the 512-row ring of gemma4e_npu (plan D6 as changed on 2026-10-29, FLM_Xclbin/Gemma4/verify_decoding/notes/attn8_L8_results.md):
///   * each SWA layer plane holds kSwaBufferRows rows; the row of position p is `p - base` (`base` is kept by the engine,
///     the same for every SWA layer);
///   * a token is appended at row `ctx - base`;
///   * the attention replay is ONE contiguous range: the last `min(L, W)` keys (W = sliding window = 512, L = keys including the
///     new token) i.e. rows [(L - L_local) - base, ...) in ascending position order, exactly the key order the ring replay produced,
///     so the kernels see the same chunk composition and the decode xclbin needs no change;
///   * when `L - base + kSwaSlack > kSwaBufferRows` the host compacts between two steps: the rows of the last window are moved to
///     the front and `base` advances to floor16(max(0, L - W));
///   * rows past the context are kept ZERO (the attention kernels read whole 16-row chunks and what sits there changes the
///     rounding slightly, see kv_rows_zero in utils/kv_rows_state.hpp).
/// The 8-row verify xclbin uses the same buffer: replay range [S - base, ctx + 8 - base), S = floor16(max(0, ctx - 511)).
#pragma once
#include <algorithm>

#ifndef FLM_SWA_BUFFER_ROWS
#define FLM_SWA_BUFFER_ROWS 2048   // a stress build (-DFLM_SWA_BUFFER_ROWS=640) compacts every ~80 tokens: same code paths, much sooner
#endif
constexpr int kSwaBufferRows = FLM_SWA_BUFFER_ROWS;   ///< rows per K / V plane of one SWA layer buffer
// after a compaction at L the buffer holds up to 527 rows (a 16-row aligned base); a verify batch adds 8 rows and the replay pads to 16
static_assert(kSwaBufferRows >= 576, "the sliding-window buffer must hold a window, a verify batch and the slack");
constexpr int kSwaSlack = 32;          ///< rows kept free behind the context: 8 verify rows + 16-row chunk padding + margin

/// \brief the base after compacting at context length L (a 16-row multiple, never past the window start)
inline int swa_base_after_compaction(int L, int W) {
    const int b = std::max(0, L - W);
    return b / 16 * 16;
}
/// \brief whether a step that reaches `L` rows needs the buffer compacted first (a verify batch passes ctx + n + 1 and compacts BEFORE the batch, never inside it)
inline bool swa_needs_compaction(int L, int base) { return L - base + kSwaSlack > kSwaBufferRows; }
