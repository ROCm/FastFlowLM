// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file image/npu/vision_mm_sequence.hpp
/// \brief Instruction sequence for the vision tower's matmul array.
///
/// Port of `gen_mm_seq()` from
/// FLM_Xclbin/Qwen3_8/vision_mm/host/seq_gen.hpp, descriptor for descriptor,
/// via Qwen3.8-27B's tower (`qwen3_8mtp_npu/image/npu/`). The bitstream is
/// REUSED UNCHANGED for MiniCPM; only the namespace moved.
/// A changed number here changes hardware behaviour, not an API.
///
/// `vision_mm` computes
///
///     y[M][N] = act(x[M][K] W[N][K]^T + bias[N])
///
/// and nothing else -- no norm, no residual, no split of a fused output. The
/// weight layout is PyTorch's `nn.Linear` order, which is how the tensors ship,
/// so no transpose happens anywhere between the checkpoint and the BO.
///
/// ### Two designs, and why
///
/// The array is 4 rows x 8 columns. Rows split M and columns split N, so one
/// round covers `m*4` rows by `n*8` columns of the output and the host walks the
/// output in those tiles. (This is the transpose of the SmolVLA scaffold the
/// design came from, which split M across the columns; rows-for-M gives an M
/// granularity of `m*4` rather than `m*8`, which is what makes the merger
/// layers' L/4 rows affordable.)
///
///     VISION_MM_A   m=64 k=384 n=48    array pass 256 x 384
///     VISION_MM_B   m=32 k=384 n=128   array pass 128 x 1024
///
/// Both are separate bitstreams and therefore separate `hw_context`s -- a
/// runlist is bound to one, so they cannot be merged
/// (memory: verify-xclbins-are-two-separate-bitstreams). B exists for
/// the projector's `merger.fc2` alone: its N of 1024 (MiniCPM's language D) is
/// not a multiple of A's 384 but is exactly one of B's 1024. (In Qwen3.8 it was
/// 5120, five of them.)
///
/// ### DDR layout the descriptors assume
///
///     x   (M, K) row-major bf16
///     w   N bias elements, THEN (N, K) row-major bf16
///     y   (M, N) row-major bf16
///
/// The bias living in front of the weights is not a packing convenience: the
/// kernel reads it out of row 0 of the memtile's `(n_slice, k)` block, one
/// `k_slice`-wide band per chunk, so a bias wider than `k_slice` arrives spread
/// over several chunks (memory: bias-prologue-spans-k-chunks). That is design
/// B's case, n_slice 128 > k_slice 64, and `passbias_aie()` in
/// `FLM_Xclbin/Qwen3_8/vision_mm/kernel/mm.cc` is what handles it.
#ifndef __MINICPM_V_4_7_NPU_VISION_MM_SEQUENCE_HPP__
#define __MINICPM_V_4_7_NPU_VISION_MM_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstdint>

namespace mcv_npu {

// ==========================================================================
/// The matmul designs' floorplan.
// ==========================================================================
namespace vision_mm_tiles {

/// Array rows, which split M. Only these four shims carry x.
constexpr uint32_t TOTAL_ROWS = 4;
/// Array columns, which split N. All eight carry w and return y.
constexpr uint32_t TOTAL_COLS = 8;

/// Positional against the xclbin's three buffer arguments. `host.cpp` calls
/// `app(y_bo, x_bo, w_bo)`, so y is 0.
constexpr int y_arg_id = 0;   ///< [M_pad][N] bf16 out
constexpr int x_arg_id = 1;   ///< [M_pad][K]  bf16 in
constexpr int w_arg_id = 2;   ///< [N] bias then [N][K] bf16 in

/// One design's per-core tile. `k` is 384 in all of them, which is what keeps
/// the mmul's inner loop trip count the same either way.
///
/// `cols_split_m` is the array mapping. A and B split M over the four ROWS and N
/// over the eight COLUMNS (an array pass is m*4 x n*8). C is the transpose:
/// M over the eight columns, N over the four rows (an array pass is m*8 x n*4),
/// so a column's output is full token rows (long DDR write segments) and the
/// weights are streamed once per 512 tokens instead of per 256.
struct design_params {
    uint32_t m, k, n;
    bool cols_split_m = false;
    bool packed = false;   ///< x and w are packed bfp16 (design P, vision_bfp16.hpp), not bf16
    uint32_t tile_M() const { return m * (cols_split_m ? TOTAL_COLS : TOTAL_ROWS); }
    uint32_t tile_N() const { return n * (cols_split_m ? TOTAL_ROWS : TOTAL_COLS); }
};

constexpr design_params A{64, 384, 48};          ///< everything but merger.fc2 (N % 384 == 0)
constexpr design_params B{32, 384, 128};         ///< merger.fc2, N = 1024
constexpr design_params C{64, 384, 96, true};    ///< N % 384 == 0, 512-row tiles (vision_mm_c.xclbin, optional)
constexpr design_params P{64, 384, 96, true, true};   ///< C's mapping on packed operands (vision_mm_p.xclbin, optional)

/// `rtp_act`, the fused epilogue the kernel bakes into the round.
///
/// **There is no gelu_erf.** `mcv::vision::activation::gelu_erf` -- the
/// merger's, and only the merger's -- has to bind `NONE` here and keep its
/// epilogue on the host. Substituting `GELU` is wrong by ~1e-3 absolute on the
/// last rows the language model reads and reads as nothing at all downstream,
/// which is the entire reason the seam makes it a separate enumerator.
constexpr int32_t RTP_NONE = 0;
constexpr int32_t RTP_SILU = 1;
constexpr int32_t RTP_GELU = 2;   ///< gelu_pytorch_tanh; kernel/lut_based_ops.h's
                                  ///< getGeluBf16_nonLUT, verified tanh-form

} // namespace vision_mm_tiles

// ==========================================================================
/// Where each matmul core keeps its three run-time values.
///
/// **This is per-build, and it is per DESIGN.** Both come out of the
/// bitstream's `build/<design>/bitstream/from_iron/mm.mlir.prj/
/// input_with_addresses.mlir` and `FLM_Xclbin/Qwen3_8/vision_mm/tools/
/// extract_rtp_addr.sh` is what reads them into `build/<design>/rtp_addr.txt`.
/// A stale value does not fault -- it writes the trip counts into some other
/// core buffer, the cores never see them, and the dispatch hangs in
/// `qds_device::wait()` (memory: rtp-addresses-are-per-build).
///
/// Whenever either bitstream is rebuilt, diff these against the two
/// `rtp_addr.txt` files. The N16 driver's check 1 does exactly that, first,
/// so a rebuild is a readable FAIL and not a two-minute hang.
///
/// Word 0 is `M/tile_M` (the core's outer round count), word 1 `K/k` (its inner
/// k-chunk count) and word 2 `rtp_act`.
struct vision_mm_rtp_addresses {
    uint32_t a = 24576;   ///< VISION_MM_A
    uint32_t b = 7936;    ///< VISION_MM_B
    uint32_t c = 65024;   ///< VISION_MM_C (FLM_Xclbin/MiniCPM_V/vision_mm_c)
    uint32_t p = 44416;   ///< VISION_MM_P (FLM_Xclbin/MiniCPM_V/vision_mm_p)
};

// ==========================================================================
/// Emit one matmul's whole pass into `seq`.
///
/// Must be re-emitted whenever any of M_pad, K, N or `rtp_act` changes: the
/// first three are the descriptors' shape and the last is an RTP write.
/// `npu_app::operator()` notices the sequence version moved and re-sets up the
/// kernel, which is why each `mm_op` owns its own app rather than sharing one --
/// 111 ops on a shared app would rebuild the executable on every dispatch.
///
/// \param seq            the app's sequence, cleared and rebuilt
/// \param M_pad          rows the array walks. Must be a positive multiple of
///                       `dp.tile_M()`; the caller pads, the kernel cannot
/// \param K              inner dim. Must be a multiple of `dp.k`
/// \param N              output dim. Must be a multiple of `dp.tile_N()`
/// \param dp             the design this sequence is for
/// \param weight_offset  element offset of the bias within the w BO. 0 when the
///                       op owns its weight BO, which is how this port uses it
/// \param rtp_act        one of vision_mm_tiles::RTP_*
/// \param rtp_address    the per-build RTP address FOR THIS DESIGN
void gen_vision_mm_seq(npu_sequence* seq,
                       uint32_t M_pad, uint32_t K, uint32_t N,
                       const vision_mm_tiles::design_params& dp,
                       uint32_t weight_offset, int32_t rtp_act,
                       uint32_t rtp_address);

/// Design P's sequence (FLM_Xclbin/MiniCPM_V/vision_mm_p/host/seq_gen.hpp): C's
/// mapping with x and w as packed bfp16 tiles. Offsets are bytes: w is the N bf16
/// bias, then N/n strips of n*K/8*9 B; x is M/m tiles of m*K/8*9 B; y as C.
void gen_vision_mm_p_seq(npu_sequence* seq,
                         uint32_t M_pad, uint32_t K, uint32_t N,
                         const vision_mm_tiles::design_params& dp,
                         int32_t rtp_act, uint32_t rtp_address);

} // namespace mcv_npu

#endif  // __MINICPM_V_4_7_NPU_VISION_MM_SEQUENCE_HPP__
