// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/npu/proj_sequence.hpp
/// \brief Instruction sequence for the fused Q4_K dequant + GEMM engine.
///
/// Port of `gen_dequant_mm()` from
/// FLM_Xclbin/Qwen3_8/dequant_mm/host/seq_gen.hpp, descriptor for descriptor.
/// A changed number here changes hardware behaviour, not an API.
///
/// The design computes
///
///     y[L][Dout] = x[L][Din] . W^T,    W stored [Dout][Din], q4_k, reordered
///
/// on a 4 row x 8 column array of core pairs. One MT tile is
/// M_TILE x K_TILE x N_TILE = 128 x 512 x 64; x broadcasts along a row and the
/// weight broadcasts down a column, so one pass of the whole array covers
/// 4*128 x 512 x 8*64 = 512 x 512 x 512. Hence the three divisibility rules:
/// L, Dout and Din are each a multiple of 512.
///
/// **This is not `mm.xclbin`.** Module 1's projection bitstream fuses the norm
/// and the residual; this one is a bare matmul with an optional activation, so
/// the caller owns every elementwise step around it. That is what lets the MTP
/// head reuse it for six differently-shaped projections out of one weight BO.
///
/// ## The cost model, and why the caller tiles rather than widening L
///
/// Counting the descriptors in one round body below: 8 weight MM2S (512 rows x
/// Din), 4 x MM2S on `IT[0,2,4,6]` (512 rows x Din -- the odd shims receive x
/// over the array), and 8 y S2MM (512 x 512). Over the full `(i, j)` grid:
///
///     weight bytes = (L / 512)    x  q4k_bytes(Dout x Din)
///     x bytes      = (Dout / 512) x  L x Din x 2
///     y bytes      =                 L x Dout x 2          once
///
/// Every term is **linear in L**: the weights are re-streamed once per row
/// block and x once per output-column block, so a wider dispatch amortises
/// nothing. Swapping the `i`/`j` loops does not help either -- both totals are
/// products over the same grid. So `dequant_mm_op` runs a fixed 512-row
/// dispatch and tiles the window on the host, which moves identical DDR traffic
/// while keeping the staging BOs and the `uint32_t` x/y offsets small.
///
/// ## The q4_k byte size, which is NOT get_quantization_byte_size(_, flm_q41)
///
/// Upstream sizes every weight descriptor with
/// `get_quantization_byte_size(elems, flm_q41)`. That reads as q4_1 and is
/// only correct because dequant_mm/host/ carries its own `weights_desc.hpp`
/// whose q4 branch returns 4736 B per 32x256 chunk for *any* non-q8 type. The
/// DLL's real `FLM_DLL/include/weight_desc.hpp` does not: there
/// `is_q4_k(qtype)` is `qtype == flm_q4k` alone, so `flm_q41` takes the slice
/// path and answers 5120 B per chunk. Copying the upstream call into this tree
/// would stride every weight descriptor 8% long and dequantise garbage without
/// a single assertion firing. Use `q38npu::q4k_bytes`, which is the packing
/// this head's weights are actually in and is static_asserted at 4736.
#ifndef __QWEN3_8MTP_NPU_PROJ_SEQUENCE_HPP__
#define __QWEN3_8MTP_NPU_PROJ_SEQUENCE_HPP__

#include "npu_utils/npu_utils.hpp"
#include "typedef.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace q38npu {

// ==========================================================================
// the engine's floorplan
// ==========================================================================
namespace dequant_mm_tiles {

/// `desired_m` upstream: x rows one shim feeds, and the MT tile's M.
constexpr int M_TILE = 128;
/// `desired_k`: the reduction tile, and the RTP `K_div_k`'s divisor.
constexpr int K_TILE = 512;
/// `desired_n`: output rows one core column owns. Also the grain the negative
/// control in the drivers rotates by, and `reorder_cpy`'s interleave depth
/// times QXNX_ROW_BLOCK_SIZE.
constexpr int N_TILE = 64;
/// Shim columns, i.e. `IT0..IT7`.
constexpr int COLUMNS = 8;

/// One array pass: 4 shim-fed row blocks of M_TILE each.
constexpr int PASS_ROWS = M_TILE * 4;            // 512
/// One array pass's output width: COLUMNS core columns of N_TILE each.
constexpr int PASS_COLS = N_TILE * COLUMNS;      // 512

/// Positional against the xclbin's three buffer arguments -- `app(y, x, w)`.
constexpr int y_arg_id = 0;   ///< [L][Dout] bf16 out
constexpr int x_arg_id = 1;   ///< [L][Din]  bf16 in
constexpr int w_arg_id = 2;   ///< [Dout][Din] q4_k, reorder_cpy'd

} // namespace dequant_mm_tiles

// ==========================================================================
/// Where each core keeps its two run-time parameters.
///
/// **These are per-build.** They come out of
/// `FLM_Xclbin/Qwen3_8/dequant_mm/build/bitstream/from_iron/dequant_mm.mlir.prj/
/// input_with_addresses.mlir` as `K_div_k_<row>_<col>` and
/// `rtp_act_<row>_<col>`. The array's four core rows pair up: rows 2 and 4
/// share one address pair, rows 3 and 5 the other. A stale value does not
/// fault -- it writes into some other buffer and the dispatch hangs in
/// `qds_device::wait()`.
/// (memory: rtp-addresses-are-per-build)
struct dequant_mm_rtp_addresses {
    uint32_t k_div_k_even = 54784;   ///< core rows 2 and 4: Din / K_TILE
    uint32_t act_even     = 54848;   ///< core rows 2 and 4: the fused activation
    uint32_t k_div_k_odd  = 61696;   ///< core rows 3 and 5
    uint32_t act_odd      = 61760;   ///< core rows 3 and 5
};

/// The kernel's fused activation selector, `rtp_act`
/// (FLM_Xclbin/Qwen3_8/dequant_mm/kernel/dequant_mm_e.cc).
enum dequant_mm_act : int {
    dequant_mm_act_none = 0,
    dequant_mm_act_silu = 1,
    dequant_mm_act_gelu = 2,
};

// ==========================================================================
/// A subset of the output-column blocks, with its own activation.
///
/// **The activation is not a property of the dispatch.** `rtp_act` is an
/// instruction at a position in the sequence, and the kernel re-reads it every
/// round -- `dequant_mm_e` runs from an infinite `for _ in range_(0xFFFFFFFF)`
/// (iron/dequant_mm_tile.py) and reads `rtp_activate[0]` after the K loop and
/// before the y output loop, once per invocation. So a sequence may write it
/// more than once and cover different column blocks with different
/// activations.
///
/// That is what the woven `up|gate` needs. `stage_mtp_proj_weights` alternates
/// 512 output rows of up with 512 of gate, and 512 is exactly `PASS_COLS`, so
/// the weave lands on the `j` loop: **even `j` is up, odd `j` is gate**. Two
/// passes, `{0, 2, none}` and `{1, 2, silu}`, put silu on the gate bands only.
///
/// ### Why re-writing the RTP mid-sequence is safe
///
/// Each pass ends with a full eight-channel `npu_dma_wait` drain, and that
/// drain is the synchronisation. Once it returns, every y byte of the pass has
/// reached DDR, so the cores have finished its last round -- the RTP read
/// strictly precedes y production. The cores then loop to the next pass's
/// first round and block on `qw_cons_lock`, because no weight BD for it has
/// been issued yet. They cannot reach the RTP read until the emitter issues
/// those BDs, which it does *after* the write. Per-round lock credits balance,
/// which is the same property the design already relies on across dispatches
/// and for the kernel's `is_qw_ping` / `is_x_ping` / `is_y_ping` globals.
///
/// The cost is one pipeline drain per extra pass. DDR traffic is unchanged:
/// the weights stream once per row block and x once per output-column block
/// either way, and the passes cover disjoint `j`.
struct dequant_mm_pass {
    int j_begin  = 0;   ///< first output-column block, in `PASS_COLS` units
    int j_stride = 1;   ///< step between this pass's blocks
    int act      = dequant_mm_act_none;   ///< `dequant_mm_act`
};

// ==========================================================================
/// Emit one projection's whole pass into `seq`.
///
/// Must be re-emitted whenever any of `L`, `Dout`, `Din`, `w_byte_offset` or
/// `act` changes: all five are baked into the descriptors. `dequant_mm_op`
/// pins the first four for the life of an app, so in practice this runs once
/// per projection and never again.
///
/// \param seq            the app's sequence, cleared and rebuilt
/// \param L              x rows. A positive multiple of `PASS_ROWS` (512);
///                       the caller pads, the kernel cannot
/// \param Dout           output width, a positive multiple of `PASS_COLS`
/// \param Din            reduction width, a positive multiple of `K_TILE`
/// \param w_byte_offset  where this matrix starts in the weight BO, in BYTES.
///                       Folded to u32 words by the DMA, so it must be a
///                       multiple of 4 -- checked here rather than assumed
/// \param passes         the output-column blocks, split into one or more
///                       `dequant_mm_pass`es. They must **partition**
///                       `[0, Dout / PASS_COLS)` -- every block covered
///                       exactly once. A gap leaves stale y bytes from the
///                       previous tile and an overlap double-writes, and
///                       neither faults, so both are checked here
/// \param rtp            the per-build core RTP addresses
///
/// \throws std::runtime_error on any shape the array cannot cover
void gen_dequant_mm_seq(npu_sequence* seq, uint32_t L, uint32_t Dout,
                        uint32_t Din, size_t w_byte_offset,
                        const std::vector<dequant_mm_pass>& passes,
                        const dequant_mm_rtp_addresses& rtp =
                            dequant_mm_rtp_addresses{});

/// One pass over every column block, at a single activation.
///
/// The common case, and byte-for-byte the sequence this generator emitted
/// before passes existed.
inline void gen_dequant_mm_seq(npu_sequence* seq, uint32_t L, uint32_t Dout,
                               uint32_t Din, size_t w_byte_offset, int act,
                               const dequant_mm_rtp_addresses& rtp =
                                   dequant_mm_rtp_addresses{}) {
    gen_dequant_mm_seq(seq, L, Dout, Din, w_byte_offset,
                       std::vector<dequant_mm_pass>{{0, 1, act}}, rtp);
}

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_PROJ_SEQUENCE_HPP__
