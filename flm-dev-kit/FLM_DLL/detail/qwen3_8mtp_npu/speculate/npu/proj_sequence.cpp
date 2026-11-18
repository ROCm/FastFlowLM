// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/npu/proj_sequence.cpp
/// \brief The fused dequant + GEMM engine's pass, descriptor for descriptor.
///
/// Every descriptor below is `gen_dequant_mm()`'s with the loose parameters
/// replaced by named constants and the three `assert`s promoted to throws --
/// an assert is compiled out of the release .so, and each of these three is a
/// shape that would otherwise hang the dispatch rather than fail it.
///
/// Three deliberate divergences from upstream. Two are explained at length in
/// the header: the weight byte size comes from `q4k_bytes` and not from
/// `get_quantization_byte_size(_, flm_q41)`, and `w_offset` is a `size_t`.
/// Upstream's is a 32-bit `int`; it holds for this head's 245 MB blob but
/// overflows on module 1's multi-GB weight BOs, and this file is the copy
/// module 1 would reuse.
///
/// The third is `dequant_mm_pass`: upstream emits its 32 `rtp_write`s once at
/// the top, which makes the activation uniform across the dispatch. That is a
/// property of the emitter, not of the hardware -- see the header. Emitting
/// the grid once per pass, each with its own activation and its own subset of
/// the column blocks, is what lets the woven `up|gate` take silu on its gate
/// bands alone. A single pass is byte-for-byte upstream's sequence.
#include "speculate/npu/proj_sequence.hpp"

#include "verify/npu/q4nx.hpp"    // q38npu::q4k_bytes

#include <stdexcept>
#include <string>

namespace q38npu {

using namespace dequant_mm_tiles;

namespace {

/// The eight shims, indexable. Upstream's `IT[8]`.
constexpr npu_tiles IT[COLUMNS] = {IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("gen_dequant_mm_seq: " + what);
}

/// Everything a pass needs that comes from the shape rather than the pass.
struct grid {
    uint32_t Dout = 0, Din = 0;
    uint32_t M_div_m = 0;   ///< row blocks
    uint32_t N_div_n = 0;   ///< output-column blocks
    uint32_t K_div_k = 0;   ///< the cores' reduction count
    size_t   w_byte_offset = 0;
    size_t   w_per_col   = 0;   ///< one core column's weight slab
    size_t   w_half_pass = 0;   ///< how far ahead the high four shims read
};

/// One pass: its RTP write, its rounds, and the drain that closes it.
void emit_pass(npu_sequence* seq, const dequant_mm_pass& pass, const grid& g,
               const dequant_mm_rtp_addresses& rtp) {
    const uint32_t Dout = g.Dout, Din = g.Din, K_div_k = g.K_div_k;

    // ---- RTP: the reduction count and the activation ----------------------
    // Every core needs both. The four core rows pair up on two address pairs
    // (rows 2 and 4, rows 3 and 5) -- see dequant_mm_rtp_addresses.
    //
    // Re-written per pass, which the previous pass's trailing drain makes
    // safe: it leaves every core blocked on `qw_cons_lock` waiting for this
    // pass's first round, and that acquire is upstream of the kernel's
    // `rtp_activate[0]` read. See dequant_mm_pass for the full argument.
    for (int row = 2; row < 6; row += 2)
        for (int col = 0; col < COLUMNS; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp.k_div_k_even, K_div_k);
            seq->rtp_write(tile, rtp.act_even,     (uint32_t)pass.act);
        }
    for (int row = 3; row < 6; row += 2)
        for (int col = 0; col < COLUMNS; col++) {
            npu_tiles tile = get_tile(row, col);
            seq->rtp_write(tile, rtp.k_div_k_odd, K_div_k);
            seq->rtp_write(tile, rtp.act_odd,     (uint32_t)pass.act);
        }

    // Counts the rounds this pass actually emits, NOT `i * N_div_n + j`: the
    // ping/pong bank and the one-behind wait below both count *dispatched*
    // rounds, and a strided pass skips column blocks. Restarting at 0 per pass
    // is right because a pass begins with every BD drained.
    uint32_t round = 0;

    // ---- the (row block, column block) grid -------------------------------
    for (uint32_t i = 0; i < g.M_div_m; i++) {
        for (uint32_t j = (uint32_t)pass.j_begin; j < g.N_div_n;
             j += (uint32_t)pass.j_stride, round++) {

            // Ping/pong: a round's BDs must not overwrite the previous round's
            // while it is still in flight, and the wait below is one round
            // behind. Eight ids per bank, five used.
            const int bd_offset = (int)(round % 2) * 8;

            // x is re-read for every output-column block and y is written
            // once; the weight walks j and is replayed for every row block.
            const uint32_t y_offset = i * PASS_ROWS * Dout + j * PASS_COLS;
            const uint32_t x_offset = i * PASS_ROWS * Din;
            const size_t   w_offset =
                g.w_byte_offset + (size_t)j * q4k_bytes((size_t)PASS_COLS * Din);

            const npu_bd_id y_bd_0 = (npu_bd_id)(0 + bd_offset);
            const npu_bd_id x_bd   = (npu_bd_id)(1 + bd_offset);
            const npu_bd_id w_bd_0 = (npu_bd_id)(2 + bd_offset);
            const npu_bd_id y_bd_1 = (npu_bd_id)(3 + bd_offset);
            const npu_bd_id w_bd_1 = (npu_bd_id)(4 + bd_offset);

            for (uint32_t col = 0; col < 4; col++) {
                // ---- weights, 8 streams ---------------------------------
                // Core column c owns output rows [c*N_TILE, (c+1)*N_TILE) of
                // this round, and the low four columns are addressed a half
                // pass ahead of the high four.
                const size_t w_col = w_offset + (size_t)col * g.w_per_col;
                seq->npu_dma_memcpy_nd(
                    sizeof(char), w_arg_id,
                    MM2S, IT[col],
                    w_bd_0, it_channel_1,
                    {0, 0, 0, (uint32_t)w_col},
                    {1, 1, 1, (uint32_t)g.w_per_col},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache);
                seq->npu_dma_memcpy_nd(
                    sizeof(char), w_arg_id,
                    MM2S, IT[col + 4],
                    w_bd_1, it_channel_1,
                    {0, 0, 0, (uint32_t)(w_col + g.w_half_pass)},
                    {1, 1, 1, (uint32_t)g.w_per_col},
                    {0, 0, 0, 1},
                    -1, 0, false, aggressive_cache);

                // ---- receive y, 8 streams -------------------------------
                // Each shim returns M_TILE rows of its column's N_TILE
                // outputs, strided by the full Dout because this round covers
                // only PASS_COLS of it.
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), y_arg_id,
                    S2MM, IT[col],
                    y_bd_0, it_channel_0,
                    {0, 0, 0, y_offset + col * M_TILE * Dout},
                    {1, 1, (uint32_t)M_TILE, (uint32_t)(N_TILE * 4)},
                    {0, 0, Dout, 1},
                    -1, 0, true);
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), y_arg_id,
                    S2MM, IT[col + 4],
                    y_bd_1, it_channel_0,
                    {0, 0, 0, y_offset + col * M_TILE * Dout + 4 * N_TILE},
                    {1, 1, (uint32_t)M_TILE, (uint32_t)(N_TILE * 4)},
                    {0, 0, Dout, 1},
                    -1, 0, true);

                // ---- send x, only 4 streams -----------------------------
                // On the EVEN shims alone: a CU's odd column receives x over
                // the array rather than from its own shim, so there is no
                // IT[col*2 + 1] descriptor to write. Adding one does not
                // fault, it deadlocks -- the extra MM2S has no consumer and
                // parks holding the shim's lock credits.
                // (memory: unrouted-mm2s-chain-still-fires)
                seq->npu_dma_memcpy_nd(
                    sizeof(bf16), x_arg_id,
                    MM2S, IT[col * 2],
                    x_bd, it_channel_0,
                    {0, 0, 0, x_offset + col * M_TILE * Din},
                    {1, K_div_k, (uint32_t)M_TILE, (uint32_t)K_TILE},
                    {0, (uint32_t)K_TILE, Din, 1},
                    -1, 0, false);
            } // col

            // One round behind: round n's BDs reuse round n-2's ids, so it is
            // round n-1 that has to have landed before they are posted.
            if (round > 0)
                for (int r = 0; r < 4; r++) {
                    seq->npu_dma_wait(IT[r],     S2MM, it_channel_0);
                    seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
                }
        } // j
    } // i

    // Drain all eight shims before the pass ends. Two jobs. It is what makes a
    // re-dispatch of the whole sequence safe -- `dequant_mm_op` tiles a window
    // by running it repeatedly, and the ping/pong above only guards BD reuse
    // *within* a pass. And it is the barrier the next pass's RTP write needs
    // in order not to race the cores.
    for (int r = 0; r < 4; r++) {
        seq->npu_dma_wait(IT[r],     S2MM, it_channel_0);
        seq->npu_dma_wait(IT[r + 4], S2MM, it_channel_0);
    }
}

} // namespace

void gen_dequant_mm_seq(npu_sequence* seq, uint32_t L, uint32_t Dout,
                        uint32_t Din, size_t w_byte_offset,
                        const std::vector<dequant_mm_pass>& passes,
                        const dequant_mm_rtp_addresses& rtp) {
    // ---- shape ------------------------------------------------------------
    // Upstream asserts these. They are throws here because an assert vanishes
    // under NDEBUG and every one of them is a silent hang, not a wrong number:
    // a short L leaves S2MM channels un-drained, a ragged Dout points the last
    // round's weight descriptor past the matrix, and a Din that is not a whole
    // number of K tiles makes `K_div_k` truncate so the cores stop reducing
    // early and then wait forever for the rest of the stream.
    require(L > 0 && L % PASS_ROWS == 0,
            "L " + std::to_string(L) + " is not a positive multiple of " +
            std::to_string(PASS_ROWS) + " -- the caller pads the window");
    require(Dout > 0 && Dout % PASS_COLS == 0,
            "Dout " + std::to_string(Dout) + " is not a positive multiple of " +
            std::to_string(PASS_COLS) + ", one array pass's output width");
    require(Din > 0 && Din % K_TILE == 0,
            "Din " + std::to_string(Din) + " is not a positive multiple of " +
            std::to_string(K_TILE) + ", the reduction tile");
    // reorder_cpy walks the matrix in 32x256 blocks, so a Din that is not a
    // whole number of them makes `blocks_per_row` truncate and folds the
    // buffer's head back in as padding. Nothing downstream notices.
    require(Din % 256 == 0, "Din is not a whole number of q4_k column blocks");
    // npu_dma_memcpy_nd folds a 1-byte element type to u32, halving-by-four
    // both the offset and the length. An odd byte offset silently truncates.
    require(w_byte_offset % 4 == 0,
            "the weight byte offset is not a multiple of 4 -- the DMA folds "
            "it to u32 words");

    grid g;
    g.Dout    = Dout;
    g.Din     = Din;
    g.M_div_m = L / PASS_ROWS;
    g.N_div_n = Dout / PASS_COLS;
    g.K_div_k = Din / K_TILE;
    g.w_byte_offset = w_byte_offset;
    // One round's weight slab, and one core column's share of it. Both are
    // whole 4736 B blocks, so both are multiples of 4 and the DMA's u32 fold
    // is exact.
    g.w_per_col   = q4k_bytes((size_t)N_TILE * Din);
    g.w_half_pass = q4k_bytes((size_t)N_TILE * Din * 4);

    // ---- the passes must partition the column blocks ----------------------
    // Neither failure faults. A block no pass covers keeps whatever the
    // previous tile left in the y BO -- which looks like a plausible row, not
    // a zero or a NaN -- and a block two passes cover is written twice with
    // the later activation winning, which is invisible when both ask for
    // `none`.
    require(!passes.empty(), "no passes");
    std::vector<int> covered_by(g.N_div_n, -1);
    for (size_t p = 0; p < passes.size(); p++) {
        const dequant_mm_pass& pa = passes[p];
        require(pa.act >= dequant_mm_act_none && pa.act <= dequant_mm_act_gelu,
                "pass " + std::to_string(p) + "'s activation selector " +
                std::to_string(pa.act) + " is not one of none/silu/gelu");
        require(pa.j_stride >= 1,
                "pass " + std::to_string(p) + "'s column-block stride " +
                std::to_string(pa.j_stride) + " is not positive");
        require(pa.j_begin >= 0 && (uint32_t)pa.j_begin < g.N_div_n,
                "pass " + std::to_string(p) + " starts at column block " +
                std::to_string(pa.j_begin) + ", outside [0, " +
                std::to_string(g.N_div_n) + ")");
        for (uint32_t j = (uint32_t)pa.j_begin; j < g.N_div_n;
             j += (uint32_t)pa.j_stride) {
            require(covered_by[j] < 0,
                    "column block " + std::to_string(j) + " is covered by "
                    "both pass " + std::to_string(covered_by[j]) +
                    " and pass " + std::to_string(p));
            covered_by[j] = (int)p;
        }
    }
    for (uint32_t j = 0; j < g.N_div_n; j++)
        require(covered_by[j] >= 0,
                "column block " + std::to_string(j) + " of " +
                std::to_string(g.N_div_n) + " is covered by no pass, so its y "
                "bytes would keep the previous tile's values");

    seq->clear_cmds();
    for (const dequant_mm_pass& pass : passes)
        emit_pass(seq, pass, g, rtp);
    seq->cmds2seq();
}

} // namespace q38npu
