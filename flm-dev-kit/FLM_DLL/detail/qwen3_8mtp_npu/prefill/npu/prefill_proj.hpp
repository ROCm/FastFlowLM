// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/prefill_proj.hpp
/// \brief A full-attention layer's prefill projections on `dequant_mm.xclbin`.
///
/// Four `dequant_mm_op`s over ONE weight BO -- `npu_attn_layer`'s existing
/// ~209 MB `proj_`, unchanged, read at four byte offsets. Seven matmuls in four
/// dispatches, because the staged `q|k|v|z` is one fused matrix and the woven
/// `up|gate` another.
///
/// The sibling of `speculate/npu/mtp_proj.hpp`, over `attn_layout` instead of
/// `mtp_layout` and without the fusion matmul the draft head has. One object
/// serves all 16 attention layers: the weight BO is a `run()` argument, not a
/// member, so what is per-layer is the buffer and what is per-shape is the app.
///
/// ## Why the existing BO is already in the right order
///
/// `stage_attn_proj_weights` (verify/npu/attn_weights.hpp) writes each
/// sub-tensor through `q38npu::reorder_cpy(..., vertical_blocks = 2)`, which is
/// the same function dequant_mm's host driver calls with the same depth
/// (`NPU_N_TILE / QXNX_ROW_BLOCK_SIZE = 64 / 32`). Reading that function's body
/// (verify/npu/q4nx.hpp): the outer loop steps `r += vertical_blocks` over
/// 32-row q4_k blocks, interleaving two of them column by column, so **the
/// permutation is local to each group of `2 * 32 = 64` weight rows** and
/// nothing crosses a 64-row boundary.
///
/// Every sub-tensor here is a multiple of 64 rows -- q 6144, k 1024, v 1024,
/// z 6144, o 5120, up and gate 17408 each, down 5120 -- so reordering each one
/// separately is byte for byte reordering their concatenation. That is what
/// lets dispatch 1 read `q|k|v|z` as one 14336-row matrix and the MLP dispatch
/// read the woven `up|gate` as one 34816-row matrix, with no restaging and no
/// second copy of the weights.
///
/// The argument was checked, not trusted, when this was ported: re-running
/// `reorder_cpy` over the concatenation and `memcmp`ing it against what the
/// staging wrote, and scoring every dispatch against a same-bytes reference
/// with a rotated control.
///
/// ## `q|k|v|z`, not `k|v|q|z`
///
/// **The staging order here is the opposite of the draft head's.**
/// `stage_mtp_proj_weights` writes k, then v, then q, then z -- the RoPE tile
/// consumes kv before q -- so `mtp_proj::qkv_z` gets `[q | gate]` back
/// contiguously at the tail and scatters it with three memcpys.
/// `stage_attn_proj_weights` writes q, then k, then v, then z, so here q is at
/// the *head* of the row and the gate at the tail, `qg` takes two memcpys, and
/// a reader that assumed the other order would get plausible garbage. The op
/// itself cannot tell: both are legal 14336-row matrices.
///
/// ## The two shapes that are not D
///
/// `o` reduces over the gated attention row (Din = DQ = 6144) and `down` over
/// the MLP hidden (Din = intermediate_size = 17408). Passing D for either
/// strides the wrong way through a matrix of exactly the same byte count, so
/// nothing asserts and the cores dequantise the wrong rows -- the same trap
/// `reorder_cpy`'s own `col` parameter carries.
#ifndef __QWEN3_8MTP_NPU_PREFILL_PROJ_HPP__
#define __QWEN3_8MTP_NPU_PREFILL_PROJ_HPP__

#include "speculate/npu/proj_op.hpp"        // q38npu::dequant_mm_op
#include "common/cpu_simd.hpp"              // the vectorised marshalling
#include "verify/npu/attn_weights.hpp"      // q38npu::attn_layout

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

class prefill_proj {
public:
    /// \param mgr     an app manager over dequant_mm.xclbin. Four apps are
    ///                taken from it, one per dispatch: a sequence is per-shape
    ///                and an app owns exactly one.
    /// \param layout  the SAME layout the weight BOs were staged from
    /// \param rtp     the per-build core RTP addresses
    prefill_proj(npu_app_manager* mgr, const attn_layout& layout,
                 const dequant_mm_rtp_addresses& rtp = dequant_mm_rtp_addresses{})
        : t_(layout) {
        if (!mgr) throw std::runtime_error("prefill_proj: no app manager");
        _check_geometry();

        const int D  = t_.D;
        const int DI = t_.intermediate_size;

        // q | k | v | z as one matrix, from byte 0. Legal only because
        // w_z_offset immediately follows q|k|v and w_o_offset immediately
        // follows z -- invariants stage_attn_proj_weights and attn_layout::build
        // already establish, and _check_geometry() re-asserts because this is
        // the dispatch that depends on them.
        qkvz_ = std::make_unique<dequant_mm_op>(mgr->create_app(), QKVZ_OUT(), D,
                                                /*w_byte_offset=*/0,
                                                dequant_mm_act_none, rtp);
        o_    = std::make_unique<dequant_mm_op>(mgr->create_app(), D, t_.DQ,
                                                t_.w_o_offset,
                                                dequant_mm_act_none, rtp);
        // The weave alternates 512 output rows of up with 512 of gate, and 512
        // is exactly PASS_COLS, so the weave lands on the generator's `j` loop:
        // even j is up, odd j is gate. Two passes therefore put the kernel's
        // fused silu on the gate bands and nothing else. The RTP is not uniform
        // across a dispatch -- it is an instruction at a position in the
        // sequence, re-read by the cores every round. See dequant_mm_pass.
        upgate_ = std::make_unique<dequant_mm_op>(
            mgr->create_app(), 2 * DI, D, t_.w_upgate_offset,
            std::vector<dequant_mm_pass>{
                {0, 2, dequant_mm_act_none},   // up
                {1, 2, dequant_mm_act_silu},   // gate
            }, rtp);
        down_ = std::make_unique<dequant_mm_op>(mgr->create_app(), D, DI,
                                                t_.w_down_offset,
                                                dequant_mm_act_none, rtp);
    }

    const attn_layout& layout() const { return t_; }

    /// q|k|v|z's fused output width: DQ + DK + DV + DQ.
    int QKVZ_OUT() const { return t_.qkv_out + t_.DQ; }

    /// q, k, v and the output gate in one dispatch, scattered on readback.
    ///
    /// The device returns [L][DQ + DK + DV + DQ] in the staging order -- q,
    /// then k, then v, then z -- while `cpu_ref::attention_layer` wants
    /// `[q | gate]` CONTIGUOUS, which is what `q4nx_matmul(w_.q, ...)` produces
    /// from the checkpoint's 2*DQ-row q_proj. So q comes off the head of the
    /// row and z off the tail, and the scatter is four memcpys per row. See the
    /// file header on why this is not the draft head's three.
    ///
    /// **The scatter is driven a tile at a time, off the bf16 y BO**, rather
    /// than widening the whole `[L][14336]` into an fp32 staging vector and
    /// memcpy-ing out of that. The fp32 intermediate was a pure round trip --
    /// written once by the widen, read once by the memcpy, never anything else
    /// -- so folding the widen into the scatter is the same arithmetic in one
    /// pass instead of two. `delta_proj::qkv_z_to_bo` has always been written
    /// this way; this is the same idiom.
    ///
    /// \param xn  [L][D] post-input_layernorm
    /// \param kk  [L][DK], vv [L][DV], qg [L][2*DQ] -- all resized here
    void qkv_z(const float* xn, int L, buffer<u8>& w,
               std::vector<float>& kk, std::vector<float>& vv,
               std::vector<float>& qg) {
        const size_t W  = (size_t)QKVZ_OUT();
        const size_t DQ = (size_t)t_.DQ, DK = (size_t)t_.DK, DV = (size_t)t_.DV;

        PROF8_Z0();
        kk.resize((size_t)L * DK);
        vv.resize((size_t)L * DV);
        qg.resize((size_t)L * 2 * DQ);
        PROF8_Z1();

        qkvz_->run_tiles(xn, L, w, [&](const bf16* y, int rows, int t) {
            PROF8_S0();
#pragma omp parallel for schedule(static) if ((size_t)rows * W >= host_par_min())
            for (int r = 0; r < rows; r++) {
                const bf16* src = y + (size_t)r * W;
                const size_t row = (size_t)(t + r);
                float* qgr = qg.data() + row * 2 * DQ;
                float* kr  = kk.data() + row * DK;
                float* vr  = vv.data() + row * DV;
                q38simd::bf16_to_f32(qgr,      src,                     DQ);
                q38simd::bf16_to_f32(kr,       src + DQ,                DK);
                q38simd::bf16_to_f32(vr,       src + DQ + DK,           DV);
                q38simd::bf16_to_f32(qgr + DQ, src + DQ + DK + DV,      DQ);
            }
            PROF8_S1();
        });
    }

    /// o_proj. \param attn [L][DQ] gated \param h1 [L][D]
    /// \param resid [L][D] added during the widen, or null for no residual
    void o(const float* attn, int L, buffer<u8>& w, float* h1,
           const float* resid = nullptr) {
        o_->run(attn, L, w, h1, resid);
    }

    /// The whole MLP up to down_proj: `g = silu(gate) * up`, one dispatch.
    ///
    /// The weight alternates 512 output rows of up with 512 of gate, so the
    /// device's y columns do too: block 2p is up's intermediate columns
    /// [p*512, (p+1)*512) and block 2p+1 is gate's. The kernel silu's the odd
    /// blocks (see the constructor), so the only thing left is the product, and
    /// that folds into the un-weave -- one multiply per element in a loop that
    /// was already reading both bands. So the host never materialises up or
    /// gate at all: no two `L x 17408` fp32 buffers and no second pass over
    /// them. At L = 512 each of the two would have been 34 MB.
    ///
    /// The silu is therefore the device's bf16 one, `x*(tanh(x/2)+1)/2` with
    /// three truncating roundings, where `cpu_ref::attention_layer::swiglu`
    /// computes it in fp32. `swiglu` remains the reference the driver scores
    /// this against, so the gap is measured rather than assumed.
    ///
    /// \param hn [L][D] post-post_attention_layernorm
    /// \param g  [L][intermediate_size] out, resized here
    void up_gate_swiglu(const float* hn, int L, buffer<u8>& w,
                        std::vector<float>& g) {
        const size_t DI = (size_t)t_.intermediate_size;
        const size_t W  = 2 * DI;
        constexpr size_t BAND = 512;          // the weave grain
        const size_t bands = DI / BAND;

        PROF8_Z0();
        g.resize((size_t)L * DI);
        PROF8_Z1();

        upgate_->run_tiles(hn, L, w, [&](const bf16* y, int rows, int t) {
            PROF8_S0();
            for (int r = 0; r < rows; r++) {
                const bf16* yr = y + (size_t)r * W;
                float* gr = g.data() + (size_t)(t + r) * DI;
                for (size_t p = 0; p < bands; p++) {
                    const bf16* u_src = yr + (2 * p)     * BAND;
                    const bf16* g_src = yr + (2 * p + 1) * BAND;   // silu'd
                    q38simd::bf16_mul_to_f32(gr + p * BAND, g_src, u_src, BAND);
                }
            }
            PROF8_S1();
        });
    }

    /// down_proj. \param g [L][intermediate_size] post-SwiGLU
    /// \param y [L][D], residual NOT added
    void down(const float* g, int L, buffer<u8>& w, float* y) {
        down_->run(g, L, w, y);
    }

    /// The whole MLP, `y = (silu(gate) * up) . W_down^T + resid`, with the
    /// SwiGLU product never materialised in fp32.
    ///
    /// `up_gate_swiglu` followed by `down` is what this replaces, and the
    /// difference is only where the product lands: there, in a
    /// `[L][17408]` fp32 vector that `down`'s own staging immediately reads
    /// back and narrows to bf16; here, straight into `down`'s x BO as the bf16
    /// it was going to become. **It is bit-identical** -- the product is
    /// computed in fp32 from the same two bf16 operands and rounded once, in
    /// both versions -- and it is the single largest buffer on the prefill
    /// path: at L = 1024 the vector alone is 71 MB written and 71 MB read back,
    /// per layer, 64 times per window.
    ///
    /// The two ops tile at the same `ROWS`, which is what lets one tile's
    /// up|gate output feed one tile of down with nothing in between. They are
    /// distinct `dequant_mm_op`s, so `upgate_`'s y BO and `down_`'s x BO are
    /// distinct allocations and the loop below is not aliasing them.
    ///
    /// `up_gate_swiglu` and `down` stay, unchanged: the hardware driver scores
    /// each dispatch on its own and needs the fp32 boundary to do it.
    ///
    /// \param hn     [L][D] post-post_attention_layernorm
    /// \param resid  [L][D] added during the widen, or null
    /// \param y      [L][D] out
    void mlp(const float* hn, int L, buffer<u8>& w, const float* resid,
             float* y) {
        const size_t DI = (size_t)t_.intermediate_size;
        const size_t W  = 2 * DI;
        const size_t D  = (size_t)t_.D;
        const int    R  = dequant_mm_op::ROWS;
        constexpr size_t BAND = 512;          // the weave grain
        const size_t bands = DI / BAND;

        // The un-weave, shared by both arms below. Reads one up|gate tile and
        // writes all `rows * DI` of down's x BO.
        const auto unweave = [&](const bf16* u, bf16* gx, int rows) {
            PROF8_S0();
#pragma omp parallel for schedule(static) if ((size_t)rows * DI >= host_par_min())
            for (int r = 0; r < rows; r++) {
                const bf16* yr = u  + (size_t)r * W;
                bf16*       gr = gx + (size_t)r * DI;
                for (size_t p = 0; p < bands; p++) {
                    const bf16* u_src = yr + (2 * p)     * BAND;
                    const bf16* g_src = yr + (2 * p + 1) * BAND;   // silu'd
                    q38simd::bf16_mul_to_bf16(gr + p * BAND, g_src, u_src, BAND);
                }
            }
            PROF8_S1();
        };

        if (L <= R || !dequant_mm_op::pipeline_enabled()) {
            for (int t = 0; t < L; t += R) {
                const int rows = (L - t) < R ? (L - t) : R;
                const bf16* u = upgate_->run_tile(hn + (size_t)t * D, rows, w);
                unweave(u, down_->stage(rows), rows);
                const bf16* dy = down_->run_staged(rows, w);
                down_->_widen(dy, rows, y + (size_t)t * D,
                              resid ? resid + (size_t)t * D : nullptr);
            }
            return;
        }

        // Pipelined. `run_tiles` does not fit here: this loop has TWO
        // dispatches per tile and the second depends on the first through the
        // un-weave, so the device order is upgate(t), down(t), upgate(t+1),
        // down(t+1)... and the host work is slotted into three different gaps.
        //
        // Only `upgate_` gets a second slot. `down_`'s x is written and
        // dispatched and read inside one iteration, with its `finish()` ahead
        // of the next `stage()`, so a second slot would never be live -- and
        // it is the 23 MB pair, worth not allocating.
        //
        // **The un-weave stays exposed, and that is a true dependency, not an
        // oversight**: down(t) reads what it writes, so the only thing that
        // could cover it is another tile's dispatch, and reaching one would
        // mean running upgate(t+1) before down(t) -- a three-deep pipeline for
        // the smallest of the three host terms. Staging and the widen are the
        // two that hide.
        upgate_->_ensure_pipelined();
        const int first = R < L ? R : L;
        upgate_->_stage_rows(hn, first);
        upgate_->dispatch_staged(first, w);

        for (int t = 0; t < L; t += R) {
            const int rows = (L - t) < R ? (L - t) : R;
            const int nt   = t + R;
            const int next = nt < L ? ((L - nt) < R ? (L - nt) : R) : 0;

            const bf16* u = upgate_->finish();          // upgate(t) lands
            unweave(u, down_->stage(rows), rows);       // device idle: the dep
            down_->dispatch_staged(rows, w);            // device: down(t)

            // upgate(t+1)'s x staging, under down(t).
            if (next) upgate_->_stage_rows(hn + (size_t)nt * D, next);

            const bf16* dy = down_->finish();           // down(t) lands
            if (next) upgate_->dispatch_staged(next, w);  // device: upgate(t+1)

            // down(t)'s widen, under upgate(t+1).
            down_->_widen(dy, rows, y + (size_t)t * D,
                          resid ? resid + (size_t)t * D : nullptr);
        }
    }

    // For the hardware driver, which scores each dispatch on its own.
    dequant_mm_op& qkvz_op()   { return *qkvz_; }
    dequant_mm_op& o_op()      { return *o_; }
    dequant_mm_op& upgate_op() { return *upgate_; }
    dequant_mm_op& down_op()   { return *down_; }

private:
    /// Everything the four dispatches assume about the layout, in one place.
    ///
    /// Each of these is silent if violated. A ragged shape hangs the dispatch
    /// (proj_sequence.cpp throws first); a Din that is not a whole number of
    /// 256-wide blocks, or a row count that is not a multiple of 64, makes
    /// `reorder_cpy` fold the buffer's head back in as padding (q4nx.hpp line
    /// 79) -- no assertion, just wrong weights.
    void _check_geometry() const {
        auto need = [](bool ok, const std::string& what) {
            if (!ok) throw std::runtime_error("prefill_proj: " + what);
        };
        const int D = t_.D, DI = t_.intermediate_size;

        // Dispatch 1 reads q|k|v|z as one matrix from byte 0, which requires z
        // to follow q|k|v with no gap and o to follow z. attn_layout::build and
        // stage_attn_proj_weights both establish this; it is repeated because
        // those guard the *staging* and this guards the *dispatch*, and the two
        // could drift apart.
        need(t_.w_z_offset == q4k_bytes((size_t)t_.qkv_out * (size_t)D),
             "z does not immediately follow q|k|v -- the fused qkv_z dispatch "
             "would read across a gap");
        need(t_.w_o_offset ==
                 t_.w_z_offset + q4k_bytes((size_t)t_.DQ * (size_t)D),
             "o does not immediately follow z");

        const int Douts[4] = {t_.qkv_out + t_.DQ, D, 2 * DI, D};
        const int Dins[4]  = {D, t_.DQ, D, DI};
        const char* names[4] = {"q|k|v|z", "o", "up|gate", "down"};
        for (int i = 0; i < 4; i++) {
            need(Douts[i] % dequant_mm_tiles::PASS_COLS == 0,
                 std::string(names[i]) + "'s Dout " + std::to_string(Douts[i]) +
                 " is not a whole array pass (512)");
            need(Dins[i] % dequant_mm_tiles::K_TILE == 0,
                 std::string(names[i]) + "'s Din " + std::to_string(Dins[i]) +
                 " is not a whole reduction tile (512)");
            need(Dins[i] % 256 == 0,
                 std::string(names[i]) + "'s Din is not a whole number of "
                 "q4_k column blocks");
        }

        // Every constituent sub-tensor, not just the fused shapes: the
        // 64-row-locality argument is what makes per-tensor reordering equal
        // concatenated reordering, and it is per *tensor*. 2*DI passes for the
        // fused up|gate but says nothing about up and gate individually.
        const int rows[7] = {t_.DQ, t_.DK, t_.DV, t_.DQ, D, DI, D};
        const char* rn[7] = {"q", "k", "v", "z", "o", "up/gate", "down"};
        constexpr int LOCALITY =
            dequant_mm_tiles::N_TILE;   // 64 = vertical_blocks * 32
        for (int i = 0; i < 7; i++)
            need(rows[i] % LOCALITY == 0,
                 std::string(rn[i]) + " is " + std::to_string(rows[i]) +
                 " rows, not a multiple of " + std::to_string(LOCALITY) +
                 " -- reorder_cpy's permutation would cross the tensor's end");

        // The weave grain, which up_gate_swiglu()'s un-weaving assumes.
        need(DI % 512 == 0,
             "intermediate_size is not a whole number of 512-row weave bands");
    }

    attn_layout t_;
    std::unique_ptr<dequant_mm_op> qkvz_, o_, upgate_, down_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_PREFILL_PROJ_HPP__
