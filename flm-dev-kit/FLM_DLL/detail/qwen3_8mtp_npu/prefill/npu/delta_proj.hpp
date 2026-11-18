// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/npu/delta_proj.hpp
/// \brief A gated-delta-net layer's prefill projections on `dequant_mm.xclbin`.
///
/// Four `dequant_mm_op`s over one weight BO at four byte offsets -- the sibling
/// of `prefill/npu/prefill_proj.hpp`, over `delta_layout` instead of
/// `attn_layout`. Six matmuls in four dispatches, because the woven `up|gate`
/// is one matrix and the adjacent `qkv|z` another.
///
/// One `delta_proj` serves all 48 layers: the weight BO is a `run()` argument,
/// not a member, so what is per-layer is the buffer (`npu_delta_weights`) and
/// what is per-shape is the app.
///
/// ## What this does NOT move
///
/// The recurrence: the depthwise conv, the per-head gates and the delta rule
/// itself all stay in `cpu_ref::delta_net_layer`, called by name. That is the
/// requested split, and it is also what the arithmetic wants -- the six
/// matmuls are 382.8 M MACs per token per layer and the fold is 3.1 M, 0.8%.
/// Moving 99.2% of the work and leaving a sequential scan on the host is the
/// whole design.
///
/// The two dense bf16 gate projections (`alpha_w`, `beta_w`) stay on the host
/// as well: they are q8_0 in the checkpoint, `dequant_mm` is q4_k only, and at
/// [48][5120] each they are 0.06% of the layer.
///
/// ## The dispatch order is not the attention layer's
///
/// `up|gate` is at byte 0 here and `qkv|z` two regions in, because the BO
/// follows `qwen3_5vl_npu`'s linear weight order -- see delta_weights.hpp on
/// why. The ops read byte offsets, so the order costs nothing; what it buys is
/// that the staged BO is already what a delta-net bitstream would want.
///
/// **The dispatches still run in the layer's own order**, which is not the
/// BO's: `qkv|z` first, then `out`, then the MLP. A reader following the file
/// top to bottom sees the constructor in BO order and
/// `prefill_linear_block::forward` in execution order, and they disagree on
/// purpose.
///
/// ## The two shapes that are not D
///
/// `out` reduces over the gated delta row (Din = dv = 6144) and `down` over
/// the MLP hidden (Din = intermediate_size = 17408). Passing D for either
/// strides the wrong way through a matrix of exactly the same byte count, so
/// nothing asserts and the cores dequantise the wrong rows -- the same trap
/// `reorder_cpy`'s `col` parameter carries at staging time.
#ifndef __QWEN3_8MTP_NPU_DELTA_PROJ_HPP__
#define __QWEN3_8MTP_NPU_DELTA_PROJ_HPP__

#include "prefill/npu/delta_weights.hpp"    // q38npu::delta_layout
#include "speculate/npu/proj_op.hpp"        // q38npu::dequant_mm_op
#include "common/cpu_simd.hpp"              // the vectorised marshalling

#include "npu_utils/npu_utils.hpp"
#include "buffer.hpp"
#include "typedef.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q38npu {

// ==========================================================================
// one layer's weight BO
// ==========================================================================
/// A single delta-net layer's 211 MB of q4_k projections, staged at load.
///
/// Mirrors `npu_attn_layer`'s constructor and `qwen3_5vl_npu.cpp:73-80`: create
/// the BO from an app, fill it through `reorder_cpy` straight out of the mmap,
/// sync once. Nothing moves per dispatch afterwards.
///
/// **Only the projections.** `qwen3_5vl_npu` gives a linear layer three more
/// BOs -- the rms gains, the ssm block, and a `| conv window | S |` cache --
/// because its recurrence runs on the device. Here it runs on the host, so
/// those stay in `cpu_ref::delta_net_layer` and nothing is allocated that no
/// kernel reads. `save_state` / `load_state` are untouched by this class.
///
/// 48 of these is 10.66 GB resident. Legal: XRT's 4 GB limit is per-BO, not
/// aggregate. The alternative -- a rotating pool restaged per layer -- would
/// pay 10.6 GB of `reorder_cpy` on every prefill window, forever.
class npu_delta_weights {
public:
    /// \param app    an app over dequant_mm.xclbin. BOs are device-scoped, so
    ///               this need not be the app that ultimately dispatches --
    ///               `buffer<T>(*device, size)` binds to the device, not the
    ///               hardware context.
    /// \param t      the layout the dispatches will read this BO with
    /// \param q4nx   an open model.q4nx
    /// \param layer  layer index, matching cpu_ref::delta_net_weights::load
    npu_delta_weights(npu_app& app, const delta_layout& t,
                      const cpu_ref::safetensors& q4nx, int layer)
        : t_(t), layer_(layer) {
        // bo_bytes, not proj_bytes: the a/b tail past the five projections is
        // 0.3% more and it is what makes this BO serve the verify bitstream
        // too. No prefill offset moves -- see delta_weights.hpp.
        proj_ = app.create_bo_buffer<u8>(t_.bo_bytes);
        stage_delta_proj_weights(proj_.data(), q4nx, t_, layer);
        proj_.sync_to_device();
    }

    buffer<u8>&         weights()       { return proj_; }
    const delta_layout& layout()  const { return t_; }
    int                 layer()   const { return layer_; }

private:
    delta_layout t_;
    int          layer_;
    buffer<u8>   proj_;
};

// ==========================================================================
// the four dispatches
// ==========================================================================
class delta_proj {
public:
    /// \param mgr     an app manager over dequant_mm.xclbin. Four apps are
    ///                taken from it, one per dispatch: a sequence is per-shape
    ///                and an app owns exactly one.
    /// \param layout  the SAME layout the weight BOs were staged from
    /// \param rtp     the per-build core RTP addresses
    delta_proj(npu_app_manager* mgr, const delta_layout& layout,
               const dequant_mm_rtp_addresses& rtp = dequant_mm_rtp_addresses{})
        : t_(layout) {
        if (!mgr) throw std::runtime_error("delta_proj: no app manager");
        _check_geometry();

        const int D  = t_.D;
        const int DI = t_.intermediate_size;

        // In BO order. The weave alternates 512 output rows of up with 512 of
        // gate, and 512 is exactly PASS_COLS, so the weave lands on the
        // generator's `j` loop: even j is up, odd j is gate. Two passes
        // therefore put the kernel's fused silu on the gate bands and nothing
        // else. The RTP is not uniform across a dispatch -- it is an
        // instruction at a position in the sequence, re-read by the cores every
        // round. (memory: rtp-act-is-per-round-not-per-dispatch)
        upgate_ = std::make_unique<dequant_mm_op>(
            mgr->create_app(), 2 * DI, D, t_.w_upgate_offset,
            std::vector<dequant_mm_pass>{
                {0, 2, dequant_mm_act_none},   // up
                {1, 2, dequant_mm_act_silu},   // gate
            }, rtp);
        down_  = std::make_unique<dequant_mm_op>(mgr->create_app(), D, DI,
                                                 t_.w_down_offset,
                                                 dequant_mm_act_none, rtp);
        // qkv | z as one matrix. Legal only because z immediately follows qkv,
        // which delta_layout::build and stage_delta_proj_weights both
        // establish; _check_geometry() re-asserts it because this is the
        // dispatch that depends on it.
        qkvz_  = std::make_unique<dequant_mm_op>(mgr->create_app(), t_.QKVZ_OUT(), D,
                                                 t_.w_qkv_offset,
                                                 dequant_mm_act_none, rtp);
        out_   = std::make_unique<dequant_mm_op>(mgr->create_app(), D, t_.DV,
                                                 t_.w_out_offset,
                                                 dequant_mm_act_none, rtp);
    }

    const delta_layout& layout() const { return t_; }

    /// The fused qkv projection and the output gate, in one dispatch.
    ///
    /// The device returns [L][conv_dim + dv] = [q 2048 | k 2048 | v 6144 |
    /// z 6144]. `conv_silu` wants qkv as one contiguous [L][conv_dim] and
    /// `gated_rmsnorm` wants z as [L][dv], and q|k|v is already contiguous at
    /// the head of the row -- so **the scatter is two memcpys**, not the
    /// attention layer's four or the draft head's three. The q4nx checkpoint
    /// helps here: a delta layer keeps q, k and v in one `qkv_proj` tensor,
    /// where an attention layer splits them across three.
    ///
    /// \param xn   [L][D] post-input_layernorm
    /// \param qkv  [L][conv_dim] out, resized here -- conv_silu mutates it
    /// \param zz   [L][dv] out, resized here
    void qkv_z(const float* xn, int L, buffer<u8>& w,
               std::vector<float>& qkv, std::vector<float>& zz) {
        const size_t W = (size_t)t_.QKVZ_OUT();
        PROF8_Z0();
        fused_.resize((size_t)L * W);
        PROF8_Z1();
        qkvz_->run(xn, L, w, fused_.data());

        const size_t CD = (size_t)t_.conv_dim, DV = (size_t)t_.DV;
        PROF8_Z0();
        qkv.resize((size_t)L * CD);
        zz .resize((size_t)L * DV);
        PROF8_Z1();
        PROF8_S0();
        for (int r = 0; r < L; r++) {
            const float* src = fused_.data() + (size_t)r * W;
            std::memcpy(qkv.data() + (size_t)r * CD, src,      CD * sizeof(float));
            std::memcpy(zz .data() + (size_t)r * DV, src + CD, DV * sizeof(float));
        }
        PROF8_S1();
    }

    /// The same dispatch, but q|k|v goes straight into a device BO as bf16.
    ///
    /// The N10 path's next consumer is `conv.xclbin`, which wants exactly this
    /// band, bf16, row-major -- which is exactly what `run_tile` already
    /// returns. So the q|k|v half of the scatter becomes a memcpy of bf16 into
    /// the conv's x BO, and only z is converted to fp32 because only z has a
    /// host consumer (`gated_rmsnorm`). That drops an `L x 10240` fp32 buffer
    /// and two conversions per element that nothing downstream would have read.
    ///
    /// \param dst            the conv input BO, from `delta_net_op::qkv_in(L)`
    /// \param dst_row_offset first row to write == `delta_net_op::qkv_in_row()`,
    ///                       the conv's history rows sit below it
    /// \param zz             [L][dv] out, resized here
    void qkv_z_to_bo(const float* xn, int L, buffer<u8>& w,
                     buffer<bf16>& dst, size_t dst_row_offset,
                     std::vector<float>& zz) {
        const size_t W  = (size_t)t_.QKVZ_OUT();
        const size_t CD = (size_t)t_.conv_dim, DV = (size_t)t_.DV;
        if (dst.size() < (dst_row_offset + (size_t)L) * CD)
            throw std::runtime_error(
                "delta_proj::qkv_z_to_bo: destination holds " +
                std::to_string(dst.size()) + " elements, " +
                std::to_string((dst_row_offset + (size_t)L) * CD) +
                " needed -- qkv_in() was called with a different L");

        PROF8_Z0();
        zz.resize((size_t)L * DV);
        PROF8_Z1();
        qkvz_->run_tiles(xn, L, w, [&](const bf16* y, int rows, int t) {
            PROF8_S0();
#pragma omp parallel for schedule(static) if ((size_t)rows * W >= host_par_min())
            for (int r = 0; r < rows; r++) {
                const bf16* yr = y + (size_t)r * W;
                std::memcpy(dst.data() + (dst_row_offset + t + r) * CD, yr,
                            CD * sizeof(bf16));
                float* zr = zz.data() + (size_t)(t + r) * DV;
                q38simd::bf16_to_f32(zr, yr + CD, DV);
            }
            PROF8_S1();
        });
    }

    /// ssm_out_proj. \param core [L][dv] post-gated-rmsnorm
    /// \param h1 [L][D]
    /// \param resid [L][D] added during the widen, or null for no residual
    void out(const float* core, int L, buffer<u8>& w, float* h1,
             const float* resid = nullptr) {
        out_->run(core, L, w, h1, resid);
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
    /// them. At L = 512 each of the two would have been 34 MB, and a delta
    /// layer's intermediate_size is the same 17408 as an attention layer's.
    ///
    /// The silu is therefore the device's bf16 one, `x*(tanh(x/2)+1)/2` with
    /// three truncating roundings, where `cpu_ref::delta_net_layer::swiglu`
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
    /// The delta layer's twin of `prefill_proj::mlp` -- same shapes, same
    /// weave, same argument for why it is bit-identical to `up_gate_swiglu`
    /// followed by `down`. See that one; the only difference here is the
    /// weight BO and that there are 48 of these per window rather than 16, so
    /// the `[L][17408]` fp32 round trip it removes is paid three times as
    /// often.
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

        // Pipelined; see the twin in prefill/npu/prefill_proj.hpp for why
        // `run_tiles` does not fit a two-dispatch tile, why only `upgate_`
        // takes a second slot, and why the un-weave stays exposed.
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
    dequant_mm_op& upgate_op() { return *upgate_; }
    dequant_mm_op& down_op()   { return *down_; }
    dequant_mm_op& qkvz_op()   { return *qkvz_; }
    dequant_mm_op& out_op()    { return *out_; }

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
            if (!ok) throw std::runtime_error("delta_proj: " + what);
        };
        const int D = t_.D, DI = t_.intermediate_size;

        // The fused qkv|z dispatch reads one matrix across two staged regions.
        need(t_.w_z_offset ==
                 t_.w_qkv_offset + q4k_bytes((size_t)t_.conv_dim * (size_t)D),
             "z does not immediately follow qkv -- the fused qkv_z dispatch "
             "would read across a gap");
        need(t_.w_out_offset ==
                 t_.w_z_offset + q4k_bytes((size_t)t_.DV * (size_t)D),
             "ssm_out does not immediately follow z");

        const int Douts[4] = {2 * DI, D, t_.QKVZ_OUT(), D};
        const int Dins[4]  = {D, DI, D, t_.DV};
        const char* names[4] = {"up|gate", "down", "qkv|z", "ssm_out"};
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
        // fused up|gate but says nothing about up and gate individually, and
        // QKVZ_OUT passes but says nothing about q, k, v or z.
        const int rows[7] = {DI, D, t_.DQ, t_.DK, t_.DV, t_.DV, D};
        const char* rn[7] = {"up/gate", "down", "q", "k", "v", "z", "ssm_out"};
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

    delta_layout t_;
    std::unique_ptr<dequant_mm_op> upgate_, down_, qkvz_, out_;
    /// qkv|z's fused output, before the scatter. A member so a window does not
    /// reallocate 16384 floats per row every call.
    std::vector<float> fused_;
};

} // namespace q38npu

#endif  // __QWEN3_8MTP_NPU_DELTA_PROJ_HPP__
