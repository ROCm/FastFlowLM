// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/speculate_npu.hpp
/// \brief NPU backend for module 2.
///
/// The xclbins replace the head's fused layer, not the draft loop: the loop
/// below is speculate_cpu::draft() with `mtp_.step()` swapped for hardware, and
/// it has to stay a mirror of it -- speculate_cpu is the reference this is
/// scored against.
///
/// **A draft is seven steps of two different shapes**, and there is a bitstream
/// for each:
///
///   step 0     a window of T tokens -- the catch-up that brings the head's
///              cache level with the base stack. A prefill, by every measure
///              that matters: length L, block-causal mask, one pass. Its
///              attention core runs on `attn.xclbin` (gated_attention_prefill)
///              in ONE dispatch and its five projections on `dequant_mm.xclbin`,
///              with the CPU reference's own elementwise code between them.
///   steps 1-6  one token each. `MTP_layer.xclbin`, one dispatch per step,
///              unchanged.
///
/// Before the split, step 0 was T sequential dispatches of the decode
/// bitstream. That bitstream streams ~250 MB of head weights per dispatch, so
/// its cost is set by the weights and not by the tokens: a T-row window paid
/// the weight stream T times to move T rows. T is the whole prompt on the first
/// cycle after a prefill, so it is routinely in the hundreds -- up to the 4096
/// the runtime chunks at.
///
/// That last sentence was false for most of this file's life, and the reason is
/// worth stating where someone will read it. The app runs a `forward()` between
/// `prefill()` and the first `speculate()`; decode is a one-row prefill, so it
/// left `last_window_len = 1`, and `speculate_inner`'s clamp cut the head's
/// 350-row catch-up to one row. Live step-0 widths were 1-4 and BOTH floors
/// below were missed, so this whole path was dead on the live engine while
/// passing every driver -- the drivers call `prefill()` then `speculate()` with
/// nothing in between. `Impl::mtp_hist` now keeps the prompt's normed hidden
/// states past that `forward()`, which is what makes T large again.
///
/// The first version of the split moved only the attention core and left the
/// *projections*, much the larger share of the FLOPs, on the host. Those now go
/// to `dequant_mm.xclbin`, which fuses the Q4_K dequant into the GEMM: five
/// dispatches per 512-row tile, reading `npu_mtp_layer`'s existing weight BO at
/// five byte offsets with no second copy and no second reorder pass.
///
/// The projections are unconditional now; FLM_Q38_MTP_PROJ, which put them back
/// on the host, is gone. FLM_Q38_MTP_PREFILL=0 remains -- it is not a host arm,
/// it withholds the wide attention core and takes step 0 down the per-row
/// decode path, which is hardware either way. It stays because the trade is
/// genuinely width-dependent: see `_prefill_min_width`.
///
/// **The lm_head has its own bitstream too**, `lm_head.xclbin`, and at L = 1
/// -- which every step of the draft loop is -- every projection goes to it. The
/// object is `q38npu::npu_lm_head`, owned by the engine and borrowed here: the
/// checkpoint has no `mtp.lm_head`, so the base model and the draft head share
/// one 1.35 GB table and must not get two copies of it on the device.
///
/// It is required now, but it was never faster here -- know what this costs
/// the draft loop specifically. Both engines are at the DRAM wall
/// on this table (25.3 ms on the device against 25.9 on 32 host threads, i.e.
/// parity), and a draft step that takes it makes two NPU dispatches instead of
/// one; the pair serialises, 40.5 ms/step against 35.8. That is 13% of a draft
/// step, but drafting is only 7-8% of cycle time, so it is ~1% of decode --
/// inside run-to-run spread. What it buys is that 32 cores go idle during a
/// draft instead of saturated, which is the precondition for overlapping the
/// draft with host work. The drafted ids and the acceptance rate were identical
/// either way, measured against `cpu_ref::lm_head` directly before the
/// switch was removed.
///
/// Two pieces still stay on the host because they are in no xclbin: the
/// embedding lookup, and `rms(h2, model.mtp.norm.weight)` -- and the norm only
/// because the *next* step consumes its output as a hidden state. The device
/// does its own norm from the pre-norm row.
#pragma once

#include "speculate/speculate_module.hpp"
#include "speculate/mtp_head.hpp"
#include "speculate/npu/mtp_layer.hpp"
#include "speculate/npu/mtp_proj.hpp"
#include "prefill/npu/prefill_attn_op.hpp"

#include "common/cpu_ref_common.hpp"
#include "common/embedding.hpp"
#include "common/lm_head.hpp"
#include "common/npu/lm_head_op.hpp"
#include "common/rope.hpp"
#include "qwen3_8mtp_model_def.hpp"

#include "npu_utils/npu_utils.hpp"
#include "utils/debug_utils.hpp"
#include "utils/utils.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {

class speculate_npu final : public speculate_module {
public:
    /// Registers a SECOND xclbin. npu_xclbin_manager dedupes by name and
    /// otherwise gives this one its own hw_context, which is what we want: the
    /// draft head dispatches standalone, never inside an attention runlist, so
    /// the two bitstreams never need to share a context.
    ///
    /// \param prefill_xclbin_path  attn.xclbin, staged as
    ///             `attention_prefill.xclbin`. Optional: when it is empty or
    ///             missing, `op_` stays null and step 0 falls back to the
    ///             per-row decode path -- which is still on the device, just at
    ///             one dispatch per row. That is the one remaining choice here
    ///             and both arms are hardware.
    /// \param proj_xclbin_path  dequant_mm.xclbin, required. Step 0's five
    ///             projections used to fall back to the host without it; they
    ///             do not any more. Only ever consulted from inside the prefill
    ///             path, so it costs nothing when `op_` is null.
    /// \param head_npu the engine's `lm_head.xclbin` operator, borrowed rather
    ///             than opened here -- one 1.35 GB table on the device, not
    ///             two. Required: there is no mtp.lm_head in the checkpoint and
    ///             mtp_use_dedicated_embeddings is false, so the draft shares
    ///             the base model's, and the base model's is on a bitstream.
    speculate_npu(const cpu_ref::safetensors& w, const qwen3_8mtp_desc* desc,
                  const cpu_ref::embedding_table* embed,
                  int max_len, npu_xclbin_manager* npu,
                  const std::string& xclbin_path,
                  const std::string& prefill_xclbin_path,
                  const std::string& proj_xclbin_path,
                  q38npu::npu_lm_head* head_npu)
        : desc_(desc), embed_(embed), head_npu_(head_npu),
          out_norm_(w.f32("model.mtp.norm.weight")) {
        if (!npu)   throw std::runtime_error("speculate_npu: no device");
        if (!embed_) throw std::runtime_error("speculate_npu: no embedding table");
        if (!head_npu_)
            throw std::runtime_error("speculate_npu: no NPU lm_head -- the draft "
                                     "head shares the base model's table and has "
                                     "no host projection");
        if (out_norm_.size() != (size_t)desc_->D)
            throw std::runtime_error("speculate_npu: model.mtp.norm.weight is " +
                                     std::to_string(out_norm_.size()) + " wide, D is " +
                                     std::to_string(desc_->D));
        // A mismatched head would not fault -- it would project the wrong D
        // columns and hand back plausible logits for the wrong distribution,
        // which reads as a quietly bad acceptance rate rather than as a bug.
        if (head_npu_->dim() != (int)desc_->D ||
            head_npu_->vocab() != (int)desc_->vocab_size)
            throw std::runtime_error(
                "speculate_npu: the NPU lm_head is " +
                std::to_string(head_npu_->vocab()) + " x " +
                std::to_string(head_npu_->dim()) + ", the model is " +
                std::to_string(desc_->vocab_size) + " x " +
                std::to_string(desc_->D));

        npu_app_manager* mgr = npu->register_xclbin(xclbin_path);
        layer_ = std::make_unique<q38npu::npu_mtp_layer>(
            mgr->create_app(), q38npu::mtp_layout::build(max_len), w,
            q38npu::mtp_rtp_addresses{});

        _open_prefill(w, npu, prefill_xclbin_path);
        _open_proj(npu, proj_xclbin_path);

        e_.resize((size_t)desc_->D);
        h2_.resize((size_t)desc_->D);
        norm_out_.resize((size_t)desc_->D);
    }

    int  max_steps() const override { return cpu_ref::mtp::MTP_STEPS; }
    int  cache_len() const override { return layer_->cache_len(); }
    void reset() override { layer_->reset(); }
    void zero_stale() override { layer_->zero_stale(); }
    void crop(int len) override { layer_->crop(len); }

    void set_max_length(int n) override {
        layer_->set_max_length(n);
        // The grow moves plane_rows() and therefore v_at(0). The op holds its
        // own copy of the layout and emits the plane hop from it, so a missed
        // adoption here puts every v read at the old base -- silently, because
        // a DDR read past a BO does not fault.
        if (op_) op_->set_layout(layer_->layout());
    }

    std::vector<int> draft(const std::vector<int>& ids, const float* prev_hidden,
                           int T, int pos0, int k) override {
        if ((int)ids.size() != T)
            throw std::runtime_error("speculate: ids and T disagree");

        const size_t D = (size_t)desc_->D;

        std::vector<int>     drafts;
        std::vector<int32_t> step_ids(ids.begin(), ids.end());
        std::vector<float>   hidden(prev_hidden, prev_hidden + (size_t)T * D);

        int pos   = pos0;
        int width = T;

        last_step0_us_ = 0;
        const time_utils::time_point t_step0 = time_utils::now();

        for (int s = 0; s < k; s++) {
            if (_use_prefill(width)) {
                // Step 0, the prefill shape: ONE attn.xclbin dispatch for the
                // whole window, with the CPU reference's own halves around it.
                _prefill_step(step_ids, hidden.data(), width, pos);
            }
            else {
                // The decode bitstream moves exactly one row per dispatch, so a
                // width-T window is T sequential dispatches. That is exact and
                // not an approximation of the wide CPU step: each dispatch
                // appends its own k|v before the next one replays the cache, so
                // row r attends to rows 0..r just as the block-causal mask
                // would have. At width 1 -- every step but the first -- it is
                // also the only shape there is.
                for (int r = 0; r < width; r++) {
                    embed_->embed_row(step_ids[(size_t)r], e_.data());
                    layer_->forward(e_.data(), hidden.data() + (size_t)r * D,
                                    pos + r, h2_.data());
                }
            }

            // One row per step, even on the wide priming step: the reference
            // projects norm_out[:, -1:, :] regardless of window width, and
            // h2_ already holds only the last row.
            //
            // The host norm runs either way. The NPU kernel norms the row
            // itself from `h2_` -- so it is handed the PRE-norm state -- but
            // the next step consumes `norm_out_` as its hidden input, and
            // taking that from the host keeps the MTP layer chain
            // bit-identical to the CPU path. 5120 flops to confine the
            // device's influence to the token choice alone.
            cpu_ref::rms_norm(h2_.data(), out_norm_.data(), (int)D, norm_out_.data());

            const bf16* lg  = head_npu_->run(h2_.data(), out_norm_.data());
            const int   tok = head_npu_->argmax(lg);
            drafts.push_back(tok);

            hidden.assign(norm_out_.begin(), norm_out_.end());
            step_ids.assign(1, (int32_t)tok);
            pos  += width;
            width = 1;

            // Taken at the bottom of the iteration so it covers the lm_head
            // and the argmax too: the priming step pays for those exactly like
            // any other step, and timing only the layer would leave the
            // remainder charged to a phase it did not run in.
            if (s == 0)
                last_step0_us_ =
                    (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                        time_utils::now() - t_step0).count();
        }
        return drafts;
    }

    /// For the hardware drivers; the engine never reaches past the interface.
    q38npu::npu_mtp_layer& layer() { return *layer_; }
    /// Null when attn.xclbin was not supplied or FLM_Q38_MTP_PREFILL=0.
    q38npu::mtp_prefill_attn_op* prefill_op() { return op_.get(); }
    /// Null only when dequant_mm.xclbin was not supplied.
    q38npu::mtp_proj* proj() { return proj_.get(); }
    /// Never null. Not owned; the engine's.
    q38npu::npu_lm_head* lm_head_op() { return head_npu_; }

private:
    // ---- step 0 on attn.xclbin ----------------------------------------------
    /// Open the prefill bitstream, if it was given one.
    ///
    /// `register_xclbin` dedupes by name, so when module 1 already registered
    /// the same file this shares its hw_context and only takes a second app --
    /// which it must, because the q and o staging BOs have to come from the app
    /// that dispatches them.
    void _open_prefill(const cpu_ref::safetensors& w, npu_xclbin_manager* npu,
                       const std::string& path) {
        if (path.empty()) return;   // the engine was not configured with one
        if (const char* e = std::getenv("FLM_Q38_MTP_PREFILL"))
            if (e[0] == '0') {
                DEBUG_BLOCK(1,
                    header_print("FLM", "FLM_Q38_MTP_PREFILL=0 -- MTP step 0 on "
                                        "the decode bitstream, one row per "
                                        "dispatch");
                );
                return;
            }
        // Announced at the same two levels module 1's builders use, and for the
        // same reason: a missing bitstream is now the only way work leaves the
        // device. This one is the mildest of them -- it degrades to a path that
        // is still correct and still on the NPU, so nothing downstream will
        // complain and the cost shows up only as dispatch count.
        if (!std::filesystem::exists(path)) {
            DEBUG_BLOCK(1,
                header_print("FLM", "no attn.xclbin (" + path + ") -- MTP step 0 "
                                    "on the decode bitstream, one row per "
                                    "dispatch");
            );
            return;
        }

        npu_app_manager* mgr = npu->register_xclbin(path);
        op_ = std::make_unique<q38npu::mtp_prefill_attn_op>(
            mgr->create_app(), layer_->layout(),
            q38npu::prefill_attn_rtp_addresses{});

        // The CPU halves of step 0, and the ONLY reason this object exists:
        // fuse / project_qkv_rope / gate_out_mlp, which are the reference's own
        // code rather than a copy of it. max_len 0 on purpose -- its cache_k_ /
        // cache_v_ are never allocated and never read, because attn_core() is
        // the one helper that touches them and it is the one attn.xclbin
        // replaces. The NPU cache BO is the only cache in this path. Do not
        // "fix" the 0: raising it would allocate 64 MB of dead memory, and
        // anything that routed back through step() on this object would then
        // corrupt a second cursor instead of throwing.
        cpu_half_ = std::make_unique<cpu_ref::mtp_head>(w, embed_, 0);

        DEBUG_BLOCK(2,
            header_print("FLM", "MTP step 0 on the NPU (attn.xclbin, one "
                                "dispatch per window from width " +
                                std::to_string(_prefill_min_width()) + ")");
        );
    }

    /// Read once, and read here so the startup line and the decision agree.
    ///
    /// **The default is 3, and it is a statement about cycles rather than a
    /// per-dispatch break-even.** Only cycle 1's step 0 is wide: the prompt's
    /// normed hidden states survive in `Impl::mtp_hist`, so the first
    /// `speculate()` catches the head up over the whole prompt and the prefill
    /// shape amortises one weight stream across hundreds of rows. From cycle 2
    /// the head is cropped to `C - 1` after every cycle, so step 0 is width 1 on
    /// a partial accept and width 2 on a full one -- and a floor of 3 keeps both
    /// of those on the decode bitstream, one `MTP_layer.xclbin` dispatch per
    /// row.
    ///
    /// That description is only true with the step-0 clamp lifted. Before it
    /// was, `last_window_len` was 1 by the first `speculate()` and the live
    /// widths were 1-4, which straddles this floor for no reason connected to
    /// the trade it encodes.
    ///
    /// The floor was 2, which put the width-2 case on the wide arm instead: one
    /// `attn.xclbin` dispatch plus a full host pass over the head's ~424 M
    /// parameters, against two 250 MB device weight streams. Close enough to
    /// break-even that it was already flagged as worth measuring, and it still
    /// is -- `FLM_Q38_MTP_PREFILL_MIN=2` restores it for exactly that A/B.
    static int _prefill_min_width() {
        static const int min_w = [] {
            const char* e = std::getenv("FLM_Q38_MTP_PREFILL_MIN");
            const int   v = e ? std::atoi(e) : 3;
            return v < 1 ? 1 : v;
        }();
        return min_w;
    }

    /// Whether this step takes the prefill shape.
    ///
    /// At width 1 there is nothing to amortise: the decode path is one dispatch
    /// and the hybrid path would be one dispatch plus a full host pass over the
    /// head's ~424 M parameters. At width 2 the two arms are within noise of
    /// each other, and the default floor sends it to the decode path so that
    /// every post-prefill step 0 takes one shape -- see `_prefill_min_width`.
    bool _use_prefill(int width) const {
        if (!op_ || !cpu_half_) return false;
        return width >= _prefill_min_width();
    }

    // ---- step 0's projections on dequant_mm.xclbin --------------------------
    /// Open the fused dequant + GEMM bitstream, if it was given one.
    ///
    /// Five apps, one per projection, because the instruction sequence is baked
    /// from the shape and the weight offset and an app owns exactly one. They
    /// all share this xclbin's hw_context; `register_xclbin` dedupes by name
    /// like everywhere else.
    ///
    /// No layout adoption on `set_max_length`, unlike `op_`: the five weight
    /// offsets `mtp_layout::build` computes are functions of the tensor shapes
    /// alone. Only the cache geometry moves with max_len, and this object never
    /// touches the cache.
    void _open_proj(npu_xclbin_manager* npu, const std::string& path) {
        if (path.empty()) return;
        if (!std::filesystem::exists(path))
            throw std::runtime_error("speculate_npu: no dequant_mm.xclbin (" +
                                     path + ") -- the draft head's step-0 "
                                     "projections have no host path");

        npu_app_manager* mgr = npu->register_xclbin(path);
        proj_ = std::make_unique<q38npu::mtp_proj>(
            mgr, layer_->layout(), q38npu::dequant_mm_rtp_addresses{});

        DEBUG_BLOCK(2,
            header_print("FLM", "the draft head's step-0 projections on the NPU "
                                "(dequant_mm.xclbin, 5 dispatches per 512-row "
                                "tile, from width " +
                                std::to_string(_proj_min_width()) + ")");
        );
    }

    /// The width at or above which step 0's projections go to the device.
    ///
    /// N6's check 15, ms for all five projections, measured on this tree:
    ///
    ///     T          1      221     1024     4096
    ///     NPU     51.2 ms  54.6 ms  130 ms   530 ms
    ///     CPU     10.5 ms  621.3 ms 2450 ms     --
    ///             0.21x    11.4x    18.9x
    ///
    /// The device is FLAT to ~52 ms below one 512-row tile, because five
    /// dispatches each stream their whole weight region regardless of how few
    /// rows ride along; the host is linear, and a two-point fit over T = 1 and
    /// T = 221 gives 2.78 ms/row from a 7.7 ms base. Solving
    /// `7.7 + 2.78 w = 52` puts the crossing at **w = 16**.
    ///
    /// The default is now **1**: the projections follow `_use_prefill`, so a
    /// step that takes the prefill shape takes it whole and no matmul in it
    /// runs on the host.
    ///
    /// It was 25, and the reasoning was sound for the tree it was written in.
    /// The crossing is at w = 16; 25 bought margin against the flat leg's
    /// 51-55 ms jitter; and the region that actually mattered was w = 1-4,
    /// because the clamp in `speculate_inner` cut every live catch-up to
    /// `last_window_len`. All of that was a way of coping with a step 0 that
    /// never got to be wide.
    ///
    /// **That clamp is gone.** `_prefill_with_mm` now captures the prompt's
    /// normed hidden states into `Impl::mtp_hist`, so the first cycle's step 0
    /// is the full prompt -- hundreds of rows, where the table above reads
    /// 11-19x in the device's favour -- and `crop()` then holds the steady
    /// state at width 1-2, below `_prefill_min_width` and therefore not this
    /// decision's business at all. The interval [3, 24] the old floor covered
    /// is now unreachable on the live path in either direction.
    ///
    /// `FLM_Q38_MTP_PROJ_MIN` still overrides it, and that is the only way back
    /// to the split arm -- the host body stays reachable so the device body
    /// can be scored against it. Read once, like
    /// `_prefill_min_width`, so the startup line and the decision cannot
    /// disagree.
    static int _proj_min_width() {
        static const int min_w = [] {
            const char* e = std::getenv("FLM_Q38_MTP_PROJ_MIN");
            const int   v = e ? std::atoi(e) : 1;
            return v < 1 ? 1 : v;
        }();
        return min_w;
    }

    /// Whether this step's projections go to the device.
    ///
    /// Gated separately from `_use_prefill` but only reachable underneath it:
    /// the projections replace the CPU halves of the prefill path, so there is
    /// no configuration where they run and the attention core does not. Both
    /// arms keep the attention core on `attn.xclbin` either way -- this chooses
    /// only where the seven matmuls run.
    ///
    /// **The floor is effectively gone**, and this is the third time it has
    /// moved, so the history is worth keeping. It was 32; then removed on the
    /// reasoning that `dequant_mm` tiles at 512 rows and zero-pads the tail, so
    /// any width is legal and there is nothing to fall back *for* -- correct,
    /// and beside the point, because it was always a throughput guard and not a
    /// correctness one. Then 25, because live widths were 1-4 and the device
    /// leg is flat at ~52 ms, so every cycle was paying a prompt-sized dispatch
    /// to move four rows. Now 1, because those widths were an artefact of the
    /// step-0 clamp and the clamp has been lifted: `Impl::mtp_hist` keeps the
    /// prompt's normed hidden states alive across the `forward()` that used to
    /// overwrite them, so cycle 1 primes at the full prompt width and every
    /// later cycle is 1-2 rows and never reaches here.
    ///
    /// So a step that takes the prefill shape now takes it whole -- one
    /// `attn.xclbin` dispatch and five `dequant_mm.xclbin` dispatches per
    /// 512-row tile, with the host keeping only the norms, the RoPE, the output
    /// gate and the SwiGLU. There is no width at which the attention core goes
    /// to the device and the matmuls do not.
    bool _use_proj(int width) const {
        return proj_ != nullptr && width >= _proj_min_width();
    }

    /// One wide step: fusion, projections, the append, the attention core, the
    /// gate and the MLP. Leaves `h2_` holding the window's LAST row, which is
    /// the only one the caller projects -- same contract as the decode loop
    /// above, and the same one speculate_cpu keeps by taking norm_out[-1].
    ///
    /// Two bodies, one contract. They differ only in where the seven matmuls
    /// run; every norm, the RoPE, the gate and the SwiGLU are the same
    /// `cpu_ref::mtp_head` code in both. `FLM_Q38_MTP_PROJ=0` used to select
    /// between them and is gone -- `_use_proj()` now answers only the width
    /// question -- but the split stays, because scoring the device body
    /// against the host one needs a shared spine to make that a
    /// comparison rather than two implementations that happen to agree.
    void _prefill_step(const std::vector<int32_t>& ids, const float* hidden,
                       int width, int pos) {
        if (_use_proj(width)) _prefill_step_npu_proj(ids, hidden, width, pos);
        else                  _prefill_step_cpu_proj(ids, hidden, width, pos);
    }

    /// The projections on the host, the attention core on attn.xclbin.
    void _prefill_step_cpu_proj(const std::vector<int32_t>& ids,
                                const float* hidden, int width, int pos) {
        const size_t D = (size_t)desc_->D;

        // The same tables cpu_ref::mtp_head::step() would have built for this
        // window, through the same helper -- the angles are one formula either
        // way, but the rounding is not free to disagree.
        std::vector<float> cos, sin;
        cpu_ref::text_rope_tables(pos, width, cos, sin);

        cpu_ref::mtp_taps tp;
        cpu_half_->fuse(ids.data(), hidden, width, tp);

        std::vector<float> qg, kk, vv, qh;
        cpu_half_->project_qkv_rope(tp.fc_out.data(), width, cos.data(),
                                    sin.data(), qg, kk, vv, qh);

        // Read the cursor BEFORE appending: it is this window's L_begin, and
        // the attention core masks on absolute position. Handing it the
        // post-append value would let every query attend to its own future.
        const int base = layer_->cache_len();
        layer_->append_rows(kk.data(), vv.data(), width);

        // [width][DQ], and DQ (6144) is not D (5120) -- the gate and o_proj
        // downstream are what narrow it.
        attn_.assign((size_t)width * (size_t)op_->layout().DQ, 0.0f);
        op_->run(qh.data(), width, base, layer_->cache(), attn_.data());

        layer_out_.assign((size_t)width * D, 0.0f);
        cpu_half_->gate_out_mlp(qg.data(), attn_.data(), tp.fc_out.data(),
                                width, layer_out_.data());

        std::memcpy(h2_.data(), layer_out_.data() + (size_t)(width - 1) * D,
                    D * sizeof(float));
    }

    /// The same step with all seven matmuls on dequant_mm.xclbin.
    ///
    /// Six dispatches per 512-row tile, in the order the layer needs them; the
    /// CPU keeps every norm, the RoPE, the output gate and the SwiGLU, and each
    /// of those is the reference's own helper called by name. `gate_out_mlp` is
    /// not called here because it *contains* three of the matmuls -- this is
    /// its body, re-expressed over the finer helpers it was split into.
    ///
    /// The one thing that is NOT a rearrangement of the CPU body: every
    /// intermediate crosses the device as bf16, so `fc_out`, `attn`, `h1` and
    /// the MLP hidden all carry bf16 rounding the host path does not have. That
    /// is the whole of the numerical difference between the two branches.
    /// **Every buffer here is a member and every sizing is a `resize`.** They
    /// were locals sized with `assign(n, 0.0f)`, which cost a malloc, a memset
    /// and a free per call for each of ten vectors, all of them fully
    /// overwritten by the next statement -- `prefill_cpu` made the same change
    /// for the same reason, and the invariant that makes it legal is per call
    /// site: `dequant_mm_op::run` covers the whole `[L][Dout]`, the scatter
    /// covers every row, the un-weave covers every band, and the two
    /// `cpu_ref::mtp_head` stages write all of what they size. A future partial
    /// writer has to put the zeroing back, and nothing here will warn it: the
    /// symptom is stale rows from the previous step, which look like plausible
    /// activations rather than a fault.
    void _prefill_step_npu_proj(const std::vector<int32_t>& ids,
                                const float* hidden, int width, int pos) {
        const size_t D  = (size_t)desc_->D;
        const size_t DQ = (size_t)proj_->layout().DQ;
        buffer<u8>&  w  = layer_->weights();

        cpu_ref::text_rope_tables(pos, width, cos_, sin_);

        // ---- fusion ---------------------------------------------------------
        cpu_half_->fuse_pre(ids.data(), hidden, width, tp_, fc_in_);
        proj_->fc(fc_in_.data(), width, w, tp_.fc_out.data());

        // ---- q, k, v and the output gate ------------------------------------
        cpu_half_->in_norm(tp_.fc_out.data(), width, xn_);

        proj_->qkv_z(xn_.data(), width, w, kk_, vv_, qg_);
        cpu_half_->qkv_post_rope(qg_.data(), kk_.data(), width, cos_.data(),
                                 sin_.data(), qh_);

        // Read the cursor BEFORE appending -- this window's L_begin. Same
        // reason as the branch above and just as load-bearing.
        const int base = layer_->cache_len();
        layer_->append_rows(kk_.data(), vv_.data(), width);

        attn_.resize((size_t)width * DQ);
        op_->run(qh_.data(), width, base, layer_->cache(), attn_.data());

        // ---- the gate, o_proj and the attention residual ---------------------
        cpu_half_->apply_gate(qg_.data(), attn_.data(), width);

        // The residual rides in on `o`'s widen rather than in a second pass
        // over `[width][5120]`: it is a read-modify-write over a buffer that
        // loop is already writing, and the sum is fp32 either way, so it is
        // bit-identical to the `h1[i] += tp.fc_out[i]` it replaces.
        h1_.resize((size_t)width * D);
        proj_->o(attn_.data(), width, w, h1_.data(), tp_.fc_out.data());

        // ---- the MLP and its residual ----------------------------------------
        cpu_half_->post_norm(h1_.data(), width, hn_);

        // up, gate, silu, the product AND down_proj, with nothing fp32 in
        // between: the kernel's fused activation runs on the gate bands, the
        // multiply folds into the un-weave, and the un-weave writes bf16
        // straight into down's x BO. So the step never holds an
        // intermediate_size-wide fp32 buffer at all -- at width 4096 the one it
        // used to need between `up_gate_swiglu` and `down` was 285 MB, written
        // once and read back once. `up_gate_swiglu` and `down` remain, because
        // scoring each dispatch on its own needs the fp32 boundary to do
        // it; only the live path takes the fused entry point.
        layer_out_.resize((size_t)width * D);
        proj_->mlp(hn_.data(), width, w, h1_.data(), layer_out_.data());

        std::memcpy(h2_.data(), layer_out_.data() + (size_t)(width - 1) * D,
                    D * sizeof(float));
    }

    const qwen3_8mtp_desc*          desc_;
    const cpu_ref::embedding_table* embed_;   ///< borrowed
    q38npu::npu_lm_head*            head_npu_;   ///< borrowed; never null
    std::vector<float>              out_norm_;   ///< mtp.norm, applied on the host

    std::unique_ptr<q38npu::npu_mtp_layer> layer_;

    // Step 0's pair. Both null together: without attn.xclbin there is no
    // prefill path and the CPU halves would have nothing to sit around.
    std::unique_ptr<q38npu::mtp_prefill_attn_op> op_;
    std::unique_ptr<cpu_ref::mtp_head>           cpu_half_;   ///< max_len 0

    /// Step 0's projections. Independent of the pair above, and only reachable
    /// from inside the prefill path. Shares `layer_`'s weight BO.
    std::unique_ptr<q38npu::mtp_proj>            proj_;

    std::vector<float> e_, h2_, norm_out_;
    std::vector<float> attn_, layer_out_;   ///< step 0's wide scratch

    /// The rest of `_prefill_step_npu_proj`'s scratch, members rather than
    /// locals so a step-0 window does not malloc, memset and free ten vectors
    /// per call. Not used by `_prefill_step_cpu_proj`, which is the scored
    /// reference arm and is left as it was.
    std::vector<float>  cos_, sin_, fc_in_, xn_, kk_, vv_, qg_, qh_, h1_, hn_;
    cpu_ref::mtp_taps   tp_;
};

} // namespace qwen38
