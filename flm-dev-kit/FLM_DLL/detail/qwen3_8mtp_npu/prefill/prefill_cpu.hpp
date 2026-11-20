// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file prefill/prefill_cpu.hpp
/// \brief Module 1, the whole-window prefill.
///
/// The host drives; every matmul, the attention core, the conv1d and the delta
/// fold are on the device. What is left here is the norms, the RoPE, the two
/// gate projections and the residual adds -- see the two block structs for the
/// stage-by-stage split.
#pragma once

#include "prefill/prefill_module.hpp"

#include "common/layer_stack.hpp"
#include "common/rope.hpp"
#include "common/window.hpp"
#include "qwen3_8mtp_model_def.hpp"

#include "utils/utils.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>      // std::swap, for the per-layer hidden/output trade

// ---------------------------------------------------------------------------
// Per-stage profiling, behind -DQWEN3_8MTP_PREFILL_PROFILE (`make
// PREFILL_PROFILE=1`, and `make clean` first -- it is a -D flag with no make
// dependency). Zero cost when off.
//
// A stage line is `wall (device, host)`: the device term is the delta of
// `dequant_mm_prof`'s two dispatch counters across the stage, so a stage that
// issues no dispatch reports all of its time as host. That split is the whole
// point of the instrumentation -- it is what showed module 1 to be 50% host.
//
// `PREFILL8_PROF_INIT()` declares the cursor, so every function that TICKs
// needs one. It is a statement, not a declaration-with-side-effects: put it
// first in the function body.
// ---------------------------------------------------------------------------
#ifdef QWEN3_8MTP_PREFILL_PROFILE
#define PREFILL8_PROF_INIT()                                                   \
    time_utils::time_point __prof_t = time_utils::now();                       \
    double __prof_dev = q38npu::dequant_mm_prof::dispatch_us +                 \
                        q38npu::dequant_mm_prof::other_dispatch_us
#define PREFILL8_PROF_TICK(name)                                               \
    do {                                                                       \
        const time_utils::time_point __prof_now = time_utils::now();           \
        const double __prof_us =                                               \
            time_utils::duration_us(__prof_t, __prof_now).first;               \
        const double __prof_dev_now =                                          \
            q38npu::dequant_mm_prof::dispatch_us +                             \
            q38npu::dequant_mm_prof::other_dispatch_us;                        \
        const double __prof_d = __prof_dev_now - __prof_dev;                   \
        header_print("prefill-prof",                                           \
                     name << ": " << __prof_us / 1000.0 << " ms ("             \
                          << __prof_d / 1000.0 << " device, "                  \
                          << (__prof_us - __prof_d) / 1000.0 << " host)");     \
        __prof_t   = __prof_now;                                               \
        __prof_dev = __prof_dev_now;                                           \
    } while (0)
#else
#define PREFILL8_PROF_INIT()     do {} while (0)
#define PREFILL8_PROF_TICK(name) do {} while (0)
#endif

namespace qwen38 {

struct prefill_attn_block {
    window_buffers* bufs;
    layer_stack*    stack;

    prefill_attn_block(window_buffers* b, layer_stack* s) : bufs(b), stack(s) {}

    void forward(int idx, const float* x, int L, float* y) {
        PREFILL8_PROF_INIT();
        q38npu::prefill_attn_op*  op   = stack->prefill_attn();
        q38npu::prefill_proj*     proj = stack->prefill_proj();
        q38npu::npu_attn_layer*   dev  = stack->npu_attn(idx);
        cpu_ref::attention_layer* half = stack->attn_half(idx);

        const size_t D  = (size_t)cpu_ref::q38::MODEL_DIM;
        const size_t DQ = (size_t)cpu_ref::q38::DQ;
        buffer<u8>&  w  = dev->weights();

        // ---- q, k, v and the output gate ------------------------------------
        half->in_norm(x, L, xn_);

        proj->qkv_z(xn_.data(), L, w, kk_, vv_, qg_);

        half->qkv_post_rope(qg_.data(), kk_.data(), L, bufs->cos.data(),
                            bufs->sin.data(), qh_);

        // Read the cursor BEFORE appending: it is this window's L_begin, and
        // the attention core masks on absolute position. Handing it the
        // post-append value would let every query attend to its own future.
        const int base = dev->tokens_seen();
        // The rows have to be in the cache before the core reads them: query
        // t attends through key base + t, which is this window's own row t.
        dev->append_rows(kk_.data(), vv_.data(), L);
        PREFILL8_PROF_TICK("attn:append_rows");

        attn_.resize((size_t)L * DQ);          // run() writes every row
        op->run(qh_.data(), L, base, dev->cache(), attn_.data());

        // ---- the gate, o_proj and the attention residual ---------------------
        half->apply_gate(qg_.data(), attn_.data(), L);
        PREFILL8_PROF_TICK("attn:gate");

        h1_.resize((size_t)L * D);             // o() writes every row
        proj->o(attn_.data(), L, w, h1_.data(), /*resid=*/x);
        PREFILL8_PROF_TICK("attn:o");

        // ---- the MLP and its residual ----------------------------------------
        half->post_norm(h1_.data(), L, hn_);
        PREFILL8_PROF_TICK("attn:post_norm");

        // up, gate, silu, the product AND down_proj, with nothing fp32 in
        // between: the kernel's fused activation runs on the gate bands, the
        // multiply folds into the un-weave, and the un-weave writes straight
        // into down's x BO as bf16. So the window never holds an
        // intermediate_size-wide buffer at all -- at L = 512 each of the three
        // it would otherwise need is 34 MB -- and
        // `cpu_ref::attention_layer::swiglu` is purely the reference N8 scores
        // this against. The residual rides in on the widen of down's output.
        proj->mlp(hn_.data(), L, w, /*resid=*/h1_.data(), y);
        PREFILL8_PROF_TICK("attn:mlp");
    }

private:
    /// Every inter-stage value of the block, kept across layers and windows.
    ///
    /// Members rather than locals because there is one of these objects and it
    /// runs 16 times per window: as locals these were 16 malloc/free pairs per
    /// window of up to `L x intermediate_size` floats -- 34 MB for `g_` alone
    /// at L = 512 -- for buffers whose shape does not change from layer to
    /// layer. Each is sized by the stage that fills it, and every stage that
    /// sizes one writes all of it (see `dequant_mm_prof::zerofill_us`).
    ///
    /// There is no MLP hidden here: `prefill_proj::mlp` keeps the SwiGLU
    /// product in down's x BO, so the widest thing this block owns is `qg_`.
    std::vector<float> xn_, kk_, vv_, qg_, qh_, attn_, h1_, hn_;
};

/// The 48 gated-delta-net layers. Same shape; a delta-net block carries
/// position implicitly, so the rope tables are dropped.
///
/// The six q4_k matmuls run on `dequant_mm.xclbin` in four dispatches per
/// 512-row tile; the conv and the fold run on `conv.xclbin` and
/// `GateDeltaNet_prefill.xclbin`. What is left on the host is the two norms,
/// the two gate projections and the gated RMSNorm.
///
/// **The split is not symmetric, and it is not meant to be.** The six matmuls
/// are 382.8 M MACs per token per layer; the conv, the gates and the delta
/// rule together are 3.1 M, 0.8% of the block. Across 48 layers the
/// projections are 18.4 G MACs per token against the 16 attention layers'
/// 5.96 G -- so this, not N8, is where prefill's remaining time is. What stays
/// on the host is a sequential scan with a loop-carried dependency on a
/// [128][128] state per head, which is exactly the shape that does not want to
/// be a matmul.
struct prefill_linear_block {
    window_buffers* bufs;
    layer_stack*    stack;

    /// Rows of ONE conv + fold pass. Not a policy: the conv's x descriptor has
    /// a 6-bit iteration field over 128-token blocks, so `delta_net_op::run`
    /// cannot address more than `MAX_WINDOW` rows at a time. Rounded down to
    /// the memtile chunk so every pass but the last is a whole number of them.
    ///
    /// The recurrence tiles across this exactly: `run()` reads the layer's conv
    /// window and S at the top and writes them back at the bottom, so pass
    /// c + 1 resumes on what pass c left, which is what one long pass would
    /// have computed. It is the same argument the fold's own 1024-row chunking
    /// rests on, one level out.
    static constexpr int DN_WINDOW =
        q38npu::delta_net_op::MAX_WINDOW / q38npu::delta_net_op::MT_L *
        q38npu::delta_net_op::MT_L;                                  // 8064

    prefill_linear_block(window_buffers* b, layer_stack* s) : bufs(b), stack(s) {}

    /// One gated-delta-net layer, across three bitstreams.
    ///
    /// This is `cpu_ref::delta_net_layer::forward`'s body re-expressed over the
    /// named stages it was split into, with the six `q4nx_matmul` calls
    /// replaced by four dispatches and `conv_silu` + `delta_rule` replaced by
    /// `delta_net_op::run`. Every elementwise step that is left -- both norms,
    /// the gates, the gated RMSNorm, both residuals -- is the reference's own
    /// code called by name, so there is no second copy of the math to keep in
    /// step.
    ///
    /// **Every row of the window passes through the two mutating stages exactly
    /// once, in order, and `advance` runs exactly once for the window.** The
    /// fold has no inverse, so running a row twice or skipping one corrupts the
    /// layer's history with nothing to compare against. That is the whole
    /// contract this function has with the reference. Here it means one
    /// `delta_net_op::run` per DN_WINDOW pass, each covering a disjoint,
    /// consecutive slice -- `run()` advances `conv_state_` and `state_` by its
    /// own pass length through the layer's mutable accessors -- and `advance(L)`
    /// once, at the end. A double advance is silent.
    ///
    /// Every L is a legal L: `dequant_mm` tiles at 512 and zero-pads the tail,
    /// and a window longer than DN_WINDOW is tiled by the pass loop rather than
    /// diverted. The pieces this reads are required, not optional; see
    /// `prefill_cpu`'s constructor.
    ///
    /// The alpha/beta gate projections stay on the host: q8_0, which
    /// dequant_mm does not read, and 0.06% of the layer.
    ///
    /// The one thing that is NOT a rearrangement of the CPU body: every
    /// intermediate crosses the device as bf16, so `xn`, the qkv rows, `h1` and
    /// the MLP hidden all carry bf16 rounding the reference does not have. As
    /// in N8, that is the whole of the numerical difference -- and here it
    /// lands upstream of a recurrence, so expect it to compound across the
    /// window rather than stay flat.
    ///
    /// Two things reorder against the reference's order. Both gate stages move
    /// above the conv, because alpha and beta are dispatch inputs rather than
    /// something computed between the two host stages -- and with the pass loop
    /// they must be ready before it starts. They are pure, so the order is
    /// free. And `qkv_z_to_bo` writes the qkv band straight into the conv's
    /// input BO as bf16 instead of materialising an `L x 10240` fp32 buffer
    /// nothing would read.
    ///
    /// The q l2norm and the per-head k l2norm ride on the conv bitstream with
    /// the conv itself; the `1/sqrt(head_dim)` on q stays in the delta-net
    /// kernel, which is why the conv is driven in NORM_K mode for q. See
    /// prefill/npu/conv1d_sequence.hpp -- getting this wrong scales q by 1/128
    /// and no cosine notices.
    void forward(int idx, const float* x, int L, float* y) {
        q38npu::delta_proj*            proj = stack->delta_proj();
        q38npu::npu_delta_weights*     dw   = stack->delta_weights(idx);
        cpu_ref::delta_net_layer*      half = stack->delta_half(idx);
        q38npu::delta_net_op*          dn   = stack->delta_net();
        q38npu::npu_delta_ssm_weights* ssm  = stack->delta_ssm(idx);

        const size_t D = (size_t)half->geometry().model_dim;
        buffer<u8>&  w = dw->weights();
        PREFILL8_PROF_INIT();

        // ---- qkv, the output gate, and the host-side gate projections -------
        half->in_norm(x, L, xn_);
        PREFILL8_PROF_TICK("linear:in_norm");

        // Both gate stages are pure and whole-window, and on the device path
        // alpha and beta are dispatch *inputs* -- so they have to be ready
        // before the first conv pass, not computed between the two host stages.
        half->alpha_beta(xn_.data(), L, aa_, bb_);
        PREFILL8_PROF_TICK("linear:alpha_beta");

        half->gates(aa_.data(), bb_.data(), L, alpha_, beta_);
        PREFILL8_PROF_TICK("linear:gates");

        const size_t DV = (size_t)half->geometry().dv();
        const size_t NV = (size_t)half->geometry().v_heads;

        // ---- the projection and the recurrence, on three bitstreams ---------
        // One pass for any window up to DN_WINDOW, which is every window this
        // engine actually sees; the loop is what keeps a longer one on the
        // device instead of handing it back. The projection is re-driven per
        // pass because it writes straight into the conv's staging BO, and that
        // BO is sized by the pass.
        // The pass loop covers [0, L) with disjoint consecutive slices, so both
        // of these are written in full and neither needs zeroing.
        zz_  .resize((size_t)L * DV);
        core_.resize((size_t)L * DV);
        for (int t = 0; t < L; t += DN_WINDOW) {
            const int n = (L - t) < DN_WINDOW ? (L - t) : DN_WINDOW;

            proj->qkv_z_to_bo(xn_.data() + (size_t)t * D, n, w,
                              dn->qkv_in(n), (size_t)dn->qkv_in_row(), zz_c_);
            PREFILL8_PROF_TICK("linear:qkv_z");

            dn->run(half, n, ssm->conv_w(),
                    alpha_.data() + (size_t)t * NV,
                    beta_ .data() + (size_t)t * NV, core_c_);
            PREFILL8_PROF_TICK("linear:delta_net");

            std::memcpy(zz_  .data() + (size_t)t * DV, zz_c_  .data(),
                        (size_t)n * DV * sizeof(float));
            std::memcpy(core_.data() + (size_t)t * DV, core_c_.data(),
                        (size_t)n * DV * sizeof(float));
        }

        half->gated_rmsnorm(core_.data(), zz_.data(), L);
        PREFILL8_PROF_TICK("linear:gated_rmsnorm");

        // ---- ssm_out_proj and the mixer residual ----------------------------
        h1_.resize((size_t)L * D);             // out() writes every row
        proj->out(core_.data(), L, w, h1_.data(), /*resid=*/x);
        PREFILL8_PROF_TICK("linear:out");

        // ---- the MLP and its residual ---------------------------------------
        half->post_norm(h1_.data(), L, hn_);
        PREFILL8_PROF_TICK("linear:post_norm");

        // up, gate, silu, the product AND down_proj, with nothing fp32 in
        // between -- see `prefill_attn_block::forward`, and `delta_proj::mlp`
        // on why it is bit-identical to the two calls it replaces. It matters
        // three times as much here: 48 of these layers per window against 16.
        proj->mlp(hn_.data(), L, w, /*resid=*/h1_.data(), y);
        PREFILL8_PROF_TICK("linear:mlp");

        half->advance(L);
    }

private:
    /// As in `prefill_attn_block`: one object, 48 layers per window, so these
    /// were 48 malloc/free pairs per window for buffers of a fixed shape.
    /// `zz_c_` and `core_c_` are the DN_WINDOW pass's slices, which only the
    /// pass loop sees -- they are members for the same reason, since a window
    /// longer than 8064 rows runs the loop more than once.
    std::vector<float> xn_, aa_, bb_, alpha_, beta_;
    std::vector<float> zz_, core_, zz_c_, core_c_, h1_, hn_;
};

class prefill_cpu final : public prefill_module {
public:
    /// Throws if any piece the two blocks dispatch against is missing.
    ///
    /// Module 1 has one path per layer type and no host alternative, so a null
    /// here is a build or staging error and not a degrade. Checking all 64
    /// layers up front buys a message that names everything that is absent; the
    /// alternative is a null dereference partway through the first window,
    /// which says only that *something* was.
    ///
    /// It used to double as the guard against a host-only stack, back when
    /// `layer_stack::build` would skip every `_build_*` without a device and
    /// all five accessors came back null at once. That cannot happen now --
    /// `build()` throws without a device and every bitstream path is a required
    /// argument -- so what is left here catches a partially staged xclbin
    /// directory.
    prefill_cpu(const qwen3_8mtp_desc* desc, layer_stack* stack)
        : desc_(desc), stack_(stack),
          attn_block_(&bufs_, stack), linear_block_(&bufs_, stack) {
        std::string missing;
        auto need = [&missing](bool ok, const char* what) {
            if (!ok) { if (!missing.empty()) missing += ", "; missing += what; }
        };

        need(stack->prefill_attn() != nullptr, "attention_prefill.xclbin");
        need(stack->prefill_proj() != nullptr, "dequant_mm.xclbin (attention projections)");
        need(stack->delta_proj()   != nullptr, "dequant_mm.xclbin (delta-net projections)");
        need(stack->delta_net()    != nullptr, "conv.xclbin + GateDeltaNet_prefill.xclbin");

        // Per-layer, and reported as a count rather than a list: if one is
        // absent they all are, because each comes from a single staging loop.
        int no_dev = 0, no_half = 0, no_dw = 0, no_dhalf = 0, no_ssm = 0;
        for (uint32_t i = 0; i < desc->num_hidden_layers; i++) {
            const int li = (int)i;
            if ((*stack)[li].is_full_attention()) {
                if (!stack->npu_attn(li))  no_dev++;
                if (!stack->attn_half(li)) no_half++;
            }
            else {
                if (!stack->delta_weights(li)) no_dw++;
                if (!stack->delta_half(li))    no_dhalf++;
                if (!stack->delta_ssm(li))     no_ssm++;
            }
        }
        auto need_n = [&missing](int n, const char* what) {
            if (n) { if (!missing.empty()) missing += ", ";
                     missing += std::to_string(n) + " x " + what; }
        };
        need_n(no_dev,   "npu_attn_layer");
        need_n(no_half,  "cpu_ref::attention_layer companion");
        need_n(no_dw,    "npu_delta_weights BO");
        need_n(no_dhalf, "cpu_ref::delta_net_layer");
        need_n(no_ssm,   "npu_delta_ssm_weights BO");

        if (!missing.empty())
            throw std::runtime_error("prefill_cpu: module 1 has no host path; "
                                     "missing " + missing);

        // Whether the fp32 <-> bf16 marshalling around each dispatch runs on
        // one core or all of them. The loops are vectorised either way
        // (common/cpu_simd.hpp); this is the 9% on top, and the environment
        // decides it, not this build, so say which way it went -- the same
        // reason `vision tower loaded (... mm=npu ...)` prints.
        // See q38npu::host_par_min().
        // header_print("prefill", "host marshalling "
        //     << (q38npu::host_par_min() == (size_t)-1
        //             ? "serial (spinning OpenMP; OMP_WAIT_POLICY=passive or "
        //               "FLM_Q38_HOST_PAR=1 to thread it)"
        //             : "threaded"));
    }

    void begin_window(int L, const int32_t* pos_thw) override {
        bufs_.ensure(L, (int)desc_->D, (int)desc_->RoPE_DIM);
        cpu_ref::mrope_table(pos_thw, L, bufs_.cos.data(), bufs_.sin.data(),
                             /*round_to_bf16=*/true);
    }

    float* hidden() override { return bufs_.x.data(); }

    void set_taps(tap_buffer* taps) override { taps_ = taps; }

    void run(int L) override {
        if (taps_ && taps_->enabled()) taps_->begin(L);
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        q38npu::dequant_mm_prof::reset();
        const auto __w0 = time_utils::now();
#endif
        for (uint32_t i = 0; i < desc_->num_hidden_layers; i++)
            forward_layer((int)i, L);
#ifdef QWEN3_8MTP_PREFILL_PROFILE
        const double wall =
            time_utils::duration_us(__w0, time_utils::now()).first / 1000.0;
        // Where a dequant_mm tile's wall clock went, summed over all 13 + 6
        // projections x 64 layers. `dispatch` is the only term the device owns;
        // everything else is this host walking the same rows again.
        using P = q38npu::dequant_mm_prof;
        header_print("prefill-prof",
            "window L=" << L << ": " << wall << " ms total, "
            << P::tiles << " dequant_mm tiles -- dispatch "
            << P::dispatch_us / 1000.0 << " ms, x stage "
            << P::stage_us / 1000.0 << " ms, y read "
            << P::read_us / 1000.0 << " ms, y widen "
            << P::widen_us / 1000.0 << " ms, zero-fill "
            << P::zerofill_us / 1000.0 << " ms, scatter "
            << P::scatter_us / 1000.0 << " ms, other-dispatch "
            << P::other_dispatch_us / 1000.0 << " ms");
#endif
    }

    /// One layer, in place on the hidden buffer.
    void forward_layer(int layer_idx, int L) {
        PREFILL8_PROF_INIT();
        decoder_layer& layer = (*stack_)[layer_idx];
        float* x = bufs_.x.data();
        float* y = bufs_.y.data();

        if (layer.is_full_attention()) {
            attn_block_.forward(layer_idx, x, L, y);
            PREFILL8_PROF_TICK("layer:attn_block_total");
        }
        else {
            linear_block_.forward(layer_idx, x, L, y);
            PREFILL8_PROF_TICK("layer:linear_block_total");
        }

        // The reference layers read x on the residual path after writing y, so
        // they cannot write in place -- but they do write all L x D of y (the
        // residual rides in on `mlp`'s final widen, which covers every row), so
        // the two buffers can simply trade names. `std::swap` on two vectors
        // swaps three pointers; the copy it replaces was 21 MB per layer, 64
        // times per window.
        //
        // Swapping rather than alternating on layer parity is deliberate:
        // `hidden()` is `bufs_.x` unconditionally, so this stays correct for an
        // odd `num_hidden_layers` as well as the 64 this model has.
        std::swap(bufs_.x, bufs_.y);

        // The residual stream after this layer: DFlash2's tap, if the layer is one.
        if (taps_ && taps_->enabled()) taps_->capture(layer_idx, bufs_.x.data(), L);
    }

private:
    tap_buffer*            taps_ = nullptr;
    const qwen3_8mtp_desc* desc_;
    layer_stack*           stack_;
    window_buffers         bufs_;
    prefill_attn_block     attn_block_;
    prefill_linear_block   linear_block_;
};

} // namespace qwen38
