// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file qwen3_8mtp_npu.cpp
/// \brief Qwen3.8-27B engine: owns the layer stack and the three modules
///        (prefill, speculate, verify) and holds the policy that wires them.
#include <omp.h>

#include "qwen3_8mtp_npu_detail.hpp"

#include "image/image_cpu.hpp"
#include "image/npu/vision_op_factory.hpp"
#include "image/vision_ops.hpp"
#include "prefill/prefill_cpu.hpp"
#include "speculate/speculate_npu.hpp"
#include "verify/verify_cpu.hpp"

#include "metrices.hpp"
#include "utils/error_measure.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace {
/// time_utils::duration_us returns a float, which cannot represent whole
/// microseconds past ~17 s of accumulation. These counters are cumulative over
/// a session, so they integrate in uint64 and convert only at print time.
inline uint64_t _elapsed_us(time_utils::time_point t0) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               time_utils::now() - t0).count();
}

/// **The requested context turned into a KV cache buffer height.** The ONE
/// place that happens, so that `Impl::MAX_L`, `attn_layout::MAX_L`,
/// `mtp_layout::MAX_L` and the rows actually allocated are all the same number
/// -- the way `MAX_L` reads in qwen3_5vl_npu, where `get_kv_cache_size(MAX_L)`
/// is simply `MAX_L * (DK + DV)`.
///
/// Up to 8192 tokens, three things happen here, in order:
///
///   the floor     4096. Below it the caches are not worth the reallocations,
///                 and attn_layout's cap is far above it.
///   the power     round up to a power of two, so a grow doubles rather than
///                 creeping, and a resumed conversation lands on the same size
///                 whatever length it asked for.
///   the headroom  one PREFILL_CHUNK on top. **This is what the apron used to
///                 be**, and moving it here is the point of the change: a pass
///                 is legal only if the WHOLE of it fits (the verify path
///                 appends L_TOKENS rows however few are real, the prefill path
///                 pads its window up to PREFILL_CHUNK), and the layouts now
///                 refuse anything else rather than quietly reading off the end
///                 of the BO. The 128 rows are what keeps a caller who fills
///                 the context to the power of two from tripping those guards.
///
/// **Above 8192 the power of two would overshoot the cap** (FLM#1001): 16384
/// plus the headroom is 16512 rows, past attn_layout's 16384, so every length
/// between 8193 and 16384 failed to load. There the headroom goes on first and
/// the sum rounds up to a whole page (4096 rows) -- 12288 for anything up to
/// 12160 -- and the result is clamped to the cap. Near the cap that leaves less
/// than a PREFILL_CHUNK of headroom; _prefill_with_mm then runs the last rows
/// of a prompt through the verify pass instead, which needs only L_TOKENS of
/// it. _usable_context() is the length that still fits.
///
/// The result is a multiple of PREFILL_CHUNK, which is what `build()` asserts
/// -- a power of two at or above 4096 already is one, so the round is a
/// formality that survives someone changing the floor.
inline uint32_t _kv_buffer_rows(uint32_t requested) {
    constexpr uint32_t C    = (uint32_t)q38npu::attn_cache::PREFILL_CHUNK;
    constexpr uint32_t PAGE = 4096;
    constexpr uint32_t CAP  = (uint32_t)q38npu::attn_cache::max_context();
    uint32_t n = std::max<uint32_t>(requested, 4096);
    if (n > 8192) {
        n = (n + C + PAGE - 1) / PAGE * PAGE;
        return std::min(n, CAP);
    }
    const float log2_n = std::log2((float)n);
    if (log2_n != std::floor(log2_n))
        n = 1u << ((int)std::floor(log2_n) + 1);
    n += C;
    return (n + C - 1) / C * C;
}

/// **The longest context the engine can hold: 16377 tokens.** The buffer tops
/// out at attn_layout's 16384 rows, and the deepest verify pass -- a decode
/// step or the tail of a prompt -- still appends a whole L_TOKENS (8) rows, so
/// it has to start at 16376 at the latest. The prefill path's 128-row padding
/// is not the bound: _prefill_with_mm hands the rows it cannot pad to the
/// verify pass.
constexpr uint32_t _usable_context() {
    return (uint32_t)q38npu::attn_cache::max_context() -
           (uint32_t)q38npu::npu_lm_head_8::ROWS + 1;
}

/// A request past _usable_context() is lowered to it, with a warning, rather
/// than refused: the runtime asks for its --ctx-len before it can ask the
/// engine what fits, and reads the real bound back from max_context_length().
inline uint32_t _clamp_context(uint32_t requested) {
    if (requested <= _usable_context()) return requested;
    header_print("WARNING", "Qwen3.8 holds at most " + std::to_string(_usable_context()) +
                            " tokens; context length " + std::to_string(requested) +
                            " lowered to " + std::to_string(_usable_context()));
    return _usable_context();
}
} // namespace

// ===============================================
// construction
// ===============================================
qwen3_8mtp_npu::Impl::Impl(LM_Config config, npu_xclbin_manager* npu_instance,
                           qwen3_8mtp_npu* parent_ptr, int MAX_L)
    : config(config), npu(npu_instance), parent_ptr(parent_ptr) {
    current_context_length = 0;
    general_token_idx      = 0;

    desc.build(this->config);
    // Shapes only; the tower's 959 MB of weights wait for load_weights(). A
    // text-only checkpoint leaves this with present == false and nothing else
    // in the engine changes.
    vision_desc.build(this->config);

    this->MAX_L = _kv_buffer_rows(_clamp_context((uint32_t)MAX_L));

    // Sized before the weights arrive: the runtime samples this and only needs
    // the vocab width.
    logits = buffer<bf16>(desc.vocab_size);
    logits.memset((bf16)0);
}

qwen3_8mtp_npu::Impl::~Impl() = default;

// ===============================================
// weights
// ===============================================
void qwen3_8mtp_npu::Impl::load_weights(Q4NX& q4nx) {
    // model.q4nx is mmap'd directly rather than staged through Q4NX: staging
    // every projection into host buffers would materialise ~17 GB, while the
    // layers dequantise one 32x256 chunk at a time.
    (void)q4nx;

    weights = std::make_unique<cpu_ref::safetensors>(desc.q4nx_path);

    // **This engine requires a device.** Every operation below runs on the NPU
    // and none of them has a host alternative any more, so a null manager is
    // refused here rather than discovered three constructors down as a stack
    // with no prefill. It was always effectively true -- `prefill_cpu`'s
    // constructor has refused a deviceless stack since module 1 lost its host
    // arms -- but it used to be true by accident, at the bottom of the call
    // chain, with a message about a missing bitstream rather than a missing
    // card.
    if (!npu)
        throw std::runtime_error(
            "qwen3_8mtp_npu: no NPU. This engine has no host path -- the 64 "
            "decoder layers, both lm_heads, the draft head and the vision "
            "tower all run on the device.");

    // Every design in this model builds to layer.xclbin and they all land in
    // one flat per-model directory, so each is deployed under a distinct name;
    // this one is attention_layer.xclbin (see the staging list in the runtime's
    // test/qwen3_8mtp_npu/activate.sh). Resolved against config.exec_path,
    // never absolute -- the runtime ships xclbins/ next to the binary and
    // exec_path is how it says where.
    const std::string xclbin_dir =
        utils::path_join(config.exec_path, "xclbins", config.model_name);
    const std::string attn_xclbin =
        utils::path_join(xclbin_dir, "attention_layer.xclbin");

    // The prefill attention core, a second bitstream over the same cache.
    // Required, not optional: module 1 has no host path, so a missing file
    // throws in layer_stack::build rather than degrading. Staged as
    // attention_prefill.xclbin for the same reason as above -- every design in
    // this model builds to layer.xclbin.
    const std::string prefill_attn_xclbin =
        utils::path_join(xclbin_dir, "attention_prefill.xclbin");

    // Module 1's projections, all thirteen of them. dequant_mm is the only
    // design here that is not built from one named layer.xclbin, so it is
    // staged under its own name; the draft head resolves the same file below
    // and register_xclbin dedupes by name, so the two share one hw_context and
    // take separate apps from it. Required, like the core above.
    const std::string proj_xclbin =
        utils::path_join(xclbin_dir, "dequant_mm.xclbin");

    // The 48 delta-net layers' recurrence: the gated delta rule and the
    // depthwise conv1d that feeds it, one bitstream each. Both or neither --
    // the conv also applies the q/k l2norms the fold assumes its input already
    // carries. Staged under their own names, like dequant_mm and unlike the
    // two attention designs. Required, like everything else module 1 reads;
    // either one missing throws, named separately.
    const std::string deltanet_xclbin =
        utils::path_join(xclbin_dir, "GateDeltaNet_prefill.xclbin");
    const std::string conv_xclbin =
        utils::path_join(xclbin_dir, "conv.xclbin");

    // The 48 delta-net layers in the verify pass, whole: one dispatch per layer
    // per 8 rows, against the same projection BO the prefill phase staged.
    // Staged under its own name too -- it is built from deltanet_layer.xclbin
    // and is a different design from GateDeltaNet_prefill.xclbin above, which
    // runs only the fold and only at prefill shapes. Required like the rest.
    const std::string delta_verify_xclbin =
        utils::path_join(xclbin_dir, "deltanet_layer.xclbin");

    stack.build(*weights, desc, (int)MAX_L, npu, attn_xclbin,
                prefill_attn_xclbin, proj_xclbin, deltanet_xclbin, conv_xclbin,
                delta_verify_xclbin);

    embedding = std::make_unique<cpu_ref::embedding_table>(*weights);
    // Kept for ONE caller: module 3's shape guard, for a verify batch wider
    // than `npu_lm_head_8::ROWS`. Every projection the engine actually makes
    // today is on a bitstream -- see get_logits() and verify_cpu::run().
    head      = std::make_unique<cpu_ref::lm_head>(*weights);

    // The L = 1 projection, for the draft loop's seven steps and get_logits().
    // Verified equivalent when it was ported -- 8/8 argmax against
    // verify.logits, cos 0.99998 against its own Q8_0 bytes, and byte-identical
    // drafts and hit rate through the engine.
    //
    // Required, like every other bitstream here. It is worth recording that
    // this one is NOT a speedup: the table is 1.35 GB and BOTH engines are at
    // the DRAM wall on it, 25.3 ms on the device against 25.9 ms on 32 host
    // threads. What it buys is the 32 cores, for 26 ms per projection -- and,
    // now that the host arm is gone, one path to keep correct instead of two.
    //
    // The two callers pay differently and it is worth knowing which is which.
    // Prefill dispatches this ONCE, at the end of a 64-layer sweep that has
    // already made thousands of dispatches, so the cost is parity and the
    // cores come free. A draft step dispatches it as the SECOND of two, and
    // the pair serialises: ~4.9 ms/step on top (40.5 vs 35.8), which is 13% of
    // the step but only ~1% of a cycle, since drafting is 7-8% of cycle time.
    //
    // Module 3 is NOT routed here -- see get_logits(). It needs L = 8 and this
    // bitstream is L = 1; its backend is `lm_head8_npu`, built just below.
    {
        const std::string lm_head_xclbin =
            utils::path_join(xclbin_dir, "lm_head.xclbin");
        if (!std::filesystem::exists(lm_head_xclbin))
            throw std::runtime_error("no lm_head.xclbin (" + lm_head_xclbin +
                                     ") -- the L = 1 projection has no host "
                                     "path");
        npu_app_manager* mgr = npu->register_xclbin(lm_head_xclbin);
        lm_head_npu = std::make_unique<q38npu::npu_lm_head>(
            mgr->create_app(), *weights);
        DEBUG_BLOCK(2,
            header_print("FLM", "lm_head on the NPU (" +
                                std::to_string(lm_head_npu->vocab()) + " x " +
                                std::to_string(lm_head_npu->dim()) + ")");
        );
    }

    // Module 3's own bitstream, at the shape module 3 actually has: all k+1
    // rows in one dispatch, reading the 1.35 GB table once instead of once per
    // row. That is the whole of the win, and it is worth stating how big it
    // is, because the design's README measures against the wrong baseline.
    //
    // Eight L = 1 dispatches would be ~204 ms, and against that this is 5x.
    // But the engine's alternative is the HOST, and the host does not scale
    // per row either -- `q4nx_matmul` keeps its row loop inside its weight
    // tile loop, so the table streams once regardless of L. Measured at the
    // shipping flags: 19-25 ms for 1 row, 61-76 for 8, against 29 ms steady
    // state on the device. So the projection itself is ~32 ms cheaper.
    //
    // END TO END THAT IS NOT MEASURABLE, and it was measured rather than
    // assumed. tools/bench_verify_lm_head.cpp, 20 real cycles, three repeats
    // each way, reading the verify-phase counter (replay excluded):
    //
    //   verify ms   device  42810 43171 38073   mean 41351   spread 13%
    //               host    40684 39911 40999   mean 40531   spread  2.7%
    //
    // The device's fastest run beats every host run and the sign flips inside
    // its own three samples: 32 ms on a ~2000 ms verify phase is 1.6%, under
    // the noise floor. The move buys PARITY, not speed -- the same finding as
    // the draft head. What it does buy is bit-identical output and 32 host
    // threads freed for the rest of the cycle.
    //
    // Unlike the draft head this does NOT pay the second-dispatch tax: verify
    // dispatches once, at the end of a 64-layer sweep, exactly like prefill.
    //
    // The cost is a second 1.35 GB weight BO, because a BO belongs to the app
    // that dispatches it and this is a different xclbin. That is now
    // unconditional: both projections are on the device, so the resident
    // weight footprint is 2.7 GB of table and there is no switch to get it
    // back. Required like the rest -- a missing file throws.
    {
        const std::string lm_head8_xclbin =
            utils::path_join(xclbin_dir, "lm_head_8.xclbin");
        if (!std::filesystem::exists(lm_head8_xclbin))
            throw std::runtime_error("no lm_head_8.xclbin (" + lm_head8_xclbin +
                                     ") -- module 3's projection has no host "
                                     "path");
        npu_app_manager* mgr = npu->register_xclbin(lm_head8_xclbin);
        lm_head8_npu = std::make_unique<q38npu::npu_lm_head_8>(
            mgr->create_app(), *weights);
        DEBUG_BLOCK(2,
            header_print("FLM", "verify lm_head on the NPU (" +
                                std::to_string(lm_head8_npu->vocab()) + " x " +
                                std::to_string(lm_head8_npu->dim()) + ", " +
                                std::to_string(q38npu::npu_lm_head_8::ROWS) +
                                " rows/dispatch)");
        );
    }

    final_norm_w = weights->f32("model.norm.weight");
    if (final_norm_w.size() != desc.D)
        throw std::runtime_error("model.norm.weight is " +
                                 std::to_string(final_norm_w.size()) +
                                 " wide, expected " + std::to_string(desc.D));

    _assert_norm_weights_folded();

    prefill_mod = std::make_unique<qwen38::prefill_cpu>(&desc, &stack);
    verify_mod  = std::make_unique<qwen38::verify_cpu>(&desc, &stack,
                                                       embedding.get(), head.get(),
                                                       &final_norm_w,
                                                       lm_head8_npu.get());

    // ---- the DFlash2 drafter ----------------------------------------------
    // A draft package (the catalog's draft_model_path, or $FLM_DFLASH_PATH for a dev run) replaces the MTP head below:
    // the same verify pass checks a block of 7 drafts, they just come from z-lab's block-diffusion drafter, which
    // conditions on the target's hidden states at five layers (tap_buffer) instead of on the last one.
    // FLM_DFLASH=0 ignores the package and falls back to the MTP head the base checkpoint carries.
    {
        std::string draft_dir = config._json_config.value("draft_model_path", std::string());
        if (const char* e = std::getenv("FLM_DFLASH_PATH")) if (e[0]) draft_dir = e;
        const char* off = std::getenv("FLM_DFLASH");
        if (!draft_dir.empty() && off && off[0] == '0') {
            header_print("FLM", "FLM_DFLASH=0 -- not loading the DFlash2 drafter");
            draft_dir.clear();
        }
        if (!draft_dir.empty()) {
            const std::string wfile = utils::path_join(draft_dir, "dflash_weight.q4nx");
            if (!std::filesystem::exists(wfile)) {
                header_print("WARNING", "no " + wfile + " -- the DFlash2 drafter is not loaded");
            } else {
                const qwen38::dflash2_config dcfg =
                    qwen38::dflash2_config::from_file(utils::path_join(draft_dir, "config.json"));
                df_weights = std::make_unique<cpu_ref::safetensors>(wfile);
                df = std::make_unique<qwen38::dflash2_draft>(dcfg, *df_weights, embedding.get(), lm_head8_npu.get(),
                                                             npu, proj_xclbin, (int)desc.D);
                taps.layers    = dcfg.taps;
                taps.D         = (int)desc.D;
                taps.keep_rows = dcfg.window + 64;
                prefill_mod->set_taps(&taps);
                verify_mod->set_taps(&taps);
                header_print("FLM", "DFlash2 drafter loaded from " + draft_dir + " (" +
                                    std::to_string(dcfg.layers) + " layers, block " + std::to_string(dcfg.block) +
                                    ", taps " + std::to_string(dcfg.taps.size()) + ")");
            }
        }
    }

    // ---- module 4: the vision tower ---------------------------------------
    // Its own file, not model.q4nx, and opened only if it is actually there: a
    // text-only deployment of this checkpoint is a normal thing to ship, and a
    // throw here would turn a missing optional file into a dead model. What
    // must NOT happen silently is the other direction -- images arriving with
    // no tower to encode them -- and prefill() throws on exactly that.
    //
    // Every matmul and the attention run on the device, behind the vision_ops
    // seam. The seam is still what makes that swappable -- `image_cpu` asks a
    // factory for its 111 matmuls and its one attention and never learns what
    // it got -- but the engine now only ever hands it the device factory. The
    // host ops survive as `vision::cpu_mm_op` / `cpu_attn_op`, kept as the
    // reference to score the device against.
    //
    // FLM_Q38_VISION=0 declines the whole tower, which is how to get a
    // text-only run out of a multimodal checkpoint. It is not a backend
    // switch: there is no configuration here that encodes an image on the host.
    if (vision_desc.present) {
        const char* vs = std::getenv("FLM_Q38_VISION");
        const bool  want = !vs || vs[0] != '0';
        if (!want) {
            header_print("FLM", "FLM_Q38_VISION=0 -- no vision tower; images "
                                "will be rejected");
        } else if (!std::filesystem::exists(vision_desc.weight_path)) {
            header_print("FLM", "config declares a vision_config but there is no "
                                + vision_desc.weight_path +
                                " -- text-only, images will be rejected");
        } else {
            // The splice memcpy's the merger's output straight into the hidden
            // states, so the two widths are one assumption and it is checked
            // here rather than discovered as a stride.
            if (vision_desc.out_hidden_size != desc.D)
                throw std::runtime_error(
                    "vision_config.out_hidden_size is " +
                    std::to_string(vision_desc.out_hidden_size) +
                    " but the language model's hidden_size is " +
                    std::to_string(desc.D));
            // The attention runs on vision_attn.xclbin and the matmuls on the
            // two vision_mm designs. All three are required: the tower has no
            // host path any more, so a missing file throws here rather than
            // silently encoding an image 5x slower.
            //
            // The device always dispatches; there is no length threshold, and
            // it does not need one. N15 check 9, both arms at these flags:
            // 1.62 vs 4.48 ms at L=256 and 7.92 vs 41.68 at L=1200, so the
            // crossover is below the smallest image this checkpoint produces.
            // Over 27 blocks at the real operating point -- a 639x480 image
            // smart_resizes to a 30x40 grid, so L = 1200 -- that is 214 ms per
            // image against 1125 for the attention alone.
            //
            // Both mm designs or neither: A serves every shape but merger.fc2,
            // whose N of 5120 is not a multiple of A's tile_N of 384, so half
            // the pair is a tower that throws on its last layer. All three
            // source files are named for their design in the build tree;
            // activate.sh renames them on the way in, which is the trap
            // vision_attn.xclbin hit.
            //
            // Scoring the port is a job for a standalone check that holds
            // `vision::cpu_op_factory` and the two host ops for exactly
            // this. Measure that host arm at -O2 and it reads 3x slower; the
            // first estimate here said the device LOST at L=256 for exactly
            // that reason.
            const std::string vision_attn_xclbin =
                utils::path_join(xclbin_dir, "vision_attn.xclbin");
            const std::string mm_a_xclbin =
                utils::path_join(xclbin_dir, "vision_mm_a.xclbin");
            const std::string mm_b_xclbin =
                utils::path_join(xclbin_dir, "vision_mm_b.xclbin");
            for (const std::string* p :
                 {&vision_attn_xclbin, &mm_a_xclbin, &mm_b_xclbin})
                if (!std::filesystem::exists(*p))
                    throw std::runtime_error("no " + *p + " -- the vision "
                                             "tower has no host path");

            auto vision_factory = std::make_unique<q38npu::vision_op_factory>(
                npu->register_xclbin(vision_attn_xclbin));
            vision_factory->enable_mm(npu->register_xclbin(mm_a_xclbin),
                                      npu->register_xclbin(mm_b_xclbin));

            image_mod = std::make_unique<qwen38::image_cpu>(
                vision_desc, std::move(vision_factory));
            DEBUG_BLOCK(2,
                header_print("FLM", "vision tower loaded (" +
                                    std::to_string(vision_desc.depth) +
                                    " blocks, " + image_mod->backend() + ")");
            );
        }
    }

    // A checkpoint without the mtp.* tensors is still a usable base model, so
    // a missing head disables speculation rather than throwing.
    if (!df && desc.mtp_num_hidden_layers > 0 &&
        weights->has("model.mtp.layers.0.self_attn.q_proj.weight")) {
        // Same rule as the attention stack: no host arm. The draft head used
        // to have one (`FLM_Q38_MTP_CPU`) because speculation is output-neutral
        // by construction -- speculate() verifies every drafted token against
        // the real stack and truncates at the first mismatch -- so the switch
        // cost throughput and nothing else. It is gone now that the device path
        // is the shipping one. `qwen38::speculate_cpu` survives as the
        // *reference* to score this against, which is
        // the only job it had left. Note what that reference measures: the
        // CPU head is ~30
        // ms/step against the device's ~34, so this is parity-with-freed-cores,
        // not a speedup.
        //
        // The head's fused layer is its own bitstream and its own hw_context --
        // it dispatches standalone, never inside an attention runlist, so the
        // two never need to share one.
        //
        // Three bitstreams, because a draft is two shapes and the wide one is
        // itself split: MTP_layer.xclbin for the six one-token steps, the same
        // attn.xclbin module 1 uses for step 0's length-L attention core, and
        // dequant_mm.xclbin for step 0's projections. register_xclbin dedupes
        // by name, so the last two share the hw_contexts the attention stack
        // and the projections already opened.
        spec_mod = std::make_unique<qwen38::speculate_npu>(
            *weights, &desc, embedding.get(), (int)MAX_L, npu,
            utils::path_join(xclbin_dir, "MTP_layer.xclbin"),
            prefill_attn_xclbin, proj_xclbin, lm_head_npu.get());
    }
    else if (!df && desc.mtp_num_hidden_layers > 0) {
        header_print("FLM", "config declares an MTP head but model.q4nx carries "
                            "no model.mtp.* tensors -- speculation disabled");
    }

    this->clear_context();
}

/// RMSNorm here is `x_hat * w`, not `x_hat * (1 + w)` -- the +1 is folded in by
/// the converter. Adding it again scales every hidden state by roughly two and
/// the model keeps emitting fluent, subtly wrong text. The signature is a norm
/// weight whose mean sits near 0 instead of near 1.
void qwen3_8mtp_npu::Impl::_assert_norm_weights_folded() {
    const std::vector<float> w =
        weights->f32("model.layers.0.input_layernorm.weight");
    double acc = 0.0;
    for (float v : w) acc += v;
    const double mean = acc / (double)w.size();
    if (mean < 0.5 || mean > 2.0)
        throw std::runtime_error(
            "layer 0 input_layernorm mean is " + std::to_string(mean) +
            ", expected ~1 -- this checkpoint's RMSNorm +1 does not look folded");
}

// ===============================================
// context
// ===============================================
void qwen3_8mtp_npu::Impl::set_context_length(int L, int new_general_token_idx) {
    this->current_context_length = L;
    this->general_token_idx      = new_general_token_idx;
    // the rows written above the cursor reach at most the verify window and the 8-token tile past it
    this->attn_dirty_rows = std::max(this->attn_dirty_rows, L + 64);
}

void qwen3_8mtp_npu::Impl::clear_context() {
    if (!stack.empty()) _zero_attn_rows(0);   // before reset(): a fresh context must not read the last conversation's rows
    stack.reset();
    if (spec_mod) { spec_mod->reset(); spec_mod->zero_stale(); }
    if (df) df->reset();
    is_checkpoint_valid = false;
    committed_ids.clear();
    mtp_pos_base    = 1;   // the head's shift, re-established on the next prime
    last_hidden     = nullptr;
    last_window_len = 0;
    mtp_hist.clear();
    mtp_hist_base   = 0;
    // No prompt is pending yet; the prefill that follows re-arms it.
    spec_prime_pending = false;
    // The m-rope walk restarts with the conversation. Leaving it would resume
    // the next prompt at the old position and, worse, half-way into an image.
    mrope = qwen38::mrope_cursor{};
    set_context_length(0, 0);
}

int qwen3_8mtp_npu::Impl::get_current_context_length() {
    return this->current_context_length;
}

void qwen3_8mtp_npu::Impl::update_max_length(uint32_t new_MAX_L) {
    // Through the same funnel as the constructor, so the number compared here,
    // the number handed to the layouts and the rows allocated are one number.
    new_MAX_L = _kv_buffer_rows(_clamp_context(new_MAX_L));
    if (new_MAX_L <= this->MAX_L) {
        header_print("FLM", "New length is shorter than the current length, no need to update!");
        return;
    }
    if (!weights) { this->MAX_L = new_MAX_L; return; }

    // Each layer grows its own cache in place, keeping its rows, so the
    // conversation survives the resize and no weight is re-read. Only the 16
    // attention layers are bounded by MAX_L.
    this->MAX_L = new_MAX_L;
    stack.set_max_length((int)this->MAX_L);
    if (spec_mod) spec_mod->set_max_length((int)this->MAX_L);

    header_print("FLM", "Context grown to " + std::to_string(this->MAX_L) +
                        "; caches were resized in place and keep their contents");
}

// ===============================================
// position ids
// ===============================================
std::vector<int32_t> qwen3_8mtp_npu::Impl::get_input_idx(
        std::vector<int>& tokens, void* payload,
        std::vector<qwen38::image_span>* spans) {
    // Pure text: all three m-rope channels carry the same value and the table
    // degenerates to 1-D RoPE. A multimodal window does not, because position
    // advance is not the token count -- in the reference dump 221 tokens reach
    // position 165.
    const int L = (int)tokens.size();
    const qwen3_8mtp_payload_t* p =
        static_cast<const qwen3_8mtp_payload_t*>(payload);

    // An explicit table wins: a caller reproducing a captured window has taken
    // the whole job, and the walk below would disagree with it over exactly the
    // rows that matter.
    if (p && p->position_ids)
        return std::vector<int32_t>(p->position_ids, p->position_ids + 3 * (size_t)L);

    // Video, before either branch below. The text walk would hand a video token
    // an ordinary text position and the image walk would never see it, so the
    // whole window past it would be off by the video's extent -- with no error
    // anywhere. config.json defines video_token_id for this checkpoint; nothing
    // in this engine can encode one. Say so here.
    if (vision_desc.video_token_id >= 0) {
        for (int t = 0; t < L; t++)
            if (tokens[(size_t)t] == vision_desc.video_token_id)
                throw std::runtime_error(
                    "prompt row " + std::to_string(t) + " is a video token (id " +
                    std::to_string(vision_desc.video_token_id) + "): this engine "
                    "has no video path -- the vision tower's attention mask is a "
                    "single full block, which is only correct within one frame");
    }

    if (p && p->images && p->images->num_images > 0) {
        if (vision_desc.image_token_id < 0)
            throw std::runtime_error("images supplied but this checkpoint has no "
                                     "image_token_id");
        // Re-seed from the engine's own counter whenever the walk is between
        // images. general_token_idx is what checkpoint()/restore() carry, so
        // this is the one place the two can be reconciled; mid-image the cursor
        // is the only authority and general_token_idx is a partial maximum.
        if (mrope.consumed == 0) mrope.pos = (int32_t)general_token_idx;

        std::vector<qwen38::image_grid> grids((size_t)p->images->num_images);
        for (int i = 0; i < p->images->num_images; i++) {
            const qwen3_8mtp_image_t& im = p->images->images[i];
            grids[(size_t)i] = qwen38::image_grid{im.grid_t, im.grid_h, im.grid_w};
        }
        return qwen38::build_mrope(tokens.data(), L, vision_desc.image_token_id,
                                   grids.data(), (int)grids.size(),
                                   (int)vision_desc.spatial_merge_size,
                                   mrope, spans, vision_desc.video_token_id);
    }

    std::vector<int32_t> pos((size_t)3 * L);
    for (int t = 0; t < L; t++) {
        const int32_t v = (int32_t)(general_token_idx + t);
        pos[(size_t)0 * L + t] = v;
        pos[(size_t)1 * L + t] = v;
        pos[(size_t)2 * L + t] = v;
    }
    return pos;
}

// ===============================================
// module 4: images
// ===============================================
void qwen3_8mtp_npu::Impl::encode_image(const float* pixel_values, int t, int h,
                                        int w, float* out) {
    if (!image_mod)
        throw std::runtime_error("encode_image: no vision tower is loaded");
    image_mod->encode(pixel_values, qwen38::image_grid{t, h, w}, out);
}

/// Overwrite the placeholder rows with the tower's output. The embedding lookup
/// has already run over the whole window, so the image slots currently hold
/// whatever vector image_token_id maps to -- a real row, not a zero, which is
/// why a failure to splice produces fluent text about nothing.
void qwen3_8mtp_npu::Impl::_splice_image_rows(
        const qwen3_8mtp_payload_t* p,
        const std::vector<qwen38::image_span>& spans, float* x, int L) {
    if (spans.empty()) return;
    if (!image_mod)
        throw std::runtime_error(
            "the prompt carries image tokens but no vision tower is loaded -- "
            "either vision_weight.q4nx is missing or FLM_Q38_VISION=0");

    const size_t D = desc.D;
    // The tower is per-image: an image whose rows a chunked prompt split across
    // two windows is encoded once per window, and `token_offset` says which
    // slice of the result this window wants. That is a repeated 27-block sweep,
    // not a wrong answer -- the tower is a pure function of the pixels. Caching
    // it is the obvious fix if chunked multimodal prompts ever become the
    // common case; today they are not.
    int last_image = -1;
    for (const qwen38::image_span& s : spans) {
        const qwen3_8mtp_image_t& im = p->images->images[s.image_index];
        const int tokens = image_mod->tokens_for(
            qwen38::image_grid{im.grid_t, im.grid_h, im.grid_w});

        if (s.image_index != last_image) {
            if (!im.pixel_values)
                throw std::runtime_error("image " + std::to_string(s.image_index) +
                                         " has no pixel_values");
            image_rows.resize((size_t)tokens * D);
            image_mod->encode(im.pixel_values,
                              qwen38::image_grid{im.grid_t, im.grid_h, im.grid_w},
                              image_rows.data());
            last_image = s.image_index;
        }

        if (s.token_offset + s.count > tokens || s.begin + s.count > L)
            throw std::runtime_error("image span runs past its grid or its window");
        std::memcpy(x + (size_t)s.begin * D,
                    image_rows.data() + (size_t)s.token_offset * D,
                    (size_t)s.count * D * sizeof(float));
    }
}

// ===============================================
// lm_head
// ===============================================
buffer<bf16> qwen3_8mtp_npu::Impl::get_logits(const float* hidden_row) {
    // The one L = 1 projection on the live path, so the one that can take the
    // bitstream. The bitstream norms the row itself, from `model.norm.weight`
    // in the second half of its input buffer -- which is why `hidden_row` goes
    // in unnormalised and `final_norm_w` alongside it.
    //
    // Module 3 is deliberately NOT routed here, and never will be: verify_cpu
    // projects k+1 = 8 rows at once and this design moves the whole 1.35 GB
    // table per row, so eight dispatches would cost ~204 ms against the host's
    // 75.6 (measured, tools/bench_lm_head_rows.cpp -- the host does NOT cost
    // 8x one row, because q4nx_matmul's row loop is inside its weight tile
    // loop). Its backend is `lm_head8_npu`, a separate bitstream that reads
    // the table once for all eight: ~29 ms. See verify/verify_cpu.hpp.
    const bf16* lg = lm_head_npu->run(hidden_row, final_norm_w.data());
    // Already bf16 on the device, so this skips the fp32 round trip a host
    // projection would need.
    std::memcpy(logits.data(), lg, (size_t)desc.vocab_size * sizeof(bf16));
    return logits;
}

// ===============================================
// module 1: prefill
// ===============================================
buffer<bf16> qwen3_8mtp_npu::Impl::prefill(std::vector<int>& ids, void* payload) {
    // Arm the prime split. This entry, not _prefill_with_mm: forward() shares
    // that body at L = 1 and the rollback path calls it to recover from a bad
    // draft, and neither is prompt-phase work. Only a prompt arrives here, and
    // only after one does the head owe a wide catch-up.
    spec_prime_pending = true;
    return this->_prefill_with_mm(ids, payload);
}

buffer<bf16> qwen3_8mtp_npu::Impl::_prefill_with_mm(std::vector<int>& ids,
                                                    void* payload) {
    if (!weights) throw std::runtime_error("prefill before load_weights");
    const int L_in = (int)ids.size();
    if (L_in == 0) return logits;
    // The cheap outer check, against the buffer. The authoritative ones are per
    // pass and live in the layers -- a pass is legal only if the WHOLE of it
    // fits, and both paths round their extent up (see _kv_buffer_rows) -- so
    // this can pass and a later one still refuse. It exists to fail a hopeless
    // prompt before 64 layers of work, not to be the bound.
    if (current_context_length + L_in > (int)MAX_L)
        throw std::runtime_error("prefill would exceed MAX_L (" +
                                 std::to_string(MAX_L) + ")");

    // **Near the top of the cache (FLM#1001).** A prefill pass pads its window
    // to a whole PREFILL_CHUNK and the padding has to fit too, so `fit` real
    // rows are all it can still take. Above 16256 tokens the buffer cannot keep
    // 128 rows of headroom (see _kv_buffer_rows), so a prompt ending there sends
    // its tail through the verify pass, eight rows at a time, which only needs
    // L_TOKENS of room. Text only: the verify pass embeds ids, it cannot splice
    // image rows.
    constexpr int C = q38npu::attn_cache::PREFILL_CHUNK;
    const int fit = ((int)MAX_L - current_context_length) / C * C;
    if (L_in > fit && verify_mod) {
        if (payload)
            throw std::runtime_error("prefill: a prompt with images has to end at least " +
                                     std::to_string(C) + " tokens below the context limit");
        buffer<bf16> out = logits;
        int at = 0;
        if (fit > 0) {
            std::vector<int> head_ids(ids.begin(), ids.begin() + fit);
            out = this->_prefill_with_mm(head_ids, nullptr);
            at = fit;
        }
        while (at < L_in) {
            const int n = std::min(L_in - at, (int)q38npu::npu_lm_head_8::ROWS);
            std::vector<int> part(ids.begin() + at, ids.begin() + at + n);
            out = _verify_window(part);
            at += n;
        }
        return out;
    }

    // Appends, never resets: _chunked_insert calls this repeatedly for one
    // prompt, so a reset here would drop every chunk but the last.
    std::vector<qwen38::image_span> spans;
    const std::vector<int32_t> input_idx = get_input_idx(ids, payload, &spans);

    prefill_mod->begin_window(L_in, input_idx.data());
    float* x = prefill_mod->hidden();

    const qwen3_8mtp_payload_t* p =
        static_cast<const qwen3_8mtp_payload_t*>(payload);
    if (p && p->embeds) {
        // A caller supplying hidden states has already resolved the image
        // rows; the placeholder ids would look up the wrong vectors.
        std::memcpy(x, p->embeds, (size_t)L_in * desc.D * sizeof(float));
    }
    else {
        std::vector<int32_t> ids32(ids.begin(), ids.end());
        embedding->embed_rows(ids32.data(), L_in, x);
        // Module 4, in the one place it belongs: after the lookup has filled
        // every row and before any layer reads one.
        _splice_image_rows(p, spans, x, L_in);

        // A prompt that reserves image rows but hands over no image is the one
        // failure this path cannot survive quietly: image_token_id has a real
        // embedding row, so every slot holds a plausible vector and the model
        // answers fluently about an image it never saw. Not reachable on a
        // text-only checkpoint, where image_token_id stays -1.
        if (spans.empty() && vision_desc.image_token_id >= 0) {
            for (int t = 0; t < L_in; t++) {
                if (ids[(size_t)t] != vision_desc.image_token_id) continue;
                throw std::runtime_error(
                    "row " + std::to_string(t) + " of the prompt is "
                    "image_token_id but prefill() got no images -- pass them in "
                    "qwen3_8mtp_payload_t::images, or resolve them yourself and "
                    "pass ::embeds");
            }
        }
    }

    time_utils::time_point start = time_utils::now();
    prefill_mod->run(L_in);
    DEBUG_BLOCK(1,
        time_utils::time_with_unit d =
            time_utils::re_unit(time_utils::duration_us(start, time_utils::now()));
        header_print("FLM", "prefill " << L_in << " tokens through "
                                       << desc.num_hidden_layers << " layers: "
                                       << d.first << " " << d.second);
    );

    // logits_to_keep = 1: prefill predicts exactly one token. Verify is the
    // phase that needs all k+1 rows.
    last_hidden     = x;
    last_window_len = L_in;
    buffer<bf16> out = get_logits(x + (size_t)(L_in - 1) * desc.D);

    // One past the largest position this window used, not general_token_idx +
    // L_in: with an image in the window those differ.
    int32_t pmax = input_idx[0];
    for (size_t i = 1; i < input_idx.size(); i++) pmax = std::max(pmax, input_idx[i]);
    // committed_ids must track the caches exactly -- speculate() declines if
    // the two ever disagree, which is safe but silent.
    committed_ids.insert(committed_ids.end(), ids.begin(), ids.end());
    // And the hidden history tracks committed_ids. THIS is the capture that
    // makes the first cycle a full-width prime: `x` holds every row this window
    // produced, and it is about to be overwritten by the next call -- which on
    // the live path is a one-row forward(), arriving before speculate() ever
    // runs. Keeping only `last_hidden` is what clamped the prompt's 350-row
    // catch-up to 1 and left the head 349 rows behind for the whole generation.
    _mtp_hist_append(x, L_in);
    // The drafter's view of the same rows: the taps this sweep left, as tokens current_context_length + r.
    _df_append(current_context_length, input_idx.data(), L_in);
    set_context_length(current_context_length + L_in, (int)pmax + 1);
    return out;
}

// ===============================================
// hidden-state taps
// ===============================================
const float* qwen3_8mtp_npu::Impl::last_hidden_states(int& L, int& D) const {
    // Valid until the next begin_window on the module that produced it.
    L = last_window_len;
    D = (int)desc.D;
    return last_window_len ? last_hidden : nullptr;
}

const float* qwen3_8mtp_npu::Impl::last_normed_states(int& L, int& D) {
    L = last_window_len;
    D = (int)desc.D;
    if (!last_window_len || !last_hidden) return nullptr;
    // model.norm, the same normalisation lm_head's input goes through. The MTP
    // head fuses these, not the raw layer-63 tap (cos 1.0 vs 0.94).
    normed_states.assign((size_t)last_window_len * desc.D, 0.0f);
    for (int t = 0; t < last_window_len; t++)
        cpu_ref::rms_norm(last_hidden + (size_t)t * desc.D, final_norm_w.data(),
                          (int)desc.D, normed_states.data() + (size_t)t * desc.D,
                          desc.rms_eps);
    return normed_states.data();
}

void qwen3_8mtp_npu::Impl::_mtp_hist_append(const float* pre_norm, int n) {
    // No head, no consumer: this is the only cost the history has on a build
    // that never speculates, and it is zero.
    if (!spec_mod || !pre_norm || n <= 0) return;

    const size_t D = (size_t)desc.D;
    const size_t at = mtp_hist.size();
    mtp_hist.resize(at + (size_t)n * D);
    // The same normalisation `last_normed_states` applies, for the same reason:
    // the MTP head fuses model.norm's output, not the raw layer-63 tap
    // (cos 1.0 vs 0.94). Doing it here rather than at draft time means a row is
    // normed once, when it is produced, instead of once per cycle that reaches
    // back for it.
    for (int t = 0; t < n; t++)
        cpu_ref::rms_norm(pre_norm + (size_t)t * D, final_norm_w.data(),
                          (int)D, mtp_hist.data() + at + (size_t)t * D,
                          desc.rms_eps);
}

const float* qwen3_8mtp_npu::Impl::mtp_prev_rows(int& rows, int& D) {
    D = (int)desc.D;
    rows = _mtp_hist_rows();
    if (rows > 0) return mtp_hist.data();
    // Nothing captured for this stretch of context -- after a restore(), or on
    // any future path that commits rows without telling the history. Fall back
    // to the last window, which is what this code did before the history
    // existed: correct, just narrow.
    return last_normed_states(rows, D);
}

// ===============================================
// module 2: speculate
// ===============================================
std::vector<int> qwen3_8mtp_npu::Impl::mtp_draft(const std::vector<int>& ids,
                                                 const float* prev_hidden,
                                                 int T, int k) {
    if (!spec_mod) return {};
    // The head's cache is shifted one position behind the base stack's, so its
    // own cursor -- not the engine's context length -- is the position base.
    // A uniform shift is invisible to RoPE; what must be right is the spacing,
    // and that comes from the head's own cache length.
    const int pos0 = mtp_pos_base + spec_mod->cache_len();
    return spec_mod->draft(ids, prev_hidden, T, pos0, k);
}

// ===============================================
// module 3: verify
// ===============================================
int qwen3_8mtp_npu::Impl::prefill_verify(const std::vector<int>& verify_ids,
                                         const std::vector<int>& drafts,
                                         std::vector<int>& out_argmax) {
    if (verify_ids.size() != drafts.size() + 1)
        throw std::runtime_error("prefill_verify: need k+1 ids for k drafts");
    const int L = (int)verify_ids.size();

    // Before the batch: once the 48 delta-net layers have folded these tokens
    // there is no way back, because the fold is destructive.
    verify_mod->snapshot();

    const int ctx_before = current_context_length;
    const int pos_before = general_token_idx;

    std::vector<int> ids = verify_ids;
    const std::vector<int32_t> input_idx = get_input_idx(ids, nullptr);

    out_argmax      = verify_mod->run(ids, input_idx.data());
    last_hidden     = verify_mod->hidden();
    last_window_len = L;

    if (spec_chooser) {
        // The caller's constraint picks each row's token. Strictly in order and only as far as the first mismatch, so the
        // chooser sees exactly the tokens accept_prefix() will commit (accepted drafts, then the free token).
        int rows = 0, vocab = 0;
        const float* lg = verify_mod->logits(rows, vocab);
        if (!lg || rows < L)
            throw std::runtime_error("prefill_verify: a speculation chooser needs the verify logits");
        for (int i = 0; i < L; i++) {
            out_argmax[(size_t)i] = spec_chooser(i, lg + (size_t)i * vocab, vocab);
            if (i < L - 1 && out_argmax[(size_t)i] != drafts[(size_t)i]) break;
        }
    }

    const qwen38::accept_result r = qwen38::accept_prefix(out_argmax, drafts);

    // `accepted + 1` tokens commit -- the first mismatching position still
    // yields a correct base-model token. So the batch over-appended by
    // L - (accepted + 1) rows.
    //
    // Unless a stop token lands inside that prefix, in which case commit stops
    // one past it. Everything after it is output the caller will never emit,
    // and leaving it in the caches is not free: the caller breaks at the stop
    // token, so those rows become a context the token log does not describe.
    // Truncating here also restores the batch's headline invariant -- the last
    // token returned is the one WITHOUT a KV row -- which is what lets the
    // caller treat a stop token out of a batch exactly like one out of
    // forward(), rather than having to know it was already committed.
    //
    // Note this is a CACHE decision only. It deliberately does not reach the
    // statistics: the drafts past the stop were still drafted and still
    // judged, and speculate() counts them, because how well the head predicts
    // the base model has nothing to do with where the turn ends.
    const int stop_at = _first_stop(out_argmax, r.accepted);
    const int commit  = (stop_at >= 0) ? stop_at + 1 : r.accepted + 1;
    spec_last_commit  = commit;
    // Cleared unconditionally: the milestone driver calls this with no cycle
    // around it, and a stale value would be charged to whichever cycle ran next.
    spec_last_replay_us = 0;
    if (commit < L) {
        const time_utils::time_point t_replay0 = time_utils::now();
        if (verify_mod->can_commit_prefix()) {
            // Keep the accepted rows where the batch already put them. The
            // attention caches need only their cursor moved; the delta-net
            // conv windows and recurrent states are re-derived from the
            // snapshot plus what the pass wrote out, and that fold is deferred
            // into the next cycle's draft. See decoder_layer::commit_prefix().
            verify_mod->commit_prefix(commit);

            // What `_prefill_with_mm` would have done on the way past, minus
            // the forward pass. `last_hidden` deliberately stays the verify
            // pass's own output: rows 0 .. commit-1 of it are causally
            // independent of the rejected rows -- causal attention, causal
            // conv, causal fold -- so they are the right vectors, and the
            // window is no shorter than a re-prefill would have produced.
            _trim_committed(ctx_before);
            committed_ids.insert(committed_ids.end(), verify_ids.begin(),
                                 verify_ids.begin() + commit);
            // Rows 0..commit-1 of the verify pass, for the same causality
            // reason the comment above gives for keeping `last_hidden`: they
            // are independent of the rejected rows, so they are exactly the
            // vectors a re-prefill would have produced. `_trim_committed`
            // already dropped anything at or above ctx_before.
            _mtp_hist_append(last_hidden, commit);
            // and the drafter's taps of the same accepted rows (rows 0 .. commit-1 are independent of the rejected ones)
            _df_append(ctx_before, input_idx.data(), commit);
            int32_t pmax = input_idx[0];
            for (int c = 0; c < 3; c++)
                for (int i = 0; i < commit; i++)
                    pmax = std::max(pmax, input_idx[(size_t)c * L + i]);
            set_context_length(ctx_before + commit, (int)pmax + 1);
            last_window_len = commit;
        }
        else {
            // Roll back and re-run. The attention layers would truncate fine,
            // but a delta-net block that kept no arena has no per-token undo,
            // so the snapshot is the only correct starting point -- and the
            // model is the only way forward from it. A full 64-layer weight
            // stream, paid to recover two tensors.
            verify_mod->rollback();
            set_context_length(ctx_before, pos_before);
            _trim_committed(ctx_before);

            std::vector<int> keep(verify_ids.begin(), verify_ids.begin() + commit);
            this->_prefill_with_mm(keep, nullptr);
        }
        spec_last_replay_us = _elapsed_us(t_replay0);
    }
    else {
        verify_mod->discard();
        int32_t pmax = input_idx[0];
        for (size_t i = 1; i < input_idx.size(); i++)
            pmax = std::max(pmax, input_idx[i]);
        // The batch bypassed _prefill_with_mm, so record the ids here -- and
        // the hidden rows with them. Every row committed, so every row is live.
        committed_ids.insert(committed_ids.end(), verify_ids.begin(), verify_ids.end());
        _mtp_hist_append(last_hidden, L);
        _df_append(ctx_before, input_idx.data(), L);
        set_context_length(ctx_before + L, (int)pmax + 1);
    }

    // The head is shifted one position behind the base stack and must stay
    // that way.
    if (spec_mod) spec_mod->crop(current_context_length - 1);

    // Both recovery paths above have to leave the caches aligned the same way,
    // and the fast one gets there by moving 16 cursors rather than by re-running
    // the layers that own them. Off-by-one there still emits fluent text, with
    // acceptance quietly decaying -- so assert it rather than watch for it.
    DEBUG_BLOCK(1,
        verify_mod->check_invariant(current_context_length,
                                    spec_mod ? spec_mod->cache_len() : -1););

    // The hidden history is the third ledger and has the same failure mode: a
    // commit site that appends ids without appending rows leaves the head
    // pairing every token with its neighbour's state, which drafts fluently and
    // shows up only as acceptance decaying over a long run.
    //
    // Debug-only and a print rather than a throw, on the same terms as the
    // check above: a desync costs acceptance and never correctness, because
    // verify overrides every draft it produces. Crashing a release build over a
    // quality regression would be the worse trade.
    DEBUG_BLOCK(1,
        if (spec_mod && mtp_hist_base + _mtp_hist_rows() != current_context_length)
            header_print("FLM", "MTP hidden history covers [" +
                                std::to_string(mtp_hist_base) + ", " +
                                std::to_string(mtp_hist_base + _mtp_hist_rows()) +
                                ") but the context is " +
                                std::to_string(current_context_length) +
                                " -- a commit path appended ids without "
                                "appending hidden rows"););

    return r.accepted;
}

// ===============================================
// the cycle: draft / verify / commit
// ===============================================
std::vector<int> qwen3_8mtp_npu::Impl::mtp_catch_up_window(int last_token,
                                                           int width) const {
    // The head's row r holds position r+1, so covering positions cache+1 .. C
    // means committed_ids[cache+1 .. C-1] followed by last_token, the only
    // committed token without a KV row yet. At cold start this reduces to
    // [prompt[1:], x_P] with no special case.
    const int C = current_context_length;
    std::vector<int> w;
    w.reserve(width);
    for (int i = C - width + 1; i < C; i++) w.push_back(committed_ids[i]);
    w.push_back(last_token);
    return w;
}

std::vector<int> qwen3_8mtp_npu::Impl::speculate(int last_token, int max_draft) {
    std::vector<int> committed = speculate_inner(last_token, max_draft);
    if (committed.empty()) spec_declines++;
    return committed;
}

/// One cycle with the DFlash2 drafter: one block pass drafts up to 7 tokens after `last_token`, then the same verify,
/// commit and statistics as the MTP cycle below. There is no catch-up window and no head cache to crop: the drafter
/// folds the taps of the rows committed since its last pass, and keeps its own ledger of them (dflash2_draft::append,
/// truncate), so what it needs from this function is the anchor's index and position.
std::vector<int> qwen3_8mtp_npu::Impl::speculate_df(int last_token, int max_draft) {
    if (!df || !weights) return {};
    if (max_draft <= 0) return {};
    if (current_context_length <= 0) return {};
    if ((int)committed_ids.size() != current_context_length) return {};

    const int k = std::min(max_draft, qwen38::dflash2_draft::ROWS - 1);
    // The verify pass appends L_TOKENS rows however few drafts there are, so
    // that, not k + 1, is what has to fit; past it decode plainly (FLM#1001).
    if (current_context_length + std::max(k + 1, (int)q38npu::npu_lm_head_8::ROWS) > (int)MAX_L) return {};

    // The drafter's rows must end at the last committed token. A hole or a stale tail means some path changed the
    // context without telling it: forget and draft cold, which costs acceptance and never correctness.
    if (df->last_idx() >= current_context_length) df->truncate(current_context_length);
    if (df->last_idx() >= 0 && df->last_idx() != current_context_length - 1) df->reset();

    const time_utils::time_point t_cycle0 = time_utils::now();
    const time_utils::time_point t_draft0 = time_utils::now();
    const std::vector<int> drafts = df->draft(last_token, current_context_length, general_token_idx, k);
    uint64_t draft_us = _elapsed_us(t_draft0);
    // The first cycle after a prompt folds the whole prompt's taps (prefill-shaped work inside speculate()): charge
    // that to prefill, as the MTP head's prime is, so decode tok/s does not depend on the prompt length.
    const bool was_prime = spec_prime_pending;
    spec_prime_pending = false;
    spec_last_prime_us = 0;
    if (was_prime) {
        const uint64_t absorb_us = std::min(df->last_absorb_us(), draft_us);
        spec_last_prime_us = absorb_us;
        spec_prime_us     += absorb_us;
        spec_prime_rows   += (uint64_t)df->last_absorb_rows();
        spec_prime_steps  += 1;
        draft_us          -= absorb_us;
    }
    if ((int)drafts.size() != k) return {};

    std::vector<int> verify_ids;
    verify_ids.reserve(k + 1);
    verify_ids.push_back(last_token);
    verify_ids.insert(verify_ids.end(), drafts.begin(), drafts.end());

    std::vector<int> vargmax;
    spec_last_replay_us = 0;
    const time_utils::time_point t_verify0 = time_utils::now();
    const int accepted = prefill_verify(verify_ids, drafts, vargmax);
    const uint64_t verify_total_us = _elapsed_us(t_verify0);
    const uint64_t replay_us = std::min(spec_last_replay_us, verify_total_us);
    const uint64_t verify_us = verify_total_us - replay_us;

    const int commit = spec_last_commit;
    std::vector<int> committed(vargmax.begin(), vargmax.begin() + commit);
    spec_logit_rows = commit;

    spec_cycles++;
    spec_accepted_total += (uint64_t)accepted;
    spec_drafted_total  += (uint64_t)k;
    spec_tokens_total   += (uint64_t)committed.size();
    spec_draft_us    += draft_us;
    spec_verify_us   += verify_us;
    spec_replay_us   += replay_us;
    spec_cycle_us    += _elapsed_us(t_cycle0);
    spec_draft_steps += 1;                       // one block pass, not k serial steps
    spec_verify_rows += (uint64_t)verify_ids.size();
    if (commit < k + 1) spec_replays++;
    DEBUG_BLOCK(1,
        header_print("FLM", "speculate (DFlash2): " + std::to_string(accepted) + "/" + std::to_string(k) +
                            " accepted, mean " + std::to_string(mean_accepted_length())););
    return committed;
}

std::vector<int> qwen3_8mtp_npu::Impl::speculate_inner(int last_token,
                                                       int max_draft) {
    // Every `return {}` is a decline, not a failure: the caller runs its
    // ordinary forward() + sample() instead. So all of them come before any
    // state is mutated.
    spec_logit_rows = 0;
    if (df) return speculate_df(last_token, max_draft);
    if (!spec_mod || !weights) return {};
    if (max_draft <= 0) return {};
    if (current_context_length <= 0) return {};
    if ((int)committed_ids.size() != current_context_length) return {};

    const int k = std::min(max_draft, spec_mod->max_steps());

    // The verify batch appends k+1 rows before any rollback, so the headroom
    // check is against k+1, not against the number finally committed -- and
    // never less than L_TOKENS, which the pass appends however few drafts
    // there are; past that, decode plainly (FLM#1001).
    if (current_context_length + std::max(k + 1, (int)q38npu::npu_lm_head_8::ROWS) > (int)MAX_L) return {};

    // ---- 1. catch the head up ---------------------------------------------
    int width = current_context_length - spec_mod->cache_len();
    if (width <= 0) return {};          // head ahead of the base stack
    const int width_raw = width;        // before the shortfall clamp below

    // Every row the head has not folded, back to `mtp_hist_base` -- the prompt
    // included, because `_prefill_with_mm` captured it on the way past. On the
    // first cycle of a prompt this is the whole prefill window, which is what
    // makes step 0 a real prefill rather than a one-row decode.
    int L = 0, D = 0;
    const float* prev = mtp_prev_rows(L, D);
    if (!prev || L <= 0) return {};

    if (width > L) {
        // Rows [C-width, C-L) have no live hidden states, so the head cannot
        // fuse them. Skip them, and move the head's row 0 forward by as many as
        // we skipped so that base + cache_len() still names the right absolute
        // position. A shorter head context costs acceptance; a wrong rotary
        // position costs the draft outright.
        //
        // With the history in place this is the fallback, not the steady state:
        // it fires after a restore(), whose continuation's hidden states were
        // discarded with the continuation. If DEBUG_LEVEL=1 shows it firing on
        // an ordinary cycle, a commit site is appending ids without appending
        // rows -- see `_trim_committed`.
        mtp_pos_base += (width - L);
        width = L;
    }
    const float* prev_tail = prev + (size_t)(L - width) * (size_t)D;

    const std::vector<int> window = mtp_catch_up_window(last_token, width);

    // Timed from here: everything above also runs on a decline, and a decline
    // is not a cycle.
    const time_utils::time_point t_cycle0 = time_utils::now();

    // ---- 2. draft ----------------------------------------------------------
    // Snapshot-free: the head runs against its own private KV cache and never
    // touches the 64 base layers.
    const time_utils::time_point t_draft0 = time_utils::now();
    const std::vector<int> drafts = mtp_draft(window, prev_tail, width, k);
    uint64_t draft_us = _elapsed_us(t_draft0);

    // The catch-up ran and the head's cache moved, so the prime is spent
    // whatever happens below -- disarm before the decline return, or the next
    // cycle's one-row step 0 would be charged to prefill instead.
    const bool was_prime = spec_prime_pending;
    spec_prime_pending = false;

    // The catch-up folded every row up to `current_context_length`, including
    // the ones the clamp above abandoned -- `mtp_pos_base` moved past those, so
    // no later cycle can use them either. Released here rather than at the
    // bottom for the same reason the line above is: the decline returns below
    // skip the bottom, and the head's cache advanced regardless of them. This
    // is what holds the history at one or two rows in the steady state instead
    // of at one whole context.
    _mtp_hist_drop_front(current_context_length);

    if ((int)drafts.size() != k) return {};

    // Split the prime out of the draft. `spec_prime_pending` was armed by the
    // public prefill(), so exactly one cycle per prompt gets this treatment --
    // the first one, whose step 0 is re-absorbing the prompt. From cycle 2 the
    // head is cropped to C - 1 every cycle, step 0 is one or two rows, and the
    // whole draft stays in spec_draft_us.
    //
    // min(): the head's clock and this one bracket nested intervals, but they
    // are two separate reads and a preempted thread can invert them.
    uint64_t prime_steps = 0;
    spec_last_prime_us = 0;
    if (was_prime) {
        const uint64_t step0_us = std::min(spec_mod->last_step0_us(), draft_us);
        spec_last_prime_us = step0_us;
        spec_prime_us     += step0_us;
        spec_prime_rows   += (uint64_t)width;
        spec_prime_steps  += 1;
        // Out of the numerator AND out of the denominator, so ms/step stays
        // the cost of a DECODE step. Leaving the wide pass in with k below
        // smears a prompt-sized cost across seven one-row steps and overstates
        // every one of them.
        draft_us    -= step0_us;
        prime_steps  = 1;
    }

    // ---- 3. verify ---------------------------------------------------------
    // [last_token, drafts...]: last_token is the committed token still missing
    // a KV row, and each draft is checked against the argmax before it.
    std::vector<int> verify_ids;
    verify_ids.reserve(k + 1);
    verify_ids.push_back(last_token);
    verify_ids.insert(verify_ids.end(), drafts.begin(), drafts.end());

    std::vector<int> vargmax;
    spec_last_replay_us = 0;
    const time_utils::time_point t_verify0 = time_utils::now();
    const int accepted = prefill_verify(verify_ids, drafts, vargmax);
    const uint64_t verify_total_us = _elapsed_us(t_verify0);
    // Subtract the replay so "verify" means the batched k+1 pass alone --
    // otherwise a cycle that rejected everything would look like slow
    // verification rather than a bad draft paying for a re-fold.
    const uint64_t replay_us = std::min(spec_last_replay_us, verify_total_us);
    const uint64_t verify_us = verify_total_us - replay_us;

    // ---- 4. commit ---------------------------------------------------------
    // Normally `accepted + 1`: the agreed drafts plus the base model's own
    // argmax at the first mismatch. This is what makes speculation
    // output-neutral. prefill_verify() shortens it when a stop token lands
    // inside that prefix, and the caches were trimmed to match, so read the
    // count from there rather than recomputing it here.
    const int commit = spec_last_commit;
    std::vector<int> committed(vargmax.begin(), vargmax.begin() + commit);
    // committed[i] is the argmax of verify row i, so these rows are its logits
    spec_logit_rows = commit;

    // Both counters are raw: `accepted` as the verifier returned it, and every
    // draft slot offered. Deliberately blind to stop tokens. These two numbers
    // measure ONE thing -- how well the draft head predicts the base model --
    // and that is a property of the head, not of where the turn happens to
    // end. Discounting the slots past an EOS made the rate depend on how short
    // the answer was, which is not something the head controls.
    //
    // Consequence, and it is intended: when a stop truncates the commit,
    // `accepted` can exceed the tokens actually returned, so the accepted
    // total does not reconcile against the caller's completion count. The
    // reconciling number is spec_tokens_total, printed on the same line.
    spec_cycles++;
    spec_accepted_total += (uint64_t)accepted;
    spec_drafted_total  += (uint64_t)k;
    spec_tokens_total   += (uint64_t)committed.size();

    spec_draft_us    += draft_us;
    spec_verify_us   += verify_us;
    spec_replay_us   += replay_us;
    spec_cycle_us    += _elapsed_us(t_cycle0);
    spec_draft_steps += (uint64_t)k - prime_steps;
    spec_verify_rows += (uint64_t)verify_ids.size();
    // The event, not the clock. In-place recovery can land inside a
    // microsecond, and `if (replay_us)` would then stop counting recoveries
    // exactly when it got fast enough to matter.
    //
    // `commit`, not `accepted`: this is the denominator of the replay row in
    // the breakdown, so it has to count what actually SPENT that time -- any
    // batch that did not commit whole, whether a draft was rejected or a stop
    // token truncated it. Keying it on `accepted` left the one cycle per turn
    // that stops on a full accept paying replay time under a count of zero.
    if (commit < k + 1) spec_replays++;

    DEBUG_BLOCK(1,
        // `width` is step 0's shape, and the only number that says whether the
        // head's catch-up took the prefill shape or the per-row one -- the
        // draft-step counter above is k per cycle either way, so it cannot.
        header_print("FLM", "speculate: " + std::to_string(accepted) + "/" +
                            std::to_string(k) + " accepted, mean " +
                            std::to_string(mean_accepted_length()) +
                            ", step-0 width " + std::to_string(width) +
                            (width == width_raw
                                 ? std::string(" (unclamped)")
                                 : " (CLAMPED from " + std::to_string(width_raw) +
                                   " by " + std::to_string(L) + " live hidden "
                                   "rows -- expected only after restore())"));
    );
    return committed;
}

// ===============================================
// decode
// ===============================================
buffer<bf16> qwen3_8mtp_npu::Impl::forward(int ids) {
    std::vector<int> one(1, ids);
    // FLM#1016: a one-row pass through the prefill module costs ~3.4 s, the verify pass ~0.9 s for eight rows.
    // Plain decode (no speculation, constrained requests) takes the verify pass at L = 1 and commits it in full:
    // no snapshot, nothing to roll back. FLM_Q38_DECODE_VIA_PREFILL=1 restores the old path for an A/B.
    static const bool via_prefill = [] {
        const char* e = std::getenv("FLM_Q38_DECODE_VIA_PREFILL");
        return e && e[0] && std::strcmp(e, "0") != 0;
    }();
    if (via_prefill || !verify_mod) return this->_prefill_with_mm(one, nullptr);
    return _verify_window(one);
}

/// Up to L_TOKENS text tokens through the verify pass, committed in full: no
/// snapshot, nothing to roll back. Returns the last row's logits. Plain decode
/// is the one-token case; _prefill_with_mm sends the tail of a prompt here when
/// the cache has too little room left for the prefill path's 128-row padding.
buffer<bf16> qwen3_8mtp_npu::Impl::_verify_window(std::vector<int>& ids) {
    const int n = (int)ids.size();
    if (n < 1 || n > q38npu::npu_lm_head_8::ROWS)
        throw std::runtime_error("_verify_window: " + std::to_string(n) + " tokens, the verify pass takes 1 to " +
                                 std::to_string(q38npu::npu_lm_head_8::ROWS));
    // The pass appends L_TOKENS rows however few are real, so the whole of it has to fit.
    if (current_context_length + q38npu::npu_lm_head_8::ROWS > (int)MAX_L)
        throw std::runtime_error("a verify pass at " + std::to_string(current_context_length) +
                                 " would exceed MAX_L (" + std::to_string(MAX_L) + ")");
    const int ctx_before = current_context_length;
    const std::vector<int32_t> input_idx = get_input_idx(ids, nullptr);
    verify_mod->run(ids, input_idx.data());
    last_hidden     = verify_mod->hidden();
    last_window_len = n;

    int rows = 0, vocab = 0;
    const float* lg = verify_mod->logits(rows, vocab);
    if (!lg || rows != n || vocab != (int)desc.vocab_size)
        throw std::runtime_error("_verify_window: verify pass returned no logits row");
    const float* last = lg + (size_t)(n - 1) * vocab;
    for (int i = 0; i < vocab; i++) logits.data()[i] = (bf16)last[i];

    // the bookkeeping _prefill_with_mm does for a window
    int32_t pmax = input_idx[0];
    for (size_t i = 1; i < input_idx.size(); i++) pmax = std::max(pmax, input_idx[i]);
    committed_ids.insert(committed_ids.end(), ids.begin(), ids.end());
    _mtp_hist_append(last_hidden, n);
    _df_append(ctx_before, input_idx.data(), n);
    set_context_length(ctx_before + n, (int)pmax + 1);
    return logits;
}

// ===============================================
// checkpoint / restore (multi-turn rewind)
// ===============================================
int qwen3_8mtp_npu::Impl::checkpoint() {
    if (stack.empty()) return -1;
    header_print_r("FLM", "Creating checkpoint at context length " +
                          std::to_string(this->current_context_length));
    stack.save_state(qwen38::state_slot::turn);
    is_checkpoint_valid          = true;
    checkpoint_context_length    = this->current_context_length;
    checkpoint_general_token_idx = this->general_token_idx;
    return this->current_context_length;
}

int qwen3_8mtp_npu::Impl::restore() {
    if (!is_checkpoint_valid) {
        header_print_r("FLM", "No valid checkpoint to restore");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " +
                          std::to_string(checkpoint_context_length));
    // The `turn` slot is untouched by speculation, so any number of cycles may
    // have run since the checkpoint was taken.
    stack.restore_state(qwen38::state_slot::turn);
    // A stale id log is longer than the context, which makes speculate()
    // decline permanently: correct output, no speculation, no diagnostic.
    _trim_committed(checkpoint_context_length);
    // The head's cache is a draft-side accelerator, not model state, so it is
    // not checkpointed -- crop it back to the invariant.
    if (spec_mod) spec_mod->crop(checkpoint_context_length - 1);
    // The restored window's hidden states belong to the discarded continuation.
    // `_trim_committed` already dropped any history row at or above the
    // checkpoint for the same reason; rows below it were produced before the
    // checkpoint was taken and are still the right vectors for their positions,
    // so they are kept and the next prime uses them.
    last_hidden     = nullptr;
    last_window_len = 0;
    // The m-rope walk too: clear_context() was the only place that reset it, so a prompt after restore() inherited the
    // image counters of the discarded continuation. Its position is re-derived from general_token_idx while no image
    // is consumed.
    mrope = qwen38::mrope_cursor{};
    set_context_length(checkpoint_context_length, checkpoint_general_token_idx);
    return this->current_context_length;
}

// ===============================================
// saved states (the multi-slot prefix cache)
// ===============================================
std::unique_ptr<hybrid_engine_state> qwen3_8mtp_npu::Impl::new_state() {
    size_t fixed = 0, rows = 0;
    for (size_t i = 0; i < stack.size(); i++) {
        if (stack[(int)i].is_full_attention()) rows += (size_t)2 * stack[(int)i].kv_row_width() * sizeof(bf16);
        else fixed += stack[(int)i].blob_bytes();
    }
    // a saved row also carries its id and its hidden-history row
    rows += sizeof(int) + (size_t)desc.D * sizeof(float);
    return std::make_unique<qwen38_engine_state>(fixed, rows);
}

int qwen3_8mtp_npu::Impl::read_state(hybrid_engine_state& base, int kv_from) {
    auto* st = dynamic_cast<qwen38_engine_state*>(&base);
    const int length = this->current_context_length;
    // speculate() declines when the id log and the context disagree; a state of such a context would carry the same mismatch
    if (st == nullptr || stack.empty() || (int)committed_ids.size() != length) return -1;
    kv_from = std::clamp(kv_from, 0, length);
    header_print_r("FLM", "Reading state at context length " + std::to_string(length) + ", K/V rows from " + std::to_string(kv_from));
    const size_t n = stack.size();
    st->state.assign(n, buffer<bf16>());
    st->kv_k.assign(n, buffer<bf16>());
    st->kv_v.assign(n, buffer<bf16>());
    const size_t rows = (size_t)(length - kv_from);
    for (size_t i = 0; i < n; i++) {
        qwen38::decoder_layer& layer = stack[(int)i];
        if (layer.is_full_attention()) {
            if (rows == 0) continue;
            const size_t w = (size_t)layer.kv_row_width();
            st->kv_k[i] = buffer<bf16>(rows * w);
            st->kv_v[i] = buffer<bf16>(rows * w);
            layer.read_rows(kv_from, length, st->kv_k[i].data(), st->kv_v[i].data());
        } else {
            st->state[i] = buffer<bf16>(layer.blob_bytes() / sizeof(bf16));
            layer.export_blob(st->state[i].data());
        }
    }
    st->ids = committed_ids;
    st->hist = mtp_hist;
    st->hist_base = mtp_hist_base;
    st->filled = true;
    st->context_length = length;
    st->general_token_idx = this->general_token_idx;
    st->kv_from = kv_from;
    return length;
}

/// @brief whether `st` holds what this engine's write_state needs (a state of another engine would not)
static bool qwen38_state_fits(const qwen38_engine_state* st, qwen38::layer_stack& stack, uint32_t D, bool need_blobs) {
    if (st == nullptr || !st->filled || stack.empty() || st->state.size() != stack.size() || st->kv_k.size() != stack.size() ||
        st->kv_v.size() != stack.size() || st->context_length < st->kv_from || (int)st->ids.size() != st->context_length) {
        return false;
    }
    const size_t rows = (size_t)(st->context_length - st->kv_from);
    for (size_t i = 0; i < stack.size(); i++) {
        qwen38::decoder_layer& layer = stack[(int)i];
        if (layer.is_full_attention()) {
            const size_t w = (size_t)layer.kv_row_width();
            if (rows > 0 && (st->kv_k[i].size() != rows * w || st->kv_v[i].size() != rows * w)) return false;
        } else if (need_blobs && st->state[i].size() * sizeof(bf16) != layer.blob_bytes()) {
            return false;
        }
    }
    return st->hist.size() % (D ? D : 1) == 0;
}

int qwen3_8mtp_npu::Impl::write_state_kv(const hybrid_engine_state& base) {
    const auto* st = dynamic_cast<const qwen38_engine_state*>(&base);
    if (!qwen38_state_fits(st, stack, desc.D, false)) return -1;
    if (df) df->reset();
    const int rows = st->context_length - st->kv_from;
    if (rows > 0) {
        for (size_t i = 0; i < stack.size(); i++) {
            if (stack[(int)i].is_full_attention()) {
                stack[(int)i].write_rows(st->kv_from, st->context_length, st->kv_k[i].data(), st->kv_v[i].data());
            }
        }
    }
    return st->context_length;
}

int qwen3_8mtp_npu::Impl::write_state(const hybrid_engine_state& base) {
    const auto* st = dynamic_cast<const qwen38_engine_state*>(&base);
    if (!qwen38_state_fits(st, stack, desc.D, true)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st->context_length));
    const int length = st->context_length;
    const int rows = length - st->kv_from;
    for (size_t i = 0; i < stack.size(); i++) {
        qwen38::decoder_layer& layer = stack[(int)i];
        if (layer.is_full_attention()) {
            if (rows > 0) layer.write_rows(st->kv_from, length, st->kv_k[i].data(), st->kv_v[i].data());
            layer.import_blob(&length);   // the cursor
        } else {
            layer.import_blob(st->state[i].data());
        }
    }
    // rows above the cursor hold what the requests since the state was read left there; the kernels stream them
    _zero_attn_rows(length);
    committed_ids = st->ids;
    mtp_hist      = st->hist;
    mtp_hist_base = st->hist_base;
    // the head's cache and the pending prime belong to the context that was live: the prefill that follows re-arms them
    // and the prime folds every history row back to mtp_hist_base, as it does after a cold prefill
    if (spec_mod) { spec_mod->reset(); spec_mod->zero_stale(); }
    // A saved state carries no target taps: the DFlash2 drafter starts cold and refills from the rows that follow
    if (df) df->reset();
    mtp_pos_base       = 1;
    last_hidden        = nullptr;
    last_window_len    = 0;
    spec_prime_pending = false;
    mrope = qwen38::mrope_cursor{};
    set_context_length(length, st->general_token_idx);
    return this->current_context_length;
}

// ===============================================
// dead ceremony: causal_lm declares these pure virtual, nothing calls them
// ===============================================
buffer<bf16> qwen3_8mtp_npu::Impl::get_k_cache(int layer_idx, int idx) {
    (void)layer_idx; (void)idx;
    return buffer<bf16>();
}

buffer<bf16> qwen3_8mtp_npu::Impl::get_v_cache(int layer_idx, int idx) {
    (void)layer_idx; (void)idx;
    return buffer<bf16>();
}

// ===============================================
// statistics
// ===============================================
std::string qwen3_8mtp_npu::Impl::spec_stats_line() const {
    // Say nothing rather than print a row of zeros that looks like
    // speculation ran and failed.
    if (spec_cycles == 0) return std::string();

    // snprintf, not a stringstream: this shares a translation unit with the
    // decode path and iostream formatting state is global.
    //
    // Two quality numbers and nothing else. Same numerator, different
    // denominators:
    //   accept length -- accepted per CYCLE. How much of a cycle's work the
    //                    head earned. Add 1.0 for the free token to get the
    //                    speedup over plain decode.
    //   hit rate      -- accepted per DRAFT SLOT. How often a slot pays off,
    //                    which is what says whether k is set too deep.
    // Both derive from raw counts, blind to stop tokens; see the commit block
    // in speculate().
    //
    // Ratios only, no counts and no throughput. Everything they are built from
    // -- cycles, draft steps, committed tokens -- is in spec_timing_block()
    // four lines below, so printing it here too said the same thing twice and
    // buried the two numbers this line exists for. Two consequences worth
    // knowing before anyone reads a bug into them:
    //
    //   * Nothing here reconciles against the caller's completion_tokens. The
    //     accepted count is the wrong number to try it with -- it is raw, so a
    //     stop-truncated commit makes it exceed what was emitted. The token
    //     total in the breakdown header is the one that reconciles.
    //   * The hit rate's denominator, k, is not shown. Accept length alone
    //     cannot say whether 1.41 is a weak head or a shallow k; read
    //     draft steps / cycles out of the breakdown for that.
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "%s acceptance length: %.2f, hit rate: %.1f%%",
                  df ? "DFlash" : "MTP",
                  mean_accepted_length(),
                  100.0 * draft_hit_rate());

    std::string s(buf);
    if (spec_declines)
        s += ", " + std::to_string((unsigned long long)spec_declines) + " declined";
    return s;
}

std::string qwen3_8mtp_npu::Impl::spec_timing_block() const {
    if (spec_cycles == 0) return std::string();

    char buf[1536];
    int n = 0;

    n += std::snprintf(buf + n, sizeof buf - n,
        "    %s phase breakdown (%llu cycles, %llu tokens):\n",
        df ? "DFlash" : "MTP",
        (unsigned long long)spec_cycles,
        (unsigned long long)spec_tokens_total);

    // ms everywhere: re_unit() would give each row a different unit, which is
    // what makes a breakdown unreadable.
    // First, and only when it happened: this is the one row that is NOT
    // decode. It is step 0 of the first cycle after each prompt prefill, where
    // the head absorbs the prompt -- prefill-phase work, and the runtime has
    // already moved this many microseconds out of DECODING_TIME and into
    // PREFILL_TIME. Printing it inside the decode breakdown anyway because
    // this is the only place its size is visible.
    if (spec_prime_steps) {
        n += std::snprintf(buf + n, sizeof buf - n,
            "      Prime   (%llu x step 0, %llu rows): %6.1f ms  %5.1f%%  %8.1f ms/row"
            "   [charged to prefill]\n",
            (unsigned long long)spec_prime_steps,
            (unsigned long long)spec_prime_rows,
            (double)spec_prime_us / 1e3, 100.0 * prime_frac(), ms_per_prime_row());
    }

    n += std::snprintf(buf + n, sizeof buf - n,
        df ? "      Draft   (%llu block passes, 5 layers): %5.1f ms  %5.1f%%  %8.1f ms/pass\n"
           : "      Draft   (%llu steps, 1 layer):   %9.1f ms  %5.1f%%  %8.1f ms/step\n",
        (unsigned long long)spec_draft_steps,
        (double)spec_draft_us / 1e3, 100.0 * draft_frac(), ms_per_draft_step());

    n += std::snprintf(buf + n, sizeof buf - n,
        "      Verify  (%llu rows, 64 layers): %9.1f ms  %5.1f%%  %8.1f ms/row\n",
        (unsigned long long)spec_verify_rows,
        (double)spec_verify_us / 1e3, 100.0 * verify_frac(), ms_per_verify_row());

    // Printed even at zero: a zero replay row says every cycle accepted in
    // full, which is a real result.
    n += std::snprintf(buf + n, sizeof buf - n,
        "      Replay  (%llu recoveries):      %9.1f ms  %5.1f%%\n",
        (unsigned long long)spec_replays,
        (double)spec_replay_us / 1e3, 100.0 * replay_frac());

    n += std::snprintf(buf + n, sizeof buf - n,
        "      Cycle total:                   %9.1f ms  %8.1f ms/cycle  %.1f ms/token\n",
        (double)spec_cycle_us / 1e3, cycle_ms(), ms_per_token());

    // ms/token above divides a prompt-sized cost by an output-sized count, so
    // it moves when the PROMPT gets longer. This is the one to quote.
    if (spec_prime_steps) {
        n += std::snprintf(buf + n, sizeof buf - n,
            "        less prime (decode only):    %9.1f ms  %.1f ms/token\n",
            (double)decode_cycle_us() / 1e3, ms_per_token_decode());
    }

    // The payoff line: a verify row went through 64 layers and a draft step
    // through one, so this ratio is the batching win.
    if (ms_per_verify_row() > 0.0) {
        std::snprintf(buf + n, sizeof buf - n,
            "      Amortised verify cost is %.2fx a draft step\n",
            ms_per_verify_row() / (ms_per_draft_step() > 0.0
                                 ? ms_per_draft_step() : 1.0));
    }
    return std::string(buf);
}

void qwen3_8mtp_npu::Impl::report_spec_stats() const {
    const std::string line = spec_stats_line();
    if (!line.empty()) header_print("FLM", line);
}

// ===============================================
// Externals for qwen3_8mtp_npu
// ===============================================
qwen3_8mtp_npu::qwen3_8mtp_npu(LM_Config config, npu_xclbin_manager* npu_instance,
                               int MAX_L) {
    this->_impl = new Impl(config, npu_instance, this, MAX_L);
}

buffer<bf16> qwen3_8mtp_npu::forward(int ids) {
    return this->_impl->forward(ids);
}

buffer<bf16> qwen3_8mtp_npu::prefill(std::vector<int>& ids, void* payload) {
    try {
        return this->_impl->prefill(ids, payload);
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void qwen3_8mtp_npu::set_context_length(int L) {
    std::cout << "Setting context length is not supported for qwen3_8mtp_npu!" << std::endl;
    (void)L;
}

void qwen3_8mtp_npu::load_weights(Q4NX& q4nx) {
    this->_impl->load_weights(q4nx);
}

void qwen3_8mtp_npu::update_max_length(uint32_t MAX_L) {
    this->_impl->update_max_length(MAX_L);
}

uint32_t qwen3_8mtp_npu::max_context_length() const {
    return _usable_context();
}

void qwen3_8mtp_npu::clear_context() {
    this->_impl->clear_context();
}

buffer<bf16> qwen3_8mtp_npu::get_k_cache(int layer_idx, int idx) {
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> qwen3_8mtp_npu::get_v_cache(int layer_idx, int idx) {
    return this->_impl->get_v_cache(layer_idx, idx);
}

int qwen3_8mtp_npu::get_current_context_length() {
    return this->_impl->get_current_context_length();
}

int qwen3_8mtp_npu::checkpoint() {
    return this->_impl->checkpoint();
}

int qwen3_8mtp_npu::restore() {
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> qwen3_8mtp_npu::new_state() {
    return this->_impl->new_state();
}

int qwen3_8mtp_npu::read_state(hybrid_engine_state& state, int kv_from) {
    return this->_impl->read_state(state, kv_from);
}

int qwen3_8mtp_npu::write_state(const hybrid_engine_state& state) {
    return this->_impl->write_state(state);
}

int qwen3_8mtp_npu::write_state_kv(const hybrid_engine_state& state) {
    return this->_impl->write_state_kv(state);
}

extern "C" const flm_state_api* qwen3_8mtp_state_api(void* engine) {
    return hybrid_state_api<qwen3_8mtp_npu>(engine);
}

bool qwen3_8mtp_npu::supports_speculation() const {
    // Not a compile-time property: a checkpoint without the mtp.* tensors runs
    // fine on the plain decode path, and should not pay the branch forever.
    return this->_impl->spec_mod != nullptr || this->_impl->df != nullptr;
}

std::vector<int> qwen3_8mtp_npu::speculate(int last_token, int max_draft) {
    return this->_impl->speculate(last_token, max_draft);
}

void qwen3_8mtp_npu::set_stop_tokens(const std::vector<int>& ids) {
    this->_impl->stop_tokens = ids;
}

void qwen3_8mtp_npu::set_speculation_chooser(std::function<int(int, const float*, int)> chooser) {
    this->_impl->spec_chooser = std::move(chooser);
}

double qwen3_8mtp_npu::mean_accepted_length() const {
    return this->_impl->mean_accepted_length();
}

double qwen3_8mtp_npu::draft_hit_rate() const {
    return this->_impl->draft_hit_rate();
}

uint64_t qwen3_8mtp_npu::speculation_cycles() const {
    return this->_impl->spec_cycles;
}

std::string qwen3_8mtp_npu::speculation_stats() const {
    return this->_impl->spec_stats_line();
}

void qwen3_8mtp_npu::report_speculation_stats() const {
    this->_impl->report_spec_stats();
}

std::string qwen3_8mtp_npu::speculation_timing() const {
    return this->_impl->spec_timing_block();
}

double qwen3_8mtp_npu::speculation_phase_fraction() const {
    return this->_impl->draft_frac() + this->_impl->verify_frac() +
           this->_impl->replay_frac();
}

void qwen3_8mtp_npu::reset_speculation_stats() {
    this->_impl->spec_cycles         = 0;
    this->_impl->spec_accepted_total = 0;
    this->_impl->spec_drafted_total  = 0;
    this->_impl->spec_declines       = 0;
    this->_impl->spec_tokens_total   = 0;
    this->_impl->spec_draft_us       = 0;
    this->_impl->spec_verify_us      = 0;
    this->_impl->spec_replay_us      = 0;
    this->_impl->spec_cycle_us       = 0;
    this->_impl->spec_draft_steps    = 0;
    this->_impl->spec_verify_rows    = 0;
    this->_impl->spec_replays        = 0;
    this->_impl->spec_last_replay_us = 0;
    this->_impl->spec_prime_us       = 0;
    this->_impl->spec_prime_steps    = 0;
    this->_impl->spec_prime_rows     = 0;
    this->_impl->spec_last_prime_us  = 0;
    // NOT spec_prime_pending: that is live state, not a statistic. A reset
    // between turns happens after the prefill has already armed it, and
    // clearing it here would charge that turn's catch-up to decode.
}

uint64_t qwen3_8mtp_npu::last_speculation_prime_us() const {
    return this->_impl->spec_last_prime_us;
}

const float* qwen3_8mtp_npu::last_speculation_logits(int& rows, int& vocab) const {
    rows  = 0;
    vocab = 0;
    const Impl* impl = this->_impl;
    if (!impl->verify_mod || impl->spec_logit_rows <= 0) return nullptr;
    int verify_rows = 0;
    const float* logits = impl->verify_mod->logits(verify_rows, vocab);
    if (!logits || verify_rows < impl->spec_logit_rows) {
        vocab = 0;
        return nullptr;
    }
    rows = impl->spec_logit_rows;
    return logits;
}

const float* qwen3_8mtp_npu::last_hidden_states(int& L, int& D) const {
    return this->_impl->last_hidden_states(L, D);
}

const float* qwen3_8mtp_npu::last_normed_states(int& L, int& D) {
    return this->_impl->last_normed_states(L, D);
}

std::vector<int> qwen3_8mtp_npu::mtp_draft(const std::vector<int>& ids,
                                           const float* prev_hidden, int T, int k) {
    return this->_impl->mtp_draft(ids, prev_hidden, T, k);
}

int qwen3_8mtp_npu::prefill_verify(const std::vector<int>& verify_ids,
                                   const std::vector<int>& drafts,
                                   std::vector<int>& out_argmax) {
    return this->_impl->prefill_verify(verify_ids, drafts, out_argmax);
}

bool qwen3_8mtp_npu::has_mtp_head() const {
    // "the speculation head": the DFlash2 drafter stands in for the MTP head when it is loaded
    return this->_impl->spec_mod != nullptr || this->_impl->df != nullptr;
}

// ---- module 4 --------------------------------------------------------------
bool qwen3_8mtp_npu::has_vision_tower() const {
    return this->_impl->image_mod != nullptr;
}

int qwen3_8mtp_npu::image_tokens_for(int grid_t, int grid_h, int grid_w) const {
    // From the desc, not the module: the answer is pure geometry, and a caller
    // sizing its prompt must get it before load_weights() has run.
    return this->_impl->vision_desc.token_count(grid_t, grid_h, grid_w);
}

void qwen3_8mtp_npu::encode_image(const float* pixel_values, int grid_t,
                                  int grid_h, int grid_w, float* out) {
    this->_impl->encode_image(pixel_values, grid_t, grid_h, grid_w, out);
}

std::string qwen3_8mtp_npu::vision_backend() const {
    return this->_impl->image_mod ? this->_impl->image_mod->backend()
                                  : std::string();
}

void qwen3_8mtp_npu::set_vision_tap(
        std::function<void(const char*, const float*, int, int)> fn) {
    if (this->_impl->image_mod) this->_impl->image_mod->set_tap(std::move(fn));
}

qwen3_8mtp_npu::~qwen3_8mtp_npu() {
    delete this->_impl;
}
