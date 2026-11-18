// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// \file common/layer_stack.hpp
/// \brief The 64-layer hybrid stack: weights, KV caches and delta-net state.
#pragma once

#include "common/layer.hpp"
#include "prefill/npu/delta_net_op.hpp"
#include "prefill/npu/delta_proj.hpp"
#include "prefill/npu/prefill_attn_op.hpp"
#include "prefill/npu/prefill_proj.hpp"
#include "qwen3_8mtp_model_def.hpp"
#include "verify/npu/attn_layer.hpp"
#include "verify/npu/delta_verify_layer.hpp"

#include "utils/debug_utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen38 {

class layer_stack {
public:
    /// `layers_` is declared first and so is destroyed *last*, after the
    /// engines its NPU layers point at. That is the wrong order for
    /// `npu_delta_layer`, whose destructor joins a state replay that reads its
    /// engine -- so retire the layers explicitly, first.
    ~layer_stack() { layers_.clear(); }

    /// Builds all 64 layers on the device. **There is no host form of a layer
    /// any more**, so every path below is required and a missing file throws
    /// here rather than degrading. They are separate parameters because they
    /// are separate bitstreams staged separately, and the message has to name
    /// which one is absent.
    ///
    /// \param npu   required; the engine refuses a null device in load_weights.
    /// RTP addresses are per-build: if the xclbin is rebuilt, diff its
    /// rtp_addresses.txt against q38npu::attn_rtp_addresses. A stale address
    /// does not fault, it hangs in wait().
    /// \param prefill_xclbin_path  attn.xclbin, the prefill attention core.
    /// \param proj_xclbin_path  dequant_mm.xclbin, the fused Q4_K dequant +
    ///        GEMM engine that runs the projections. Reachable only underneath
    ///        the one above: the projections feed the attention core, so there
    ///        is no configuration where they run and it does not.
    /// \param deltanet_xclbin_path  GateDeltaNet_prefill.xclbin, the gated
    ///        delta rule. \param conv_xclbin_path  conv.xclbin, the depthwise
    ///        conv1d + SiLU + the q/k l2norms. Both are needed or neither runs:
    ///        the conv feeds the fold's input BO and applies the norms the fold
    ///        assumes. Independent of the two attention paths, and reachable
    ///        only underneath the delta projections, which stage its input.
    /// \param delta_verify_xclbin_path  deltanet_layer.xclbin, the whole
    ///        gated-delta-net block for the L = 8 verify pass -- the six
    ///        projections, the conv, the fold, the gate and the MLP residual in
    ///        one dispatch. Reachable only underneath the delta projections,
    ///        whose staged BO it reads (including the a/b tail past
    ///        `proj_bytes`, which exists for this bitstream alone). Unlike the
    ///        three above it changes what the 48 layers *are* rather than how
    ///        one of their stages runs, so it is built before `layers_` is
    ///        populated.
    void build(const cpu_ref::safetensors& w, const qwen3_8mtp_desc& desc,
               int max_len, npu_xclbin_manager* npu,
               const std::string& xclbin_path,
               const std::string& prefill_xclbin_path,
               const std::string& proj_xclbin_path,
               const std::string& deltanet_xclbin_path,
               const std::string& conv_xclbin_path,
               const std::string& delta_verify_xclbin_path) {
        layers_.clear();
        npu_attn_.clear();
        attn_halves_.clear();
        delta_halves_.clear();
        npu_delta_.clear();
        delta_w_.clear();
        delta_ssm_.clear();

        if (!npu) throw std::runtime_error("layer_stack::build: no device");
        npu_app_manager* mgr = npu->register_xclbin(xclbin_path);
        attn_eng_ = std::make_unique<q38npu::attn_engine>(
            mgr->create_app(),
            q38npu::attn_layout::build(max_len, /*L=*/8),
            q38npu::attn_rtp_addresses{});

        layers_.reserve(desc.layer_types.size());
        npu_attn_.assign(desc.layer_types.size(), nullptr);
        attn_halves_.resize(desc.layer_types.size());
        delta_halves_.assign(desc.layer_types.size(), nullptr);
        npu_delta_.assign(desc.layer_types.size(), nullptr);

        // Before the loop, and this is the one ordering constraint in build():
        // a delta-net layer's 212 MB projection BO is a *constructor argument*
        // to the NPU form of that layer, so it has to exist before the layer
        // does. Staging one per layer inside the loop instead would be the same
        // 10.66 GB a second time.
        _build_delta_proj(w, desc, npu, proj_xclbin_path);
        _build_delta_verify(desc, npu, delta_verify_xclbin_path);

        // Every layer is a device layer. The two host adapters this used to
        // fall back to are gone: `prefill_cpu` refused a stack containing one
        // anyway, so the branches were only ever reachable as a slower, wronger
        // way to reach that throw.
        for (size_t i = 0; i < desc.layer_types.size(); i++) {
            if (check_layer_type(desc.layer_types[i], i)) {
                auto* l = new q38npu::npu_attn_layer(attn_eng_.get(), w, (int)i);
                npu_attn_[i] = l;
                layers_.emplace_back(l);
            } else {
                auto* l = new q38npu::npu_delta_layer(delta_verify_eng_.get(), w,
                                                      desc.delta, (int)i,
                                                      delta_w_[i].get());
                npu_delta_[i] = l;
                delta_halves_[i] = &l->inner();
                layers_.emplace_back(l);
            }
        }

        _build_prefill_attn(w, npu, prefill_xclbin_path);
        _build_prefill_proj(npu, proj_xclbin_path);
        _build_delta_net(desc, npu, deltanet_xclbin_path, conv_xclbin_path);
    }

    /// The shared prefill attention core. Never null on a built stack -- a
    /// missing bitstream throws in `build()`. Module 1 requires it; module 3
    /// does not use it.
    q38npu::prefill_attn_op* prefill_attn() { return prefill_op_.get(); }

    /// The shared prefill projections. Never null on a built stack, on the
    /// same terms.
    q38npu::prefill_proj* prefill_proj() { return prefill_proj_.get(); }

    /// The NPU attention layer at `i`, or null if layer `i` is a delta-net
    /// block. Non-owning -- `layers_` owns it.
    q38npu::npu_attn_layer* npu_attn(int i) {
        return (size_t)i < npu_attn_.size() ? npu_attn_[(size_t)i] : nullptr;
    }

    /// The CPU half of a prefill attention layer: a `cpu_ref::attention_layer`
    /// built with **max_len = 0**, so it carries the weights and none of the
    /// 32 MB of cache. Only the four stage helpers module 1 still runs on the
    /// host are ever called -- `in_norm`, `qkv_post_rope`, `apply_gate`,
    /// `post_norm` -- and none of them touches the cache. `forward()` on one of
    /// these would throw, which is the intent: the cache that layer `i` shares
    /// with the device lives in `npu_attn(i)`, not here.
    cpu_ref::attention_layer* attn_half(int i) {
        return (size_t)i < attn_halves_.size() ? attn_halves_[(size_t)i].get() : nullptr;
    }

    /// The shared delta-net projections. Independent of `prefill_proj()`: a
    /// delta layer has no attention core to sit around. Never null on a built
    /// stack -- it is the first thing `build()` makes.
    q38npu::delta_proj* delta_proj() { return delta_proj_.get(); }

    /// Layer `i`'s staged 211 MB projection BO, or null if that layer is not a
    /// delta-net block (or there is no device).
    q38npu::npu_delta_weights* delta_weights(int i) {
        return (size_t)i < delta_w_.size() ? delta_w_[(size_t)i].get() : nullptr;
    }

    /// The `cpu_ref::delta_net_layer` inside layer `i`, or null if that layer
    /// is not a delta-net block. **Not a weights-only companion** -- unlike
    /// `attn_half`, this is the layer itself, because a delta-net block's
    /// weights and its recurrent state live in one object and the hybrid path
    /// needs both. Non-owning; `layers_` owns it.
    /// When layer `i` is on `deltanet_layer.xclbin` this goes through the NPU
    /// layer's `inner()` rather than straight to the pointer, which **joins its
    /// pending state replay first**. The hybrid prefill path mutates what this
    /// returns -- `prefill_verify()` rolls back and re-runs the accepted prefix
    /// through it -- and a fold left running from the last verify pass would
    /// otherwise be writing into the same state at the same time.
    cpu_ref::delta_net_layer* delta_half(int i) {
        if ((size_t)i >= delta_halves_.size()) return nullptr;
        if (npu_delta_[(size_t)i]) return &npu_delta_[(size_t)i]->inner();
        return delta_halves_[(size_t)i];
    }

    /// The NPU delta-net layer at `i`, or null if that layer is not one (or the
    /// block stays on the host). Non-owning -- `layers_` owns it.
    q38npu::npu_delta_layer* npu_delta(int i) {
        return (size_t)i < npu_delta_.size() ? npu_delta_[(size_t)i] : nullptr;
    }

    /// The shared conv1d + delta-rule pair. Never null on a built stack; the
    /// projections, which write its input BO, are built before it.
    q38npu::delta_net_op* delta_net() { return delta_net_.get(); }

    /// Layer `i`'s 80 KB conv-weight BO, or null if that layer is not a
    /// delta-net block (or there is no device).
    q38npu::npu_delta_ssm_weights* delta_ssm(int i) {
        return (size_t)i < delta_ssm_.size() ? delta_ssm_[(size_t)i].get() : nullptr;
    }

    bool   empty() const { return layers_.empty(); }
    size_t size()  const { return layers_.size(); }
    decoder_layer& operator[](int i) { return *layers_[(size_t)i]; }

    void reset() { for (auto& l : layers_) l->reset(); }
    /// Grows every layer, then re-hands the grown layout to the prefill op.
    /// It owns no BO sized by MAX_L, but MAX_L *is* the plane pitch and so is
    /// where the v plane starts -- skipping this reads every v row out of the
    /// old k plane's tail, which cosine on the output notices only faintly.
    void set_max_length(int n) {
        for (auto& l : layers_) l->set_max_length(n);
        if (prefill_op_ && attn_eng_) prefill_op_->set_layout(attn_eng_->t);
    }

    void save_state(state_slot s)    { for (auto& l : layers_) l->save_state(s); }
    void restore_state(state_slot s) { for (auto& l : layers_) l->restore_state(s); }

    /// Whether every layer can re-derive an accepted prefix in place. Asked of
    /// all 64 and not just one: the stack has to agree on how far it advanced,
    /// so a single delta-net block left on the host sends the whole stack down
    /// the re-prefill path.
    bool can_commit_prefix() const {
        if (layers_.empty()) return false;
        for (const auto& l : layers_) if (!l->can_commit_prefix()) return false;
        return true;
    }

    /// Advance every layer from `s` by `accept_len` of the last window's
    /// tokens. Check `can_commit_prefix()` first; a layer that cannot throws.
    void commit_prefix(state_slot s, int accept_len) {
        for (auto& l : layers_) l->commit_prefix(s, accept_len);
    }

    /// One slot's shadow footprint: ~157 MB, nearly all delta-net.
    size_t state_bytes() const {
        size_t n = 0;
        for (const auto& l : layers_) n += l->state_bytes();
        return n;
    }

    /// Every attention layer must agree on how many rows it holds; a
    /// divergence means a layer was skipped or double-fed. -1 if there are none.
    int attention_cache_len() const {
        int seen = -1;
        for (const auto& l : layers_) {
            if (!l->is_full_attention()) continue;
            const int n = l->tokens_seen();
            if (seen < 0) seen = n;
            else if (n != seen)
                throw std::runtime_error("attention layers disagree on cache length: " +
                                         std::to_string(seen) + " vs " + std::to_string(n));
        }
        return seen;
    }

private:
    /// Register attn.xclbin and build the CPU halves. Split out for the message
    /// rather than for a fallback: the prefill bitstream is staged separately
    /// from the verify one, so a missing file here is the likeliest single
    /// failure and it should say which file.
    ///
    /// **A missing bitstream throws, here and in the two below.** Module 1 used
    /// to degrade to the host, which was defensible while both paths existed;
    /// with the host path gone there is nothing to degrade to, and a stale or
    /// unstaged xclbin is a deployment bug rather than a slower configuration.
    /// This is the failure mode the stale staged `lm_head_8.xclbin` already
    /// cost us once, and a throw is what would have caught it that day.
    ///
    /// The `DEBUG_BLOCK(2, ...)` confirmations stay. They are the record of
    /// what actually got wired -- only interesting when you are reading the
    /// whole startup, which is why they are not level 1.
    void _build_prefill_attn(const cpu_ref::safetensors& w,
                             npu_xclbin_manager* npu,
                             const std::string& path) {
        if (!std::filesystem::exists(path))
            throw std::runtime_error("layer_stack: no attention_prefill.xclbin (" +
                                     path + ") -- module 1's attention core has "
                                     "no host path");

        npu_app_manager* mgr = npu->register_xclbin(path);
        prefill_op_ = std::make_unique<q38npu::prefill_attn_op>(
            mgr->create_app(), attn_eng_->t, q38npu::prefill_attn_rtp_addresses{});

        // max_len = 0: weights only. See attn_half().
        for (size_t i = 0; i < npu_attn_.size(); i++)
            if (npu_attn_[i])
                attn_halves_[i] =
                    std::make_unique<cpu_ref::attention_layer>(w, (int)i, /*max_len=*/0);

        DEBUG_BLOCK(2,
            header_print("FLM", "prefill attention on the NPU (attn.xclbin, one "
                                "dispatch per layer, L padded to " +
                                std::to_string(q38npu::prefill_attn_op::CHUNK) + ")");
        );
    }

    /// Register dequant_mm.xclbin and build the four projection ops.
    ///
    /// Four apps, one per dispatch, because the instruction sequence is baked
    /// from the shape and the weight offset and an app owns exactly one. They
    /// all share this xclbin's hw_context; `register_xclbin` dedupes by name,
    /// so when the draft head has already registered the same file this takes
    /// four more apps from the manager it built rather than a second context.
    ///
    /// Throws on a missing file, like `_build_prefill_attn`.
    ///
    /// No layout adoption on `set_max_length`, unlike `prefill_op_`: the four
    /// weight offsets `attn_layout::build` computes are functions of the tensor
    /// shapes alone. Only the cache geometry moves with max_len, and this
    /// object never touches the cache.
    void _build_prefill_proj(npu_xclbin_manager* npu, const std::string& path) {
        if (!std::filesystem::exists(path))
            throw std::runtime_error("layer_stack: no dequant_mm.xclbin (" + path +
                                     ") -- the 16 attention layers' projections "
                                     "have no host path");

        npu_app_manager* mgr = npu->register_xclbin(path);
        prefill_proj_ = std::make_unique<q38npu::prefill_proj>(
            mgr, attn_eng_->t, q38npu::dequant_mm_rtp_addresses{});

        DEBUG_BLOCK(2,
            header_print("FLM", "the 16 attention layers' projections on the NPU "
                                "(dequant_mm.xclbin, 4 dispatches per 512-row "
                                "tile)");
        );
    }

    /// Register dequant_mm.xclbin and stage all 48 delta-net weight BOs.
    ///
    /// **This is where the 10.62 GB and the load-time reorder pass are paid**
    /// -- 221,237,512 B per layer, one `reorder_cpy` sweep each, straight out
    /// of the mmap. The alternative the user weighed and declined was a
    /// rotating restaged pool, which trades all of that resident memory for the
    /// same permutation re-run on every prefill window, forever.
    ///
    /// Independent of `prefill_op_` and of `attn_eng_`, unlike
    /// `_build_prefill_proj`: the delta path replaces matmuls with matmuls and
    /// has no attention core it has to sit around.
    ///
    /// Throws on a missing file, like the two above. It is the same file as
    /// `_build_prefill_proj`'s, so in practice they fail together; the two
    /// messages differ because the two phases are staged independently and
    /// either could in principle be the one that moved.
    ///
    /// No layout adoption on `set_max_length`: `delta_layout::build`'s offsets
    /// are functions of the tensor shapes alone, and a delta-net block's state
    /// is a fixed-size fold rather than a growing cache.
    void _build_delta_proj(const cpu_ref::safetensors& w,
                           const qwen3_8mtp_desc& desc,
                           npu_xclbin_manager* npu, const std::string& path) {
        if (!std::filesystem::exists(path))
            throw std::runtime_error("layer_stack: no dequant_mm.xclbin (" + path +
                                     ") -- the 48 delta-net layers' projections "
                                     "have no host path (and the recurrence reads "
                                     "their BO, so it cannot run either)");

        const q38npu::delta_layout t = q38npu::delta_layout::build(desc.delta);
        npu_app_manager* mgr = npu->register_xclbin(path);
        delta_proj_ = std::make_unique<q38npu::delta_proj>(
            mgr, t, q38npu::dequant_mm_rtp_addresses{});

        // One app for all 48 BOs. BOs are device-scoped -- create_bo_buffer
        // builds a buffer<T>(*device, size) -- so the app a BO is allocated
        // from need not be the one that dispatches against it, and taking 48
        // more apps here would be 48 more instruction sequences for nothing.
        // The layer type, not `delta_halves_`: this runs before the layer loop
        // (the NPU form of a delta layer takes its BO as a constructor
        // argument), so nothing has registered a half yet.
        npu_app staging = mgr->create_app();
        delta_w_.resize(desc.layer_types.size());
        for (size_t i = 0; i < desc.layer_types.size(); i++)
            if (!check_layer_type(desc.layer_types[i], i))
                delta_w_[i] = std::make_unique<q38npu::npu_delta_weights>(
                    staging, t, w, (int)i);

        DEBUG_BLOCK(2,
            header_print("FLM", "the 48 delta-net layers' projections on the NPU "
                                "(dequant_mm.xclbin, 9.98 GiB resident)");
        );
    }

    /// Register deltanet_layer.xclbin and build the shared verify engine.
    ///
    /// This decides what the 48 delta layers *are*, so it runs before `layers_`
    /// is populated and cannot be turned on afterwards. It used to be the one
    /// path here that degraded -- `FLM_Q38_DELTA_VERIFY` chose between this and
    /// a host delta-net layer, a real throughput A/B worth 20% of the verify
    /// pass -- and it throws now, like everything else in this file. The A/B it
    /// supported is a job for a driver, not a shipping env var: both arms
    /// produced fluent text, so a silent degrade presented as nothing but a
    /// slower verify, which nobody traced back to a file that was not staged.
    /// `tools/verify_n12_delta_verify` is where that comparison lives now, and
    /// it scores against `cpu_ref::delta_net_layer` directly.
    ///
    /// Underneath `_build_delta_proj` because the layer reads that phase's
    /// staged BO, including the a/b tail past `proj_bytes` that exists for this
    /// bitstream alone. Nothing is restaged here: the two phases share the
    /// 10.66 GB rather than each holding their own copy, which is the whole
    /// reason the tail was appended to the prefill BO instead of given its own.
    ///
    /// No `set_max_length` hook, for the same reason `_build_delta_proj` has
    /// none: a delta-net block's history is a fixed-size fold and a fixed-size
    /// conv window, so nothing here is a function of the context length.
    void _build_delta_verify(const qwen3_8mtp_desc& desc,
                             npu_xclbin_manager* npu, const std::string& path) {
        if (!std::filesystem::exists(path))
            throw std::runtime_error("layer_stack: no deltanet_layer.xclbin (" +
                                     path + ") -- the 48 delta-net layers have "
                                     "no host form");

        npu_app_manager* mgr = npu->register_xclbin(path);
        delta_verify_eng_ = std::make_unique<q38npu::delta_engine>(
            mgr->create_app(),
            q38npu::delta_verify_layout::build(desc.delta, /*L=*/8));

        DEBUG_BLOCK(2,
            header_print("FLM", "the 48 delta-net layers on the NPU whole "
                                "(deltanet_layer.xclbin, one dispatch per layer "
                                "per 8 rows; the state replay overlaps the next "
                                "layer's)");
        );
    }

    /// Register the two recurrence bitstreams and stage the 48 conv-weight BOs.
    ///
    /// Two xclbins, two managers, one op -- the conv is one dispatch per layer
    /// per window and the fold is one per 1024-token chunk, and they chain
    /// through a BO rather than through the host.
    ///
    /// Underneath `_build_delta_proj` because `delta_proj::qkv_z_to_bo` is what
    /// fills the conv's input BO. Without the projections on the device the qkv
    /// rows are fp32 on the host and would have to be converted and copied in,
    /// which is most of what moving the conv saves.
    ///
    /// Throws on a missing file, like the three above. 3.8 MB of conv weights
    /// across the 48 layers -- three orders below the projection BOs, so
    /// nothing here is worth a pool.
    void _build_delta_net(const qwen3_8mtp_desc& desc, npu_xclbin_manager* npu,
                          const std::string& deltanet_path,
                          const std::string& conv_path) {
        // Checked separately: these are two files and either one missing costs
        // the same thing, so a message that said only "the recurrence" would
        // leave you diffing the xclbins directory to find out which.
        if (!std::filesystem::exists(deltanet_path))
            throw std::runtime_error("layer_stack: no GateDeltaNet_prefill.xclbin (" +
                                     deltanet_path + ") -- the 48 delta-net "
                                     "layers' fold has no host path");
        if (!std::filesystem::exists(conv_path))
            throw std::runtime_error("layer_stack: no conv.xclbin (" + conv_path +
                                     ") -- the 48 delta-net layers' conv1d has no "
                                     "host path");

        npu_app_manager* dn_mgr   = npu->register_xclbin(deltanet_path);
        npu_app_manager* conv_mgr = npu->register_xclbin(conv_path);
        delta_net_ = std::make_unique<q38npu::delta_net_op>(
            conv_mgr->create_app(), dn_mgr, desc.delta,
            q38npu::conv1d_rtp_addresses{});

        // One app for all 48 BOs, as _build_delta_proj does and for the same
        // reason: a BO binds to the device, not to a hardware context.
        npu_app staging = conv_mgr->create_app();
        delta_ssm_.resize(delta_halves_.size());
        for (size_t i = 0; i < delta_halves_.size(); i++)
            if (delta_halves_[i])
                delta_ssm_[i] = std::make_unique<q38npu::npu_delta_ssm_weights>(
                    staging, *delta_halves_[i]);

        DEBUG_BLOCK(2,
            header_print("FLM", "the 48 delta-net layers' recurrence on the NPU "
                                "(conv1d + GateDeltaNet_prefill, L padded to " +
                                std::to_string(q38npu::delta_net_op::MT_L) +
                                ", windows of " +
                                std::to_string(q38npu::delta_net_op::MAX_WINDOW /
                                               q38npu::delta_net_op::MT_L *
                                               q38npu::delta_net_op::MT_L) + ")");
        );
    }

    std::vector<std::unique_ptr<decoder_layer>> layers_;
    /// Shared by all 16 NPU attention layers; must outlive them.
    std::unique_ptr<q38npu::attn_engine>        attn_eng_;
    /// Shared by all 16 as well, and holds no per-layer state of its own.
    /// Never null on a built stack.
    std::unique_ptr<q38npu::prefill_attn_op>    prefill_op_;
    /// Shared by all 16 too: the weight BO is a run() argument, so what is
    /// per-layer is the buffer and what is per-shape is the app. Null on the
    /// same terms as `prefill_op_`.
    std::unique_ptr<q38npu::prefill_proj>       prefill_proj_;
    /// Non-owning, parallel to layers_; null wherever the layer is not an NPU
    /// attention block.
    std::vector<q38npu::npu_attn_layer*>        npu_attn_;
    /// The weights-only CPU companions, parallel to layers_.
    std::vector<std::unique_ptr<cpu_ref::attention_layer>> attn_halves_;
    /// Shared by all 48 delta-net layers, like prefill_proj_ above and for the
    /// same reason: the weight BO is a run() argument. Null on the same terms.
    std::unique_ptr<q38npu::delta_proj>                    delta_proj_;
    /// Per-layer weight BOs, parallel to layers_; null wherever the layer is
    /// not a delta-net block. 10.62 GB across the 48, and shared: the prefill
    /// projections and the verify bitstream dispatch against the same bytes.
    std::vector<std::unique_ptr<q38npu::npu_delta_weights>> delta_w_;
    /// Shared by all 48 whole-block verify layers; must outlive them, and is
    /// what makes them NPU layers at all -- see build().
    std::unique_ptr<q38npu::delta_engine>                   delta_verify_eng_;
    /// Non-owning, parallel to layers_; null wherever the layer is not a
    /// delta-net block on the NPU.
    std::vector<q38npu::npu_delta_layer*>                   npu_delta_;
    /// Shared by all 48, like delta_proj_: the conv weights and the state are
    /// both run() arguments. Null on the same terms.
    std::unique_ptr<q38npu::delta_net_op>                   delta_net_;
    /// Per-layer conv-weight BOs, parallel to layers_. 80 KB each.
    std::vector<std::unique_ptr<q38npu::npu_delta_ssm_weights>> delta_ssm_;
    /// Non-owning, parallel to layers_. The delta-net layer *itself*, not a
    /// weights-only half -- see delta_half().
    std::vector<cpu_ref::delta_net_layer*>                 delta_halves_;
};

} // namespace qwen38
