// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hunyuan_npu.cpp
/// \brief engine for the hunyuan-dense architecture (Hy-MT2-1.8B)
/// \author FastFlowLM Team
/// \date 2026-09-01
/// \note CPU-first bring-up: the whole graph -- projections, rope, QK norm,
///       attention, GLU, lm_head -- runs on the host out of cpu_functions.hpp,
///       reading the exact reordered q4_k bytes the array would read. See
///       hunyuan_prefill.hpp for the HUNYUAN_*_ON_NPU switches and
///       hunyuan_npu_sequence.cpp for the (throwing) instruction generators.
#include <omp.h>
#include "hunyuan_npu_detail.hpp"
#include "metrices.hpp"
#include "utils/error_measure.hpp"

/// @brief kv cache rows for a context of @p MAX_L tokens: whole prefill grains
///        plus two spare grains for the prefill attention's read past the end
static uint32_t kv_cache_rows(uint32_t MAX_L){
    const uint32_t grain = hunyuan_npu_sequence::PREFILL_L_GRAIN;
    const uint32_t whole_grains = (MAX_L + grain - 1) / grain;
    return (whole_grains + 2) * grain;
}

hunyuan_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, hunyuan_npu* parent_ptr, int MAX_L)
    : config(config), npu(npu_instance), parent_ptr(parent_ptr){
    current_context_length = 0;
    // build the weight-shape definition (dimensions + per-layer weight descriptors)
    desc.build(config);

    vocab_size = desc.vocab_size;
    vocab_size_padded = desc.vocab_size_padded;

    // see "Long context" in README.md
    this->MAX_L = (uint32_t)MAX_L;

    this->sequence = std::make_unique<hunyuan_npu_sequence>(this->desc, config, kv_cache_rows(this->MAX_L));
    this->K03_offset = this->sequence->get_k03_offset();
    this->K47_offset = this->sequence->get_k47_offset();
    this->V03_offset = this->sequence->get_v03_offset();
    this->V47_offset = this->sequence->get_v47_offset();

    // one bitstream covers both the decode layer and the lm_head: they are the
    // same 16 proj cores + rms core, told apart by the is_attn RTP.
#if HUNYUAN_DECODE_ON_NPU || HUNYUAN_LM_HEAD_ON_NPU
    layer_app_manager = npu->register_xclbin(utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin"));
#endif
#if HUNYUAN_DECODE_ON_NPU
    for (int s = 0; s < 2; s++){
        this->decode_slots[s].layer = layer_app_manager->create_app();
        this->decode_slots[s].set_layer_rtp = layer_app_manager->create_app();
        if (!this->npu->is_preemption_enabled()){
            this->decode_slots[s].run = layer_app_manager->create_runlist();
#if HUNYUAN_LM_HEAD_ON_NPU
            this->decode_slots[s].run_logits = layer_app_manager->create_runlist();
#endif
        }
        this->decode_slots[s].L = -1;
    }
    this->active_slot = 0;
#endif
#if HUNYUAN_LM_HEAD_ON_NPU
    this->lm_head = layer_app_manager->create_app();
#endif

    // allocate all buffers. Everything is a device BO even while the graph runs
    // on the host: create_bo_buffer needs no registered xclbin, and allocating
    // this way now means flipping a stage to the array later changes no
    // allocation and no sync_* call.
    proj_weights.resize(desc.num_hidden_layers);
    rms_weights.resize(desc.num_hidden_layers);
    rope_rms_weights.resize(desc.num_hidden_layers);
    kv_caches.resize(desc.num_hidden_layers);

#if HUNYUAN_DECODE_ON_NPU
    for (int s = 0; s < 2; s++){
        this->decode_slots[s].rope.resize(desc.num_hidden_layers);
    }
#endif

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->proj_weights[i] = npu->create_bo_buffer<u8>(desc.total_proj_weights_byte_size);
        this->rms_weights[i] = npu->create_bo_buffer<bf16>(desc.rms_weights_byte_size / sizeof(bf16));
        this->rope_rms_weights[i] = npu->create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
#if HUNYUAN_DECODE_ON_NPU
        for (int s = 0; s < 2; s++){
            this->decode_slots[s].rope[i] = npu->create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
            this->decode_slots[s].rope[i].memset((bf16)0);
        }
#endif
        this->kv_caches[i] = npu->create_bo_buffer<bf16>(this->sequence->kv_buffer_size);
        this->kv_caches[i].memset((bf16)0);
        this->kv_caches[i].sync_to_device();
    }
    this->bfp16_chunks = 0;

    is_checkpoint_valid = false;
    x = npu->create_bo_buffer<bf16>(desc.D);
    final_rms_weight = npu->create_bo_buffer<bf16>(desc.D);
    decode_hidden = buffer<bf16>(desc.D);
    lm_head_weights = npu->create_bo_buffer<u8>(desc.get_lm_head_w_size());
    // the checkpoint's lm_head is padded up to a whole 32-row block but not up
    // to vocab_size_padded, so the tail bands are never written -- zero them so
    // they dequantize to a zero scale rather than garbage.
    memset(lm_head_weights.data(), 0, lm_head_weights.size());
    logits = npu->create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);

    this->clear_context();

    pctx = std::make_unique<hunyuan_prefill_context>(npu, &desc, sequence.get(), config);

#if HUNYUAN_LM_HEAD_ON_NPU
    this->sequence->gen_lm_head_seq(lm_head.seq(), desc.vocab_size_padded, desc.D);
    this->lm_head_run = lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight);
#endif

    set_context_length(0);
}

void hunyuan_npu::Impl::load_weights(Q4NX& q4nx){
    if (!q4nx.has_tensor("model.embed_tokens.weight")){
        // the hunyuan converter always emits the bf16 embedding table alongside
        // the quantized lm_head, so there is no tied-weight fallback here.
        throw std::runtime_error("hunyuan_npu: checkpoint has no model.embed_tokens.weight");
    }
    this->embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);
    this->embedding->init_weights(&q4nx, "model.embed_tokens");

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        desc.load_layer_weights(i, q4nx, this->proj_weights[i], this->rms_weights[i], this->rope_rms_weights[i]);
    }

    {
        buffer<bf16> final_rms_norm_weight;
        q4nx.load_weights(final_rms_norm_weight, "model.norm.weight");
        memcpy(this->final_rms_weight.data(), final_rms_norm_weight.data(), desc.D * sizeof(bf16));
        this->final_rms_weight.sync_to_device();
    }
    {
        buffer<u8> lm_head_w;
        q4nx.load_weights(lm_head_w, desc.lm_head.name);

        // the lm_head runs on the decode layer's own bitstream: Q8_0 halves padded
        // to q4_k chunks, then the same row interleave as the proj weights
        desc.reorder_cpy_lm_head(this->lm_head_weights.data(), lm_head_w);
        desc.lm_head.load();
        this->lm_head_weights.sync_to_device();
    }

#if HUNYUAN_DECODE_ON_NPU
    // q_norm / k_norm live in the upper half of the rope buffer, so each slot
    // needs its own copy of what load_layer_weights just wrote
    this->_sync_slot_rope_weights();
    this->_invalidate_slots();
#endif

    this->clear_context();
}

/// @brief cos/sin table for the decode-side rope tile
/// @note  Unused by the host path: cpu_func::_rope_then_qk_rms_batch computes
///        cos/sin in fp32 from desc.inv_freq at the absolute token position.
void hunyuan_npu::Impl::_set_rope_rms_weights(int idx, std::vector<buffer<bf16>>& rope){
    int RoPE_DIM = desc.RoPE_DIM;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        bf16* w_rope_ptr = rope[i].data();
        for (int j = 0; j < RoPE_DIM / 2; j++){
            w_rope_ptr[j]                = (bf16)std::cos(desc.inv_freq[j] * idx);
            w_rope_ptr[RoPE_DIM / 2 + j] = (bf16)std::sin(desc.inv_freq[j] * idx);
        }
        rope[i].sync_to_device();
    }
}

#if HUNYUAN_DECODE_ON_NPU
/// @brief move the engine to context length @p L on @p slot: build the slot's
///        sequences, runlist and rope table if it does not already carry L,
///        then adopt it as the active slot and set current_context_length.
/// @note  The build and the context-length bookkeeping are the same step. All
///        three artefacts depend on L alone, so a slot that already carries L
///        only needs adopting -- which is exactly what the token after a
///        ping-pong build hits.
void hunyuan_npu::Impl::_build_slot(int slot, int L){
    decode_slot_t& s = this->decode_slots[slot];
    if (s.L != L){
        this->sequence->gen_layer_rtp_seq(s.set_layer_rtp.seq(), L + 1);
        this->sequence->gen_layer_seq(s.layer.seq(), L + 1);
        if (!this->npu->is_preemption_enabled()){
            auto add_layers = [&](flm_rt::runlist& rl){
                rl.reset();
                rl.add(s.set_layer_rtp.create_run());
                for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
                    rl.add(s.layer.create_run(this->x, this->proj_weights[i], this->rms_weights[i], s.rope[i], this->kv_caches[i]));
                }
            };
            add_layers(s.run);
#if HUNYUAN_LM_HEAD_ON_NPU
            add_layers(s.run_logits);
            s.run_logits.add(this->lm_head.create_run(this->logits, this->lm_head_weights, this->x, this->final_rms_weight));
#endif
        }
        this->_set_rope_rms_weights(L, s.rope);
        s.L = L;
    }
    this->active_slot = slot;
    this->set_context_length(L);
}

/// @brief make the slot built for @p L live. In steady-state decode this costs
///        nothing but the adoption: forward() built it for L while the previous
///        token was on the array.
int hunyuan_npu::Impl::_select_slot(int L){
    int slot = this->active_slot;
    if (this->decode_slots[slot].L != L && this->decode_slots[slot ^ 1].L == L){
        slot ^= 1;
    }
    this->_build_slot(slot, L);
    return slot;
}

void hunyuan_npu::Impl::_invalidate_slots(){
    this->decode_slots[0].L = -1;
    this->decode_slots[1].L = -1;
}

void hunyuan_npu::Impl::_sync_slot_rope_weights(){
    const size_t bytes = (desc.RoPE_DIM + desc.DH * 2) * sizeof(bf16);
    for (int s = 0; s < 2; s++){
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            memcpy(this->decode_slots[s].rope[i].data(), this->rope_rms_weights[i].data(), bytes);
            this->decode_slots[s].rope[i].sync_to_device();
        }
    }
}
#endif

/// @note This is now just bookkeeping: the decode slots are built lazily, by
///       forward() for the token it is about to submit and, behind that run,
///       for the token after it. Nothing here touches the array.
void hunyuan_npu::Impl::set_context_length(int L){
    this->current_context_length = L;
    this->kv_rows_used = std::max(this->kv_rows_used, L);
}

/// @brief single-token decode
/// @note  The host fallback (HUNYUAN_DECODE_ON_NPU == 0) runs the prefill graph
///        at L = 1 rather than a separate decode kernel. That is what made the
///        engine testable before the overlay existed; it is kept as the golden
///        reference for the array path.
buffer<bf16> hunyuan_npu::Impl::forward(int ids, bool need_logits){
#if HUNYUAN_DECODE_ON_NPU
    this->pctx->overlay_lost();   // layer.xclbin runs now
    this->_update_bfp16(this->current_context_length);   // normally already done
    const int slot = this->_select_slot(this->current_context_length);
    decode_slot_t& s = this->decode_slots[slot];
    const int next_L = this->current_context_length + 1;
    bool next_ready = false;
    // the lm_head rides in the same runlist when this token's logits are read
    const bool lm_head_in_runlist = HUNYUAN_LM_HEAD_ON_NPU && need_logits && !this->npu->is_preemption_enabled();

    this->embedding->forward(ids, this->x);
    this->x.sync_to_device();

    if (!this->npu->is_preemption_enabled()){
        flm_rt::runlist& rl = lm_head_in_runlist ? s.run_logits : s.run;
        HY_TRACE_WHERE(-1, this->current_context_length, 1);
        rl.execute();
        HY_TRACE(lm_head_in_runlist ? "decode+lm" : "decode", 'S');
        // ping-pong: assemble the next token's elf and runlist while this
        // token's is on the array. The other slot has its own apps and its own
        // rope buffers, so nothing it writes is being read by the run in flight.
        this->_build_slot(slot ^ 1, next_L);
        next_ready = true;
        flm::npu::wait_for_completion(
            rl, lm_head_in_runlist ? "hunyuan decode+lm_head" : "hunyuan decode");
        HY_TRACE(lm_head_in_runlist ? "decode+lm" : "decode", 'D');
        this->_update_bfp16(next_L);
    }
    else{
        // preemption mode has no runlist and every dispatch blocks, so the only
        // window left to hide the next slot behind is the lm_head run below
        s.set_layer_rtp();
        for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
            s.layer(this->x, this->proj_weights[i], this->rms_weights[i], s.rope[i], this->kv_caches[i]);
        }
        this->_update_bfp16(next_L);
    }

#if HUNYUAN_LM_HEAD_ON_NPU
    // the layer left the hidden state in x on the device and the lm_head reads
    // it from there -- no round trip through the host at all
    if (need_logits && !lm_head_in_runlist){
        this->lm_head_run.start();
        HY_TRACE("lm_head", 'S');
    }
    if (!next_ready){
        this->_build_slot(slot ^ 1, next_L);
    }
    if (!need_logits){
        return this->logits_valid; // stale on purpose: the caller is not reading it
    }
    if (!lm_head_in_runlist){
        const ert_cmd_state lm_state = this->lm_head_run.wait();
        HY_TRACE("lm_head", 'D', (int)lm_state);
        flm::npu::require_completed_ert_state(flm::npu::to_ert_cmd_state(lm_state), "lm_head_run");
    }
    this->logits.sync_from_device();
    return this->logits_valid;
#else
    if (!need_logits){
        if (!next_ready){
            this->_build_slot(slot ^ 1, next_L);
        }
            return this->logits_valid;
    }
    this->x.sync_from_device();
    // get_logits normalizes in place into x, so hand it its own copy
    memcpy(this->decode_hidden.data(), this->x.data(), desc.D * sizeof(bf16));
    buffer<bf16> predict(this->decode_hidden.data(), desc.D);
    get_logits(predict);
    if (!next_ready){
        this->_build_slot(slot ^ 1, next_L);
    }
    return this->logits_valid;
#endif
#else
    this->embedding->forward(ids, this->decode_hidden);

    std::vector<int> input_idx(1, this->current_context_length);

    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++){
        pctx->forward((int)layer_idx, this->decode_hidden, 1,
                      proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx],
                      kv_caches[layer_idx], input_idx, this->current_context_length);
    }

    if (need_logits){
        buffer<bf16> predict(this->decode_hidden.data(), desc.D);
        get_logits(predict);
    }
    this->set_context_length(this->current_context_length + 1);
    this->_update_bfp16(this->current_context_length);
    return this->logits_valid;
#endif
}

/// @brief number of prompt tokens above which the batched prefill beats pushing
///        the prompt through the decode overlay one token at a time. Below it
///        the mv path wins whatever the build says: a decode step is a single
///        array pass with its elf already assembled by the ping-pong, while a
///        batched prefill pays for the L-padded matmuls whether L is 1 token or
///        512. MVPREFILL only widens this -- it drops the ceiling entirely.
///        Krackan after WP11: mv ~15 ms/token + 5 ms, batched ~54 ms flat (32-row
///        plan), so batched wins from 4 tokens and is closer to the CPU reference.
///        3 since 2026-11-05 (user's decision); see "Small-row plans" in README.md.
static constexpr int HUNYUAN_MV_PREFILL_MAX_TOKENS = 3;

/// @brief HUNYUAN_MV_PREFILL_MAX_TOKENS, or HY_MV_PREFILL_MAX from the environment
///        (for measuring the crossover; 0 sends every prompt down the batched path)
static int mv_prefill_max_tokens(){
    static const int value = [](){
        const char* env = std::getenv("HY_MV_PREFILL_MAX");
        return env ? std::atoi(env) : HUNYUAN_MV_PREFILL_MAX_TOKENS;
    }();
    return value;
}

/// @brief the 32 / 64-row matmul plans for short batched prompts (on by default);
///        HY_SMALL_ROWS=0 pads every prompt to 128 rows again, as before WP11
static bool small_rows_enabled(){
    static const bool value = [](){
        const char* env = std::getenv("HY_SMALL_ROWS");
        return env ? std::atoi(env) != 0 : true;
    }();
    return value;
}

/// @brief restore() zeroes the cache rows the turns after the checkpoint wrote, so a turn
///        does not depend on the one before it. On by default since 2026-11-06 (user);
///        HY_ZERO_STALE_KV=0 turns it off. See "Stale KV rows after restore" in README.md.
static bool zero_stale_kv_enabled(){
    static const bool value = [](){
        const char* env = std::getenv("HY_ZERO_STALE_KV");
        return env ? std::atoi(env) != 0 : true;
    }();
    return value;
}

/// @brief longest prompt the short path takes in one call; longer prompts go
///        through _prefill_long. See "Long context" in README.md.
static constexpr int HUNYUAN_SHORT_PREFILL_MAX_TOKENS = 512;

/// @brief run a prompt through the graph
/// @note  Prompt-prefix reuse is not done here -- the UI drives it with
///        checkpoint() / restore() -- so every call prefills all of @p ids from
///        wherever the context currently stands.
buffer<bf16> hunyuan_npu::Impl::prefill(std::vector<int>& ids, void* payload){
    (void)payload;
    if (ids.empty()){
        return this->logits_valid;
    }

    // Nothing below bounds-checks the prompt against the pinned context: a longer
    // one sizes the prefill scratch for rows the kv cache has no room for, and the
    // mha sequence then addresses past the end of the cache bo -- a silent
    // out-of-bounds write on the device, not a failure. So refuse it here.
    if ((size_t)this->current_context_length + ids.size() > (size_t)this->MAX_L) {
        throw std::runtime_error(
            "This model only supports a " + std::to_string(this->MAX_L) + " token context: "
            + std::to_string(this->current_context_length) + " already cached + "
            + std::to_string(ids.size()) + " requested exceeds it");
    }

    if ((int)ids.size() <= HUNYUAN_SHORT_PREFILL_MAX_TOKENS){
        return this->_prefill_short(ids);
    }
    return this->_prefill_long(ids);
}

/// @brief L <= 512: the original path, unchanged
buffer<bf16> hunyuan_npu::Impl::_prefill_short(std::vector<int>& ids){
#if MVPREFILL
    // the build asked for every prompt to go through the decode overlay
    const bool use_mv = true;
#else
    // a short tail after a long chunk (e.g. FastFlowLM's 4096 + 1) goes batched;
    // see "Short tails after a long prompt" in README.md
    const int context_after = this->current_context_length + (int)ids.size();
    const bool short_context = context_after <= HUNYUAN_SHORT_PREFILL_MAX_TOKENS;
    const bool few_tokens = (int)ids.size() <= mv_prefill_max_tokens();
    const bool use_mv = short_context && few_tokens;
#endif
    if (use_mv){
        // only the last token's logits are ever read, so the lm_head -- a full
        // vocab_size x D matvec -- runs once, not once per token
        buffer<bf16> y = this->logits_valid;
        for (size_t i = 0; i < ids.size(); i++){
            y = this->forward(ids[i], i + 1 == ids.size());
        }
        return y;
    }
    return this->_prefill_batched(ids, true, small_rows_enabled());
}

/// @brief L > 512: 512-token pieces through the batched path, lm_head on the last only
buffer<bf16> hunyuan_npu::Impl::_prefill_long(std::vector<int>& ids){
    const size_t total = ids.size();
    const size_t piece = HUNYUAN_SHORT_PREFILL_MAX_TOKENS;
    for (size_t start = 0; start < total; start += piece){
        const size_t end = std::min(total, start + piece);
        std::vector<int> piece_ids(ids.begin() + start, ids.begin() + end);
        this->_prefill_batched(piece_ids, end == total);
    }
    return this->logits_valid;
}

/// @brief one batched (matmul) prefill call over all of @p ids
/// @param need_logits false skips the lm_head; the kv cache is filled either way
/// @param small_rows  let <= 64-token calls use the 32 / 64-row matmul plans
buffer<bf16> hunyuan_npu::Impl::_prefill_batched(std::vector<int>& ids, bool need_logits, bool small_rows){
    // PREFILL_PROF_MARK spans the whole path; the ticks below split it, so the
    // three parts sum to the total.
    PREFILL_PROF_INIT();
    PREFILL_PROF_MARK(mm);
    const int L_in = (int)ids.size();

    buffer<bf16> hidden_states((size_t)desc.D * L_in);
    tensor_2d<bf16> hidden_states_2d(hidden_states, desc.D);

    // plain 1-D absolute positions: no mrope, no vision tokens in this model
    std::vector<int> input_idx(L_in);
    for (int l = 0; l < L_in; l++) {
        input_idx[l] = this->current_context_length + l;
        this->embedding->forward(ids[l], hidden_states_2d[l]);
    }
    PREFILL_PROF_TICK("prefill:embedding");
    HY_TRACE_WHERE(-1, this->current_context_length, L_in);
    HY_TRACE("piece", 'S');

    // the bfp16 copy of each layer's newly complete chunks is made while that
    // layer's attention runs; see "Prefill syncs" in README.md
    const int bfp16_first = this->bfp16_chunks;
    const int bfp16_target = (this->current_context_length + L_in) / hunyuan_bfp16::CHUNK_KEYS;
    for (uint32_t layer_idx = 0; layer_idx < desc.num_hidden_layers; layer_idx++) {
        auto bfp16_layer = [&](){ this->_update_bfp16_layer(layer_idx, bfp16_first, bfp16_target); };
        pctx->forward((int)layer_idx, hidden_states, L_in,
                      proj_weights[layer_idx], rms_weights[layer_idx], rope_rms_weights[layer_idx],
                      kv_caches[layer_idx], input_idx, this->current_context_length, bfp16_layer, small_rows);
    }
    this->bfp16_chunks = std::max(this->bfp16_chunks, bfp16_target);
    HY_TRACE_WHERE(-1, this->current_context_length, L_in);
    HY_TRACE("piece", 'D');
    if (std::getenv("HY_BFP16_CHECK")) this->_check_bfp16();
    PREFILL_PROF_TICK("prefill:layers");

    if (need_logits){
        buffer<bf16> predict = hidden_states_2d[L_in - 1];
        this->lm_head_weights.sync_to_device();
        get_logits(predict);
    }
    PREFILL_PROF_TICK("prefill:lm_head");
    PREFILL_PROF_RATE("prefill:total(mm)", mm, L_in);
    this->set_context_length(this->current_context_length + L_in);
    this->_update_bfp16(this->current_context_length);
    PREFILL_PROF_TICK("prefill:update_bfp16");

    return logits_valid;
}

buffer<bf16> hunyuan_npu::Impl::get_logits(buffer<bf16>& predict){
#if HUNYUAN_LM_HEAD_ON_NPU
    this->pctx->overlay_lost();   // the lm_head runs on layer.xclbin
    // the rms core does the final norm on the array, so hand it the raw hidden
    // state and the model.norm weight (already resident in final_rms_weight)
    if (predict.data() != this->x.data()){
        memcpy(this->x.data(), predict.data(), desc.D * sizeof(bf16));
    }
    this->x.sync_to_device();
    this->lm_head_run.start();
    HY_TRACE("prefill_lm", 'S');
    const ert_cmd_state lm_state = this->lm_head_run.wait();
    HY_TRACE("prefill_lm", 'D', (int)lm_state);
    flm::npu::require_completed_ert_state(flm::npu::to_ert_cmd_state(lm_state), "lm_head_run");
    this->logits.sync_from_device();
#else
    // final norm into x, then the host matvec
    cpu_func::_rms_norm(this->x.data(), predict.data(), this->final_rms_weight.data(), desc.D, 1, 0, 0, desc.rms_norm_eps);
    this->x.sync_to_device();
    cpu_func::_dequant_mm_lm_head(this->logits.data(), this->lm_head_weights.data(), this->x.data(),
                                  1, desc.D, desc.vocab_size_padded);
    this->logits.sync_to_device();
#endif
    return this->logits_valid;
}

/// @note the cache is stored in the array's four-way split layout, so a logical
///       row is reassembled from two quadrants (kv heads 0..1, then 2..3).
buffer<bf16> hunyuan_npu::Impl::get_k_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> k_cache(desc.DK);
    const bf16* base = this->kv_caches[layer_idx].data();
    const size_t half = desc.DK / 2;
    const size_t offset = (size_t)idx * half;
    memcpy(static_cast<void*>(k_cache.data()), static_cast<const void*>(base + this->K03_offset + offset), half * sizeof(bf16));
    memcpy(static_cast<void*>(k_cache.data() + half), static_cast<const void*>(base + this->K47_offset + offset), half * sizeof(bf16));
    return k_cache;
}

buffer<bf16> hunyuan_npu::Impl::get_v_cache(int layer_idx, int idx){
    this->kv_caches[layer_idx].sync_from_device();
    buffer<bf16> v_cache(desc.DV);
    const bf16* base = this->kv_caches[layer_idx].data();
    const size_t half = desc.DV / 2;
    const size_t offset = (size_t)idx * half;
    memcpy(static_cast<void*>(v_cache.data()), static_cast<const void*>(base + this->V03_offset + offset), half * sizeof(bf16));
    memcpy(static_cast<void*>(v_cache.data() + half), static_cast<const void*>(base + this->V47_offset + offset), half * sizeof(bf16));
    return v_cache;
}


int hunyuan_npu::Impl::checkpoint() {
    header_print_r("FLM", "Creating checkpoint at context length " + std::to_string(this->current_context_length));
    is_checkpoint_valid = true;
    checkpoint_context_length = this->current_context_length;
    return this->current_context_length;
}

int hunyuan_npu::Impl::restore() {
    if (!is_checkpoint_valid) {
        header_print_r("FLM", "No valid checkpoint to restore");
        return -1;
    }
    header_print_r("FLM", "Restoring checkpoint at context length " + std::to_string(this->checkpoint_context_length));
    if (zero_stale_kv_enabled()) this->_zero_stale_kv(this->checkpoint_context_length);
    this->set_context_length(this->checkpoint_context_length);
    this->bfp16_chunks = std::min(this->bfp16_chunks, this->current_context_length / hunyuan_bfp16::CHUNK_KEYS);
    return this->current_context_length;
}

/// @note The attention kernels read past the last real row (prefill: the 8-key bfp16 block of
///       P.V, decode: the 16-key bf16 tail chunk), so rows a longer earlier turn left behind
///       change this turn's output. See "Stale KV rows after restore" in README.md.
void hunyuan_npu::Impl::_zero_stale_kv(int first){
    if (first >= this->kv_rows_used) return;
    const size_t row = desc.DK / 2;
    const size_t begin = (size_t)first * row;
    const size_t count = (size_t)(this->kv_rows_used - first) * row;
    for (auto& cache : this->kv_caches){
        for (size_t quadrant : {this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset}){
            memset(cache.data() + quadrant + begin, 0, count * sizeof(bf16));
            hunyuan_flush_range(cache, quadrant + begin, count);
        }
    }
    this->kv_rows_used = first;
}

/// @note The kv cache is zeroed: the NPU prefill attention runs over whole
///       L_padded rounds, so rows past the prompt are read, and an earlier
///       request's k/v left there made identical requests give different logits.
///       Reuse of a shared prompt prefix is the UI's job, through
///       checkpoint() / restore().
void hunyuan_npu::Impl::clear_context(){
    for (auto& kv_cache : this->kv_caches){
        kv_cache.sync_from_device();
        memset(static_cast<void*>(kv_cache.data()), 0, kv_cache.size() * sizeof(bf16));
        kv_cache.sync_to_device();
    }
    this->bfp16_chunks = 0;
    this->kv_rows_used = 0;
    this->set_context_length(0);
}

int hunyuan_npu::Impl::get_current_context_length(){
    return this->current_context_length;
}

/// @note Grows only; a smaller request keeps the current cache.
void hunyuan_npu::Impl::update_max_length(uint32_t MAX_L){
    if (MAX_L <= this->MAX_L) {
        return;
    }

    this->MAX_L = MAX_L;
    const uint32_t rows = kv_cache_rows(MAX_L);
    buffer<bf16> kv_cache_temp(this->kv_caches[0].size());
    const size_t old_quadrant[4] = {this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset};
    // the bf16 part only; the bfp16 V copy is rebuilt below
    const size_t quadrant_bytes = (this->sequence->kv_cache_size / 4) * sizeof(bf16);
    this->sequence->set_max_length(rows);
    this->K03_offset = this->sequence->get_k03_offset();
    this->K47_offset = this->sequence->get_k47_offset();
    this->V03_offset = this->sequence->get_v03_offset();
    this->V47_offset = this->sequence->get_v47_offset();
    const size_t new_quadrant[4] = {this->K03_offset, this->K47_offset, this->V03_offset, this->V47_offset};
    // every quadrant moves, because each one starts at a new fraction of the
    // now-larger buffer
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        memcpy(static_cast<void*>(kv_cache_temp.data()), static_cast<void*>(this->kv_caches[i].data()), this->kv_caches[i].size() * sizeof(bf16));
        this->kv_caches[i] = npu->create_bo_buffer<bf16>(this->sequence->kv_buffer_size);
        memset(this->kv_caches[i].data(), 0, this->kv_caches[i].size() * sizeof(bf16));
        for (int q = 0; q < 4; q++){
            memcpy(static_cast<void*>(this->kv_caches[i].data() + new_quadrant[q]),
                   static_cast<void*>(kv_cache_temp.data() + old_quadrant[q]), quadrant_bytes);
        }
        this->kv_caches[i].sync_to_device();
    }
#if HUNYUAN_DECODE_ON_NPU
    // both slots' runs bind the kv cache BOs that were just replaced
    this->_invalidate_slots();
#endif
    this->pctx->attn_block->invalidate_mha();
    this->set_context_length(this->current_context_length);
    this->bfp16_chunks = 0;
    this->_update_bfp16(this->current_context_length);
    is_checkpoint_valid = false; // the underlying kv cache changed
}

/// @note Reads rows the array may have written, so it invalidates exactly those rows first,
///       and flushes exactly the bytes it writes (an unflushed host write can be evicted over
///       later device data). Each head's 16 rows are a 256-wide slice of the quadrant.
void hunyuan_npu::Impl::_update_bfp16(int context_len){
    const int target = context_len / hunyuan_bfp16::CHUNK_KEYS;
    const int first = this->bfp16_chunks;
    if (target <= first) return;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->_update_bfp16_layer(i, first, target);
    }
    this->bfp16_chunks = target;
    if (std::getenv("HY_BFP16_CHECK")) this->_check_bfp16();
}

void hunyuan_npu::Impl::_update_bfp16_layer(uint32_t layer_idx, int first, int target){
    using namespace hunyuan_bfp16;
    if (target <= first) return;
    assert(this->sequence->v_bfp16_head_chunk_bytes() == HEAD_CHUNK_BYTES);

    const size_t row = desc.DK / 2;   // elements per row of one quadrant (2 kv heads)
    const size_t k_quadrant[2] = {this->K03_offset, this->K47_offset};
    const size_t v_quadrant[2] = {this->V03_offset, this->V47_offset};
    const size_t kb_block[2] = {this->sequence->Kb03_offset * sizeof(bf16), this->sequence->Kb47_offset * sizeof(bf16)};
    const size_t vb_block[2] = {this->sequence->Vb03_offset * sizeof(bf16), this->sequence->Vb47_offset * sizeof(bf16)};
    const size_t rows_bytes = (size_t)(target - first) * CHUNK_KEYS * row * sizeof(bf16);
    const size_t chunk_bytes = 2 * HEAD_CHUNK_BYTES;

    buffer<bf16>& cache = this->kv_caches[layer_idx];
    u8* bytes = reinterpret_cast<u8*>(cache.data());
    for (int col = 0; col < 2; col++){
        invalidate_bytes(cache, (k_quadrant[col] + (size_t)first * CHUNK_KEYS * row) * sizeof(bf16), rows_bytes);
        invalidate_bytes(cache, (v_quadrant[col] + (size_t)first * CHUNK_KEYS * row) * sizeof(bf16), rows_bytes);
    }
    for (int c = first; c < target; c++){
        for (int col = 0; col < 2; col++){
            for (int h = 0; h < 2; h++){
                const bf16* k_rows = cache.data() + k_quadrant[col] + (size_t)c * CHUNK_KEYS * row + h * desc.DH;
                const bf16* v_rows = cache.data() + v_quadrant[col] + (size_t)c * CHUNK_KEYS * row + h * desc.DH;
                const size_t slot = ((size_t)c * 2 + h) * HEAD_CHUNK_BYTES;
                _bfp16_tile_group(bytes + kb_block[col] + slot, k_rows, desc.DH, (int)row);
                _bfp16_vt_tile_group(bytes + vb_block[col] + slot, v_rows, desc.DH, (int)row);
            }
        }
    }
    for (int col = 0; col < 2; col++){
        flush_bytes(cache, kb_block[col] + (size_t)first * chunk_bytes, (size_t)(target - first) * chunk_bytes);
        flush_bytes(cache, vb_block[col] + (size_t)first * chunk_bytes, (size_t)(target - first) * chunk_bytes);
    }
}

void hunyuan_npu::Impl::_check_bfp16(){
    using namespace hunyuan_bfp16;
    const size_t row = desc.DK / 2;
    const size_t k_quadrant[2] = {this->K03_offset, this->K47_offset};
    const size_t v_quadrant[2] = {this->V03_offset, this->V47_offset};
    const size_t kb_block[2] = {this->sequence->Kb03_offset * sizeof(bf16), this->sequence->Kb47_offset * sizeof(bf16)};
    const size_t vb_block[2] = {this->sequence->Vb03_offset * sizeof(bf16), this->sequence->Vb47_offset * sizeof(bf16)};
    const size_t len = (size_t)this->bfp16_chunks * 2 * HEAD_CHUNK_BYTES;
    std::vector<u8> fresh_k(len), fresh_v(len);
    size_t differ = 0, total = 0;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++){
        this->kv_caches[i].sync_from_device();
        const bf16* cache = this->kv_caches[i].data();
        const u8* bytes = reinterpret_cast<const u8*>(cache);
        for (int col = 0; col < 2; col++){
            for (int c = 0; c < this->bfp16_chunks; c++){
                for (int h = 0; h < 2; h++){
                    const size_t slot = ((size_t)c * 2 + h) * HEAD_CHUNK_BYTES;
                    _bfp16_tile_group(fresh_k.data() + slot,
                                      cache + k_quadrant[col] + (size_t)c * CHUNK_KEYS * row + h * desc.DH, desc.DH, (int)row);
                    _bfp16_vt_tile_group(fresh_v.data() + slot,
                                         cache + v_quadrant[col] + (size_t)c * CHUNK_KEYS * row + h * desc.DH, desc.DH, (int)row);
                }
            }
            for (size_t b = 0; b < len; b++) differ += fresh_k[b] != bytes[kb_block[col] + b];
            for (size_t b = 0; b < len; b++) differ += fresh_v[b] != bytes[vb_block[col] + b];
            total += 2 * len;
        }
    }
    header_print("bfp16-check", this->bfp16_chunks << " chunks, " << differ << " of " << total << " bytes differ");
}

///@brief destructor of hunyuan_npu::Impl
hunyuan_npu::Impl::~Impl() = default;

// ===============================================
// Externals for hunyuan_npu
// ===============================================
hunyuan_npu::hunyuan_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L){
    this->_impl = new Impl(config, npu_instance, this, MAX_L);
}

buffer<bf16> hunyuan_npu::forward(int ids){
    return this->_impl->forward(ids);
}

buffer<bf16> hunyuan_npu::prefill(std::vector<int>& ids, void* payload){
    try {
        buffer<bf16> result = this->_impl->prefill(ids, payload);
        return result;
    }
    catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}

void hunyuan_npu::set_context_length(int L){
    std::cout << "Setting context length is not supported for hunyuan_npu!" << std::endl;
}

void hunyuan_npu::load_weights(Q4NX& q4nx){
    this->_impl->load_weights(q4nx);
}

void hunyuan_npu::update_max_length(uint32_t MAX_L){
    this->_impl->update_max_length(MAX_L);
}

void hunyuan_npu::clear_context(){
    this->_impl->clear_context();
}

buffer<bf16> hunyuan_npu::get_k_cache(int layer_idx, int idx){
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> hunyuan_npu::get_v_cache(int layer_idx, int idx){
    return this->_impl->get_v_cache(layer_idx, idx);
}

int hunyuan_npu::get_current_context_length(){
    return this->_impl->get_current_context_length();
}

int hunyuan_npu::checkpoint(){
    return this->_impl->checkpoint();
}

int hunyuan_npu::restore(){
    return this->_impl->restore();
}

hunyuan_npu::~hunyuan_npu(){
    delete this->_impl;
}
