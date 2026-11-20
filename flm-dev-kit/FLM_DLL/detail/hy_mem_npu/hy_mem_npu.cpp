// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file hy_mem_npu.cpp
/// \brief hy_mem_npu: Hy-Memory-30B-A3B on the NPU2 decode overlays
/// \note See hy_mem_npu_detail.hpp for what one token runs, and README.md.
#include <omp.h>
#include <chrono>
#include "hy_mem_npu_detail.hpp"
#include "npu_utils/runtime_wait.hpp"

hy_mem_npu::Impl::Impl(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L)
    : config(config), npu(npu_instance) {
    current_context_length = 0;
    checkpoint_context_length = 0;
    is_checkpoint_valid = false;
    desc.build(config);
    pctx = std::make_unique<hy_mem_prefill_context>(desc);
    this->MAX_L = _normalize_max_length(MAX_L > 0 ? (uint32_t)MAX_L : 0);

    // Three bitstreams. The two layer overlays were both built as layer.xclbin, and
    // register_xclbin() dedupes by file name, so they are installed under their own
    // names. Their RTP addresses are constants in hy_mem_npu_sequence.hpp, which has
    // the note about re-checking them after a kernel rebuild.
    const std::string xclbin_dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
    dense_app_manager = npu->register_xclbin(utils::path_join(xclbin_dir, "dense_layer.xclbin"));
    moe_app_manager = npu->register_xclbin(utils::path_join(xclbin_dir, "moe_layer.xclbin"));
    lm_head_app_manager = npu->register_xclbin(utils::path_join(xclbin_dir, "lm_head.xclbin"));
#if HY_MEM_ATTN_ON_NPU || HY_MEM_MM_ON_NPU
    // the batched prefill's dequant_mm / attention (one xclbin, see hy_mem_prefill_overlay.hpp)
    overlay = std::make_unique<hy_mem_prefill_overlay>(npu, utils::path_join(xclbin_dir, "prefill_overlay.xclbin"), desc);
    pctx->set_overlay(overlay.get());
#endif
    sequence = std::make_unique<hy_mem_npu_sequence>(desc, this->MAX_L);

    dense_layer = dense_app_manager->create_app();
    moe_layer = moe_app_manager->create_app();
    lm_head = lm_head_app_manager->create_app();
    router_rtp.resize(desc.num_hidden_layers);
    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        if (desc.layer_kind(l) == hy_mem_layer_kind::moe) {
            router_rtp[l] = moe_app_manager->create_app();
        }
    }
    moe_runlist = moe_app_manager->create_runlist();

    // ---- per-layer buffers; sizes come from desc only ----
    const uint32_t N = desc.num_hidden_layers;
    proj_weights.resize(N);
    prefill_proj.resize(N);
    rms_weights.resize(N);
    rope_rms_weights.resize(N);
    kv_caches.resize(N);
    router_bias.resize(N);
    const size_t kv_elems = desc.get_kv_cache_alloc_elems(this->MAX_L);
    for (uint32_t l = 0; l < N; l++) {
        const bool dense = desc.layer_kind(l) == hy_mem_layer_kind::dense;
        npu_app& app = dense ? dense_layer : moe_layer;
        proj_weights[l] = app.create_bo_buffer<u8>(dense ? desc.dense_proj_bytes : desc.moe_proj_bytes);
        // the prefill overlay's copy: only a dense layer needs one (see prefill_proj)
        if (dense) prefill_proj[l] = app.create_bo_buffer<u8>(desc.dense_proj_bytes);
        rms_weights[l] = app.create_bo_buffer<bf16>((dense ? desc.dense_rms_bytes : desc.moe_rms_bytes) / sizeof(bf16));
        rope_rms_weights[l] = app.create_bo_buffer<bf16>(desc.get_rope_rms_elems());
        // Zeroed once here: the attention reads whole 16-token chunks, so rows past
        // L are read (and masked) and only have to be finite. Rows a token has
        // written stay finite, so nothing has to re-zero them later.
        kv_caches[l] = app.create_bo_buffer<bf16>(kv_elems);
        kv_caches[l].memset((bf16)0.0f);
        kv_caches[l].sync_to_device();
    }

    x = moe_layer.create_bo_buffer<bf16>(X_ELEMS);
    assert(desc.D + desc.DQ <= X_ELEMS);
    x.memset((bf16)0.0f);
    lm_x = lm_head.create_bo_buffer<bf16>(desc.D * 2);
    lm_head_weights = lm_head.create_bo_buffer<u8>(desc.get_lm_head_w_size());
    logits = lm_head.create_bo_buffer<bf16>(desc.vocab_size_padded);
    logits_valid = buffer<bf16>(logits.data(), desc.vocab_size);
    embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);

    sequence->gen_lm_head_seq(lm_head.seq());
    lm_head_run = lm_head.create_run(logits, lm_head_weights, lm_x);

    LOG_VERBOSE(1, "hy_mem_npu: MAX_L " << this->MAX_L << ", KV cache " << (kv_elems * sizeof(bf16) * N >> 20) << " MB, weights "
                << ((desc.dense_proj_bytes + (size_t)(N - 1) * desc.moe_proj_bytes) >> 20) << " MB");
}

uint32_t hy_mem_npu::Impl::_normalize_max_length(uint32_t requested) {
    uint32_t L = std::max<uint32_t>(requested, 4096);
    // a power of two is also a whole number of 16-token KV chunks
    uint32_t pow2 = 1;
    while (pow2 < L) pow2 <<= 1;
    if (pow2 > hy_mem_desc::MAX_CONTEXT) {
        header_print("WARNING", "hy_mem_npu: context " << requested << " capped at " << hy_mem_desc::MAX_CONTEXT
                     << " (the longest KV cache the attention was checked at)");
        pow2 = hy_mem_desc::MAX_CONTEXT;
    }
    return pow2;
}

void hy_mem_npu::Impl::load_weights(Q4NX& q4nx) {
    embedding->init_weights(&q4nx, "model.embed_tokens");
    const bool check_layout = std::getenv("HY_CHECK_LAYOUT") != nullptr;

    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        if (desc.layer_kind(l) == hy_mem_layer_kind::dense) {
            desc.load_dense_layer(l, q4nx, proj_weights[l], rms_weights[l], rope_rms_weights[l]);
            // the prefill overlay reads this layer through its own relaid copy
            desc.make_prefill_proj(prefill_proj[l], proj_weights[l]);
            prefill_proj[l].sync_to_device();
        } else {
            desc.load_moe_layer(l, q4nx, proj_weights[l], rms_weights[l], rope_rms_weights[l], router_bias[l]);
            // never changes, so this sequence is built once
            sequence->gen_router_rtp_seq(router_rtp[l].seq(), l, router_bias[l]);
        }
        // HY_CHECK_LAYOUT=1: the prefill's weight views against model.q4nx, byte for byte
        if (check_layout) {
            const size_t bad = pctx->check_layout(l, q4nx, proj_weights[l]);
            const std::string tag = bad ? "ERROR" : "info";   // header_print does not parenthesize its tag
            header_print(tag, "layout check, layer " << l << ": " << bad << " blocks differ");
            if (bad) throw std::runtime_error("hy_mem_npu: the prefill weight views do not match the decode layout");
        }
        proj_weights[l].sync_to_device();
        rms_weights[l].sync_to_device();
        rope_rms_weights[l].sync_to_device();
    }

    // the lm_head kernel reads the final norm weight from the second half of its x buffer
    {
        buffer<bf16> final_norm;
        q4nx.load_weights(final_norm, "model.norm.weight");
        memcpy(lm_x.data() + desc.D, final_norm.data(), desc.D * sizeof(bf16));
        lm_x.sync_to_device();
    }
    desc.load_lm_head(q4nx, lm_head_weights);
    lm_head_weights.sync_to_device();

    clear_context();
}

void hy_mem_npu::Impl::_set_rope(int pos) {
    const uint32_t half = desc.RoPE_DIM / 2;
    std::vector<bf16> cos_sin(desc.RoPE_DIM);
    for (uint32_t i = 0; i < half; i++) {
        // fp32 angle, as transformers computes it (inv_freq fp32 * position fp32)
        const float angle = desc.inv_freq[i] * (float)pos;
        cos_sin[i] = (bf16)std::cos(angle);
        cos_sin[i + half] = (bf16)std::sin(angle);
    }
    // every layer has its own buffer (its q/k norms sit behind the table)
    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        memcpy(rope_rms_weights[l].data(), cos_sin.data(), desc.RoPE_DIM * sizeof(bf16));
        rope_rms_weights[l].sync_to_device();
    }
}

namespace {
// HY_TIME_HOST=1: print where each token's host time goes, in microseconds (the
// same switch as the kernel host test in FLM_Xclbin/Hy_mem_30BA3B/moe_decoding)
const bool time_host = std::getenv("HY_TIME_HOST") != nullptr;
using host_clock = std::chrono::steady_clock;
long us_between(host_clock::time_point a, host_clock::time_point b) {
    return (long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}
}  // namespace

/// L = tokens already in the cache; prepares the runs for the token at position L
void hy_mem_npu::Impl::set_context_length(int L) {
    current_context_length = L;
    // The one rule the rest of this engine leans on: rows at or past the context are
    // zero. Both attentions read whole rounds, so they reach past the last real key
    // (prefill a 128-row round, decode a 16-row chunk), and the kernels do not keep
    // those rows out of the answer -- as every other engine here already documents
    // (kv_rows_state.hpp, qwen3_6_moe_npu.cpp). qwen3_6 wipes the whole cache on every
    // clear; we only have to clean what has actually been written, which is what
    // dirty_rows tracks. Growing the context needs nothing: those rows were zero and
    // are being filled. Shrinking it (a clear, a restore, a slot written back) is where
    // another conversation's rows would be left exposed.
    // Callers set the length after the rows are in place, so afterwards exactly the
    // first `want` rows hold K/V either way, whether we just grew or just cleaned.
    const size_t want = (size_t)std::max(L, 0);
    if (want < dirty_rows) {
        kv_rows_zero(kv_caches, kv_layout(), want, dirty_rows);
    }
    dirty_rows = want;
    // a full cache has no next call to prepare: that call would write row L, past
    // the cache. forward() / prefill() refuse to run then.
    if (L >= (int)MAX_L) return;

    const host_clock::time_point t0 = host_clock::now();
    sequence->gen_dense_layer_seq(dense_layer.seq(), L + 1);
    sequence->gen_moe_layer_seq(moe_layer.seq(), L + 1);
    const host_clock::time_point t1 = host_clock::now();
    // the runs hold the sequence's kernel object, so they are rebuilt with it
    // (the first create_run after a sequence change also assembles its ELF)
    dense_run = dense_layer.create_run(x, proj_weights[0], rms_weights[0], rope_rms_weights[0], kv_caches[0]);
    moe_runlist.reset();
    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        if (desc.layer_kind(l) != hy_mem_layer_kind::moe) continue;
        moe_runlist.add(router_rtp[l].create_run());
        moe_runlist.add(moe_layer.create_run(x, proj_weights[l], rms_weights[l], rope_rms_weights[l], kv_caches[l]));
    }
    const host_clock::time_point t2 = host_clock::now();
    _set_rope(L);
    if (time_host) {
        std::cout << "[hy_mem time] L " << L << ": sequences " << us_between(t0, t1) << " us, runs (+ ELF) "
                  << us_between(t1, t2) << " us, rope " << us_between(t2, host_clock::now()) << " us" << std::endl;
    }
}

void hy_mem_npu::Impl::_run_layers(int id) {
    if (current_context_length >= (int)MAX_L) {
        throw std::runtime_error("hy_mem_npu: the KV cache is full (" + std::to_string(MAX_L) + " tokens)");
    }
    const host_clock::time_point t0 = host_clock::now();
    embedding->forward(id, x);
    x.sync_to_device();
    // the three contexts do not order against each other: wait for each before the next
    const host_clock::time_point t1 = host_clock::now();
    dense_run.start();
    flm::npu::wait_for_completion(dense_run, "hy_mem dense");
    const host_clock::time_point t2 = host_clock::now();
    moe_runlist.execute();
    flm::npu::wait_for_completion(moe_runlist, "hy_mem moe runlist");
    if (time_host) {
        std::cout << "[hy_mem time] embed " << us_between(t0, t1) << " us, dense run " << us_between(t1, t2)
                  << " us, moe runlist " << us_between(t2, host_clock::now()) << " us" << std::endl;
    }
}

void hy_mem_npu::Impl::_start_lm_head() {
    x.sync_from_device();
    memcpy(lm_x.data(), x.data(), desc.D * sizeof(bf16));
    lm_x.sync_to_device();
    lm_head_run.start();
}

void hy_mem_npu::Impl::_finish_lm_head() {
    flm::npu::wait_for_completion(lm_head_run, "hy_mem lm_head");
    logits.sync_from_device();
}

buffer<bf16> hy_mem_npu::Impl::forward(int id) {
    _run_layers(id);
    const host_clock::time_point t0 = host_clock::now();
    _start_lm_head();
    // the next token's sequences are built while the lm_head runs (it touches none of them)
    set_context_length(current_context_length + 1);
    _finish_lm_head();
    if (time_host) {
        std::cout << "[hy_mem time] lm_head + next sequences " << us_between(t0, host_clock::now()) << " us" << std::endl;
    }
    return logits_valid;
}

/// Two ways to run the prompt (Makefile PREFILL_WITH_MV picks):
///   default          batched, hy_mem_prefill.hpp: chunks of PREFILL_CHUNK tokens
///                    through all layers at once (dequant_mm + attention on the host
///                    for now), writing the decode KV cache
///   PREFILL_WITH_MV  the decode overlays, one token at a time
buffer<bf16> hy_mem_npu::Impl::prefill(std::vector<int>& ids, void* payload) {
    if (ids.empty()) {
        throw std::runtime_error("hy_mem_npu: empty prefill");
    }
    if (current_context_length + ids.size() > MAX_L) {
        throw std::runtime_error("hy_mem_npu: prompt of " + std::to_string(ids.size()) + " tokens does not fit the KV cache ("
                                 + std::to_string(current_context_length) + " of " + std::to_string(MAX_L) + " used)");
    }
#ifdef PREFILL_WITH_MV
    return _prefill_with_mv(ids);
#else
    return _prefill_batched(ids);
#endif
}

buffer<bf16> hy_mem_npu::Impl::_prefill_with_mv(std::vector<int>& ids) {
    for (size_t i = 0; i + 1 < ids.size(); i++) {
        _run_layers(ids[i]);
        set_context_length(current_context_length + 1);
    }
    return forward(ids.back());
}

buffer<bf16> hy_mem_npu::Impl::_prefill_batched(std::vector<int>& ids) {
    // the decode buffers, which the prefill reads (weights) and writes (KV cache) in place
    std::vector<hy_mem_layer_buffers> layers(desc.num_hidden_layers);
    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        const bool dense = desc.layer_kind(l) == hy_mem_layer_kind::dense;
        layers[l] = {&proj_weights[l], &rms_weights[l], &rope_rms_weights[l], &kv_caches[l], &router_bias[l],
                     dense ? &prefill_proj[l] : &proj_weights[l]};
    }

    // decode / lm_head ran since the last prefill (or nothing did yet): the overlay's
    // configuration is gone, its first run writes all of it again
    if (overlay) overlay->invalidate();

    const int total = (int)ids.size();
    const int D = desc.D;
    std::vector<bf16> x;
    int last_rows = 0;
    // Chunks bound the scratch (the dense up|gate rows alone are 27 KB a token); a
    // later chunk attends to the earlier ones through the KV cache, like a new turn.
    for (int c0 = 0; c0 < total; c0 += PREFILL_CHUNK) {
        const int n = std::min(PREFILL_CHUNK, total - c0);
        x.resize((size_t)n * D);
        for (int i = 0; i < n; i++) {
            memcpy(x.data() + (size_t)i * D, embedding->w.data() + (size_t)ids[c0 + i] * D, D * sizeof(bf16));
        }
        pctx->forward(x.data(), n, current_context_length, layers, MAX_L);
        current_context_length += n;
        last_rows = n;
    }

    // the last token's hidden state -> lm_head; the decode sequences for the next
    // position are built while it runs
    memcpy(lm_x.data(), x.data() + (size_t)(last_rows - 1) * D, D * sizeof(bf16));
    lm_x.sync_to_device();
    lm_head_run.start();
    set_context_length(current_context_length);
    _finish_lm_head();
    return logits_valid;
}

buffer<bf16> hy_mem_npu::Impl::_gather_kv_row(int layer_idx, int kv, int idx) {
    assert(idx >= 0 && idx < (int)MAX_L);
    kv_caches[layer_idx].sync_from_device();
    const size_t col_row = desc.kv_col_row();
    buffer<bf16> row(hy_mem_desc::ATTN_COLS * col_row);
    for (int col = 0; col < hy_mem_desc::ATTN_COLS; col++) {
        const bf16* src = kv_caches[layer_idx].data() + desc.kv_block_offset(col, kv, MAX_L) + (size_t)idx * col_row;
        memcpy(row.data() + col * col_row, src, col_row * sizeof(bf16));
    }
    return row;
}

buffer<bf16> hy_mem_npu::Impl::get_k_cache(int layer_idx, int idx) { return _gather_kv_row(layer_idx, 0, idx); }
buffer<bf16> hy_mem_npu::Impl::get_v_cache(int layer_idx, int idx) { return _gather_kv_row(layer_idx, 1, idx); }

/// Every layer is attention, so the state is only the KV rows, and a later token
/// only ever writes rows at or past its own position: rows below the checkpoint
/// stay as they are. A checkpoint is therefore just the context length.
int hy_mem_npu::Impl::checkpoint() {
    checkpoint_context_length = current_context_length;
    is_checkpoint_valid = true;
    return current_context_length;
}

/// Rows past the checkpoint keep stale (finite) values; they are masked until a
/// new token overwrites them.
int hy_mem_npu::Impl::restore() {
    if (!is_checkpoint_valid) {
        header_print_r("FLM", "No valid checkpoint to restore");
        return -1;
    }
    set_context_length(checkpoint_context_length);
    return current_context_length;
}

/// Every layer is full attention and nothing else survives a token, so a saved
/// state is just K/V rows: kv_rows_state.hpp does the copying, and all this has
/// to say is where the four planes are.
///
/// The "striped" layout there means four planes of one row each per position,
/// stride DK/2. That is exactly our [K c0][K c1][V c0][V c1]: ATTN_COLS is 2 and
/// DK == DV, so kv_col_row() == DK/2. K c0 sits at 0, the other three at
/// kv_block_offset().
kv_row_layout hy_mem_npu::Impl::kv_layout() const {
    kv_row_layout l;
    l.DK = desc.DK;
    l.DV = desc.DV;
    l.striped = true;
    l.k47 = desc.kv_block_offset(1, 0, MAX_L);   // K column 1
    l.v03 = desc.kv_block_offset(0, 1, MAX_L);   // V column 0
    l.v47 = desc.kv_block_offset(1, 1, MAX_L);   // V column 1
    return l;
}

std::unique_ptr<hybrid_engine_state> hy_mem_npu::Impl::new_state() {
    return kv_rows_new_state(desc.num_hidden_layers, kv_layout());
}

int hy_mem_npu::Impl::read_state(hybrid_engine_state& state, int kv_from) {
    header_print_r("FLM", "Reading state at context length " + std::to_string(current_context_length)
                          + ", K/V rows from " + std::to_string(kv_from));
    return kv_rows_read(state, kv_caches, kv_layout(), current_context_length, kv_from);
}

int hy_mem_npu::Impl::write_state_kv(const hybrid_engine_state& state) {
    return kv_rows_write(state, kv_caches, kv_layout(), MAX_L);
}

/// Rows past the restored length hold whatever ran since the state was read. They are
/// not ours to keep: set_context_length() puts the zeros of a fresh context back.
int hy_mem_npu::Impl::write_state(const hybrid_engine_state& state) {
    const int length = kv_rows_write(state, kv_caches, kv_layout(), MAX_L);
    if (length < 0) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    // the legacy checkpoint was taken against a context this state replaces
    is_checkpoint_valid = false;
    set_context_length(length);
    return current_context_length;
}

/// set_context_length(0) does the wiping: it zeroes every row that has been written
/// since they were last clean, which is qwen3_6_moe_npu::clear_context's full memset
/// without the part that was never touched.
void hy_mem_npu::Impl::clear_context() {
    is_checkpoint_valid = false;
    set_context_length(0);
}

/// The KV block offsets scale with MAX_L, so each column's K and V block moves
/// to its new place (only the rows in use are copied).
void hy_mem_npu::Impl::update_max_length(uint32_t new_max) {
    new_max = _normalize_max_length(new_max);
    if (new_max <= MAX_L) {
        header_print("FLM", "New length is not longer than the current length, no need to update!");
        return;
    }
    const size_t col_row = desc.kv_col_row();
    const size_t rows = (size_t)current_context_length;
    for (uint32_t l = 0; l < desc.num_hidden_layers; l++) {
        buffer<bf16>& old_cache = kv_caches[l];
        old_cache.sync_from_device();
        npu_app& app = desc.layer_kind(l) == hy_mem_layer_kind::dense ? dense_layer : moe_layer;
        buffer<bf16> new_cache = app.create_bo_buffer<bf16>(desc.get_kv_cache_alloc_elems(new_max));
        new_cache.memset((bf16)0.0f);
        for (int col = 0; col < hy_mem_desc::ATTN_COLS; col++) {
            for (int kv = 0; kv < 2; kv++) {
                memcpy(new_cache.data() + desc.kv_block_offset(col, kv, new_max),
                       old_cache.data() + desc.kv_block_offset(col, kv, MAX_L),
                       rows * col_row * sizeof(bf16));
            }
        }
        new_cache.sync_to_device();
        kv_caches[l] = new_cache;
    }
    MAX_L = new_max;
    sequence->set_max_length(MAX_L);
    set_context_length(current_context_length);   // new sequences, runs bound to the new caches
}

hy_mem_npu::Impl::~Impl() = default;

// ===============================================
// Externals for hy_mem_npu
// ===============================================
hy_mem_npu::hy_mem_npu(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L) {
    this->_impl = new Impl(config, npu_instance, MAX_L);
}

buffer<bf16> hy_mem_npu::forward(int ids) {
    return this->_impl->forward(ids);
}

buffer<bf16> hy_mem_npu::prefill(std::vector<int>& ids, void* payload) {
    return this->_impl->prefill(ids, payload);
}

void hy_mem_npu::set_context_length(int L) {
    this->_impl->set_context_length(L);
}

void hy_mem_npu::load_weights(Q4NX& q4nx) {
    this->_impl->load_weights(q4nx);
}

void hy_mem_npu::update_max_length(uint32_t MAX_L) {
    this->_impl->update_max_length(MAX_L);
}

void hy_mem_npu::clear_context() {
    this->_impl->clear_context();
}

buffer<bf16> hy_mem_npu::get_k_cache(int layer_idx, int idx) {
    return this->_impl->get_k_cache(layer_idx, idx);
}

buffer<bf16> hy_mem_npu::get_v_cache(int layer_idx, int idx) {
    return this->_impl->get_v_cache(layer_idx, idx);
}

int hy_mem_npu::get_current_context_length() {
    return this->_impl->get_current_context_length();
}

int hy_mem_npu::checkpoint() {
    return this->_impl->checkpoint();
}

int hy_mem_npu::restore() {
    return this->_impl->restore();
}

std::unique_ptr<hybrid_engine_state> hy_mem_npu::new_state() {
    return this->_impl->new_state();
}

int hy_mem_npu::read_state(hybrid_engine_state& state, int kv_from) {
    return this->_impl->read_state(state, kv_from);
}

int hy_mem_npu::write_state(const hybrid_engine_state& state) {
    return this->_impl->write_state(state);
}

int hy_mem_npu::write_state_kv(const hybrid_engine_state& state) {
    return this->_impl->write_state_kv(state);
}

hy_mem_npu::~hy_mem_npu() {
    delete this->_impl;
}

extern "C" const flm_state_api* hy_mem_npu_state_api(void* engine) {
    return hybrid_state_api<hy_mem_npu>(engine);
}
