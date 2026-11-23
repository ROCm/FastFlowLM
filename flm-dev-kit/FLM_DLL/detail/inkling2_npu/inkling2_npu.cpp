// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_npu.cpp
/// \brief inkling2_npu: the causal_lm interface over inkling2_host::model
#include "inkling2_npu_detail.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

using namespace inkling2_host;

namespace {

/// \brief throw if config.json does not describe the model this engine was written for
void check_config(const LM_Config& config) {
    const nlohmann::json& t = config.sub("text_config");
    const nlohmann::json& c = t.empty() ? config._json_config : t;
    auto expect = [&](const char* key, long long want) {
        const long long got = cfg_get<long long>(c, key, want);
        if (got != want)
            throw std::runtime_error(std::string("inkling2: config.json ") + key + " = " + std::to_string(got) +
                                     ", this engine supports " + std::to_string(want));
    };
    expect("hidden_size", D);
    expect("num_hidden_layers", NL);
    expect("num_attention_heads", NH);
    expect("num_key_value_heads", NKV);
    expect("head_dim", DH);
    expect("d_rel", NREL);
    expect("rel_extent", REL_EXT);
    expect("sliding_window_size", SWA);
    expect("n_layers_per_attn_res_block", BLOCK);
    expect("n_routed_experts", NE);
    expect("num_experts_per_tok", TOPK);
    expect("intermediate_size", FF);
    expect("sconv_kernel_size", NCONV);
    expect("vocab_size", VOCAB_PAD);
    expect("unpadded_vocab_size", VOCAB);
}

}  // namespace

// =============================================================================
// Impl
// =============================================================================
inkling2_npu::Impl::Impl(LM_Config config, npu_xclbin_manager* npu, int MAX_L)
    : config(config), npu(npu), MAX_L(MAX_L), logits_f32(VOCAB), logits(VOCAB_PAD) {
    check_config(this->config);
    const nlohmann::json& t = this->config.sub("text_config");
    const int model_max = cfg_get<int>(t, "model_max_length", 1 << 20);
    this->MAX_L = std::clamp(MAX_L, 1, model_max);
}

buffer<bf16>& inkling2_npu::Impl::publish_logits() {
    for (int i = 0; i < VOCAB; i++) this->logits[i] = bf16(this->logits_f32[i]);
    // the frontend samples over config vocab_size (the padded one); the padding rows are no tokens
    for (int i = VOCAB; i < VOCAB_PAD; i++) this->logits[i] = bf16(-1e30f);
    return this->logits;
}

std::unique_ptr<hybrid_engine_state> inkling2_npu::Impl::new_state() {
    return std::make_unique<hybrid_engine_state>(model::fixed_bytes(), model::row_bytes());
}

int inkling2_npu::Impl::read_state(hybrid_engine_state& st, int kv_from) {
    const int length = this->model.position();
    kv_from = std::clamp(kv_from, 0, length);
    header_print_r("FLM", "Reading state at context length " + std::to_string(length) + ", K/V rows from " + std::to_string(kv_from));
    const size_t rows = static_cast<size_t>(length - kv_from);
    st.state.resize(1);
    const size_t fixed_elems = model::fixed_bytes() / sizeof(bf16);
    if (st.state[0].size() != fixed_elems) st.state[0] = buffer<bf16>(fixed_elems);
    this->model.save_fixed(reinterpret_cast<u8*>(st.state[0].data()));
    st.kv_k.resize(model::NGLOBAL);
    st.kv_v.resize(model::NGLOBAL);
    for (int j = 0; j < model::NGLOBAL; j++) {
        if (rows == 0) {
            // nothing to keep: a reused state must not hold rows of its previous contents
            st.kv_k[j] = buffer<bf16>();
            st.kv_v[j] = buffer<bf16>();
            continue;
        }
        if (st.kv_k[j].size() != rows * KVD) st.kv_k[j] = buffer<bf16>(rows * KVD);
        if (st.kv_v[j].size() != rows * KVD) st.kv_v[j] = buffer<bf16>(rows * KVD);
        this->model.read_rows(j, kv_from, (int)rows, reinterpret_cast<u16*>(st.kv_k[j].data()),
                              reinterpret_cast<u16*>(st.kv_v[j].data()));
    }
    st.filled = true;
    st.context_length = length;
    st.general_token_idx = length;
    st.kv_from = kv_from;
    return length;
}

bool inkling2_npu::Impl::state_fits(const hybrid_engine_state& st) const {
    if (!st.filled || st.state.size() != 1 || st.state[0].size() * sizeof(bf16) != model::fixed_bytes() ||
        st.kv_k.size() != (size_t)model::NGLOBAL || st.kv_v.size() != (size_t)model::NGLOBAL ||
        st.context_length > this->MAX_L) {
        return false;
    }
    const size_t rows = static_cast<size_t>(st.context_length - st.kv_from);
    for (int j = 0; j < model::NGLOBAL; j++) {
        if (rows > 0 && (st.kv_k[j].size() != rows * KVD || st.kv_v[j].size() != rows * KVD)) return false;
    }
    return true;
}

int inkling2_npu::Impl::write_state_kv(const hybrid_engine_state& st) {
    if (!this->state_fits(st)) {
        return -1;
    }
    const int rows = st.context_length - st.kv_from;
    for (int j = 0; j < model::NGLOBAL && rows > 0; j++) {
        this->model.write_rows(j, st.kv_from, rows, reinterpret_cast<const u16*>(st.kv_k[j].data()),
                               reinterpret_cast<const u16*>(st.kv_v[j].data()));
    }
    return st.context_length;
}

int inkling2_npu::Impl::write_state(const hybrid_engine_state& st) {
    if (!this->state_fits(st)) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(st.context_length));
    this->write_state_kv(st);
    // rows before kv_from are the caller's to have put back (write_state_kv of an older state)
    this->model.load_fixed(reinterpret_cast<const u8*>(st.state[0].data()), st.context_length);
    return st.context_length;
}

// =============================================================================
// inkling2_npu
// =============================================================================
inkling2_npu::inkling2_npu(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L)
    : _impl(new Impl(config, npu_instance, MAX_L)) {}

inkling2_npu::~inkling2_npu() { delete this->_impl; }

void inkling2_npu::load_weights(Q4NX& q4nx) {
    (void)q4nx;   // the host model maps model.q4nx itself: nothing is copied
    this->_impl->model.load(utils::path_join(this->_impl->config.model_path, "model.q4nx"));
    this->_impl->model.set_max_length(this->_impl->MAX_L);
    // the prompt matmuls and attention on the NPU; INKLING2_NPU_PREFILL=0 keeps them on the host,
    // INKLING2_NPU_MIN_T sets the shortest prompt step that goes to the NPU
    const char* on = std::getenv("INKLING2_NPU_PREFILL");
    if (!on || std::strcmp(on, "0") != 0) {
        Impl* d = this->_impl;
        d->prefill = std::make_unique<inkling2_prefill>(
            d->npu, utils::path_join(d->config.exec_path, "xclbins", d->config.model_name, "mm_rtm2_glu.xclbin"));
        if (const char* t = std::getenv("INKLING2_NPU_MIN_T")) d->prefill->min_T = std::max(1, std::atoi(t));
        // INKLING2_NPU_ATTN=0 keeps the prompt attention on the host
        const char* att = std::getenv("INKLING2_NPU_ATTN");
        if (!att || std::strcmp(att, "0") != 0)
            d->prefill->set_attention(
                utils::path_join(d->config.exec_path, "xclbins", d->config.model_name, "prefill_attn.xclbin"),
                utils::path_join(d->config.exec_path, "xclbins", d->config.model_name, "inkling2.switches"));
        d->prefill->load(d->model.w_mut());
        d->prefill->prewarm(std::min(inkling2_host::model::chunk(), d->MAX_L));
        d->model.set_offload(d->prefill.get());
    }
    this->clear_context();
}

buffer<bf16> inkling2_npu::forward(int ids) {
    Impl* d = this->_impl;
    d->compute.call([d, ids] { d->model.run(&ids, 1, d->logits_f32.data()); });
    return this->_impl->publish_logits();
}

buffer<bf16> inkling2_npu::prefill(std::vector<int>& ids, void* payload) {
    (void)payload;
    if (ids.empty()) {
        throw std::runtime_error("inkling2: empty prompt");
    }
    Impl* d = this->_impl;
    // warm switches need the array to themselves from the first layer to the last (inkling2_launch.hpp)
    inkling2_array_lock array(d->prefill && d->prefill->switched());
    d->compute.call([d, &ids] { d->model.run(ids.data(), (int)ids.size(), d->logits_f32.data()); });
    return this->_impl->publish_logits();
}

void inkling2_npu::set_context_length(int L) {
    if (L == 0) {
        this->_impl->model.reset();
    }
    else {
        this->_impl->model.set_position(std::clamp(L, 0, this->_impl->MAX_L));
    }
}

void inkling2_npu::clear_context() {
    this->_impl->legacy_checkpoint.reset();
    this->set_context_length(0);
}

buffer<bf16> inkling2_npu::get_k_cache(int layer_idx, int idx) {
    buffer<bf16> row(KVD);
    this->_impl->model.slot_row(layer_idx, idx, false, reinterpret_cast<u16*>(row.data()));
    return row;
}

buffer<bf16> inkling2_npu::get_v_cache(int layer_idx, int idx) {
    buffer<bf16> row(KVD);
    this->_impl->model.slot_row(layer_idx, idx, true, reinterpret_cast<u16*>(row.data()));
    return row;
}

void inkling2_npu::update_max_length(uint32_t MAX_L) {
    const nlohmann::json& t = this->_impl->config.sub("text_config");
    const int model_max = cfg_get<int>(t, "model_max_length", 1 << 20);
    this->_impl->MAX_L = std::clamp((int)std::min<uint32_t>(MAX_L, std::numeric_limits<int>::max()), 1, model_max);
    this->_impl->model.set_max_length(this->_impl->MAX_L);
}

int inkling2_npu::get_current_context_length() { return this->_impl->model.position(); }

int inkling2_npu::checkpoint() {
    // the single legacy copy keeps no global K/V rows: its caller continues from the same cache
    if (!this->_impl->legacy_checkpoint) {
        this->_impl->legacy_checkpoint = this->_impl->new_state();
    }
    return this->_impl->read_state(*this->_impl->legacy_checkpoint, std::numeric_limits<int>::max());
}

int inkling2_npu::restore() {
    if (!this->_impl->legacy_checkpoint) {
        return -1;
    }
    return this->_impl->write_state(*this->_impl->legacy_checkpoint);
}

std::unique_ptr<hybrid_engine_state> inkling2_npu::new_state() { return this->_impl->new_state(); }

int inkling2_npu::read_state(hybrid_engine_state& state, int kv_from) { return this->_impl->read_state(state, kv_from); }

int inkling2_npu::write_state(const hybrid_engine_state& state) { return this->_impl->write_state(state); }

int inkling2_npu::write_state_kv(const hybrid_engine_state& state) { return this->_impl->write_state_kv(state); }

extern "C" const flm_state_api* inkling2_npu_state_api(void* engine) { return hybrid_state_api<inkling2_npu>(engine); }
