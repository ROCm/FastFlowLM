// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_npu.cpp
/// \brief inkling2_npu: the causal_lm interface over inkling2_host::model
#include "inkling2_npu_detail.hpp"
#include "inkling2_vision_npu.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <thread>
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

void inkling2_npu::Impl::make_fused() {
    this->fused.reset();
    this->on_device = false;
    this->dev_ids.clear();
    const char* on = std::getenv("INKLING2_NPU_DECODE");
    if (!this->prefill || (on && std::strcmp(on, "0") == 0)) return;
    std::vector<inkling2_fused::layer_src> src;
    for (int il = 0; il < NL; il++) {
        const auto l = this->prefill->device_layer(il);
        src.push_back({l.w, l.o, l.gate_up, l.down});
    }
    try {
        this->fused = std::make_unique<inkling2_fused>(
            this->npu, utils::path_join(this->config.exec_path, "xclbins", this->config.model_name, "decode.xclbin"),
            this->model.w(), src, this->MAX_L);
        this->fused->sync_all = false;
    } catch (const std::exception& e) {
        header_print_r("FLM", std::string("inkling2: decode on the host (") + e.what() + ")");
        return;
    }
    // the lm_head on the NPU, in the decode's hardware context (control-packet switches built with this decode.xclbin);
    // INKLING2_NPU_LMH=0: on the host
    const char* lmh = std::getenv("INKLING2_NPU_LMH");
    const std::string sw = utils::path_join(this->config.exec_path, "xclbins", this->config.model_name, "inkling2_decode.switches");
    if (!(lmh && std::strcmp(lmh, "0") == 0) && std::filesystem::exists(sw)) {
        if (this->fused->enable_lm_head(this->npu, sw)) header_print("inkling2", "lm_head on the NPU (" << sw << ")");
        else header_print_r("FLM", "inkling2: lm_head on the host (the decode switches did not load)");
    }
}

void inkling2_npu::Impl::sync_host() {
    if (!this->on_device) return;
    this->fused->store_state(this->model);
    this->on_device = false;
    this->dev_ids.clear();
}

void inkling2_npu::Impl::decode_npu(int id) {
    if (!this->on_device) {
        this->fused->load_state(this->model);
        this->on_device = true;
        this->dev_ids.clear();
    }
    this->fused->begin_token(id);
    if (!this->fused->run_token()) {
        // the device is suspect: the host replays the device's tokens and runs this one; decode stays on the host
        header_print_r("FLM", "inkling2: an NPU decode run did not complete; decode continues on the host");
        this->fused.reset();
        this->on_device = false;
        if (!this->dev_ids.empty()) this->model.run(this->dev_ids.data(), (int)this->dev_ids.size(), nullptr);
        this->dev_ids.clear();
        this->model.run(&id, 1, this->logits_f32.data());
        return;
    }
    if (this->fused->lm_on_npu()) {
        this->fused->logits(this->logits_f32.data());
    } else {
        const uint16_t* x = this->fused->x_out();
        this->fused_x.resize(D);
        for (int c = 0; c < D; c++) {
            const uint32_t u = (uint32_t)x[c] << 16;
            std::memcpy(&this->fused_x[c], &u, 4);
        }
        lm_head(this->model.w(), this->fused_x.data(), this->logits_f32.data());
    }
    this->fused->end_token();
    this->dev_ids.push_back(id);
}

int inkling2_npu::Impl::read_state(hybrid_engine_state& st, int kv_from) {
    this->sync_host();
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
    this->sync_host();
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
    this->sync_host();
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

std::vector<uint16_t> inkling2_npu::Impl::encode_images(const inkling2_payload_t& payload) {
    if (!this->vision) throw std::runtime_error("inkling2: this model folder has no vision tower (vision_weight.q4nx)");
    if (!this->vision_npu_tried) {
        this->vision_npu_tried = true;
        const char* off = std::getenv("INKLING2_VISION_NPU");
        const std::string dir = utils::path_join(this->config.exec_path, "xclbins", this->config.model_name);
        const std::string mm = utils::path_join(dir, "vision_mm.xclbin");
        const std::string at = utils::path_join(dir, "vision_attn.xclbin");
        if (off && std::strcmp(off, "0") == 0) {
            header_print_r("FLM", "Inkling-2 image tower on the host (INKLING2_VISION_NPU=0)");
        }
        else if (!std::filesystem::exists(mm) || !std::filesystem::exists(at)) {
            header_print_r("FLM", "Inkling-2 image tower on the host: vision_mm.xclbin / vision_attn.xclbin not found");
        }
        else {
            try {
                inkling2_vision::npu_options opt;
                opt.profile = std::getenv("INKLING2_VISION_PROF") != nullptr;
                this->vision->set_accel(inkling2_vision::make_npu_accel(this->npu, mm, at, opt));
            } catch (const std::exception& e) {
                header_print_r("FLM", std::string("Inkling-2 image tower on the host: ") + e.what());
                this->vision->set_accel(nullptr);
            }
        }
    }
    std::vector<uint16_t> rows;
    std::vector<float> pixels;
    for (int i = 0; i < payload.num_images; i++) {
        const inkling2_image_t& im = payload.images[i];
        if (!im.rgb || im.height <= 0 || im.width <= 0) throw std::runtime_error("inkling2: empty image");
        const inkling2_vision::grid g = inkling2_vision::grid_for(im.height, im.width);
        pixels.resize((size_t)g.tokens() * inkling2_vision::CELL * inkling2_vision::CELL * 3);
        inkling2_vision::patchify(im.rgb, im.height, im.width, pixels.data());
        const size_t at = rows.size();
        rows.resize(at + (size_t)g.tokens() * D);
        this->vision->encode(pixels.data(), g, rows.data() + at);
    }
    return rows;
}

// =============================================================================
// inkling2_npu
// =============================================================================
inkling2_npu::inkling2_npu(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L)
    : _impl(new Impl(config, npu_instance, MAX_L)) {}

inkling2_npu::~inkling2_npu() { delete this->_impl; }

void inkling2_npu::load_weights(Q4NX& q4nx) {
    (void)q4nx;   // the host model maps model.q4nx itself: nothing is copied
    Impl* d = this->_impl;
    // on a thread of its own (not std::async: MSVC runs it on a pool thread that lives on): the OpenMP pool the
    // load grows ends with it (inkling2_host.cpp, use_core_threads)
    std::exception_ptr err;
    std::thread([d, &err] {
        try {
            d->model.load(utils::path_join(d->config.model_path, "model.q4nx"));
            d->model.set_max_length(d->MAX_L);
            // the image tower: vision_weight.q4nx beside model.q4nx (INKLING2_VISION=0 leaves it out)
            const char* vis = std::getenv("INKLING2_VISION");
            const std::string vq = utils::path_join(d->config.model_path, "vision_weight.q4nx");
            if ((!vis || std::strcmp(vis, "0") != 0) && std::filesystem::exists(vq)) {
                d->vision = std::make_unique<inkling2_vision::tower>();
                d->vision->load(vq);
            }
            // the prompt matmuls and attention on the NPU; INKLING2_NPU_PREFILL=0 keeps them on the host,
            // INKLING2_NPU_MIN_T sets the shortest prompt step that goes to the NPU
            const char* on = std::getenv("INKLING2_NPU_PREFILL");
            if (on && std::strcmp(on, "0") == 0) return;
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
            d->make_fused();
        } catch (...) {
            err = std::current_exception();
        }
    }).join();
    if (err) std::rethrow_exception(err);
    this->clear_context();
}

buffer<bf16> inkling2_npu::forward(int ids) {
    Impl* d = this->_impl;
    d->compute.call([d, ids] {
        if (d->fused) d->decode_npu(ids);
        else d->model.run(&ids, 1, d->logits_f32.data());
    });
    return this->_impl->publish_logits();
}

bool inkling2_npu::has_vision_tower() const { return this->_impl->vision != nullptr; }

int inkling2_npu::image_tokens_for(int height, int width) const {
    return inkling2_vision::grid_for(height, width).tokens();
}

buffer<bf16> inkling2_npu::prefill(std::vector<int>& ids, void* payload) {
    const inkling2_payload_t* images = static_cast<const inkling2_payload_t*>(payload);
    if (images && images->num_images <= 0) images = nullptr;
    if (ids.empty()) {
        throw std::runtime_error("inkling2: empty prompt");
    }
    Impl* d = this->_impl;
    // warm switches need the array to themselves from the first layer to the last (inkling2_launch.hpp)
    inkling2_array_lock array(d->prefill && d->prefill->switched());
    d->compute.call([d, &ids, images] {
        d->sync_host();
        if (!images) {
            d->model.run(ids.data(), (int)ids.size(), d->logits_f32.data());
            return;
        }
        // the tower's rows replace the embeddings of the placeholder tokens (inside the array lock above)
        const std::vector<uint16_t> rows = d->encode_images(*images);
        d->model.run(ids.data(), (int)ids.size(), d->logits_f32.data(), rows.data(), (int)(rows.size() / D),
                     inkling2_npu::image_token_id);
    });
    return this->_impl->publish_logits();
}

void inkling2_npu::set_context_length(int L) {
    if (L == 0) this->_impl->on_device = false;   // nothing of the device's tokens is kept
    this->_impl->sync_host();
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
    this->_impl->sync_host();
    buffer<bf16> row(KVD);
    this->_impl->model.slot_row(layer_idx, idx, false, reinterpret_cast<u16*>(row.data()));
    return row;
}

buffer<bf16> inkling2_npu::get_v_cache(int layer_idx, int idx) {
    this->_impl->sync_host();
    buffer<bf16> row(KVD);
    this->_impl->model.slot_row(layer_idx, idx, true, reinterpret_cast<u16*>(row.data()));
    return row;
}

void inkling2_npu::update_max_length(uint32_t MAX_L) {
    const nlohmann::json& t = this->_impl->config.sub("text_config");
    const int model_max = cfg_get<int>(t, "model_max_length", 1 << 20);
    this->_impl->sync_host();
    this->_impl->MAX_L = std::clamp((int)std::min<uint32_t>(MAX_L, std::numeric_limits<int>::max()), 1, model_max);
    this->_impl->model.set_max_length(this->_impl->MAX_L);
    // the device caches are sized for the context length
    this->_impl->make_fused();
}

int inkling2_npu::get_current_context_length() {
    return this->_impl->on_device ? this->_impl->fused->position() : this->_impl->model.position();
}

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
