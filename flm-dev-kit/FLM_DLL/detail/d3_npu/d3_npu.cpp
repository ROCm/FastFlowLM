// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <map>
#include <set>
#include <stdexcept>
#include "d3_npu_detail.hpp"
#include "d3_par.hpp"
// after every engine header, so none of them sees windows.h's macros
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fstream>
#include <string>
#include <utility>
#endif

int d3_physical_cores() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int cores = 0;
    for (DWORD off = 0; off < len;) {   // one RelationProcessorCore record per physical core
        auto* rec = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (rec->Relationship == RelationProcessorCore) cores++;
        off += rec->Size;
    }
    return cores;
#else
    // distinct (package, core) pairs over the online CPUs
    std::set<std::pair<int, int>> cores;
    const unsigned n = std::thread::hardware_concurrency();
    for (unsigned cpu = 0; cpu < 4 * n + 64; cpu++) {   // CPU ids may have gaps (offline CPUs)
        const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream core(dir + "core_id"), pkg(dir + "physical_package_id");
        int c = -1, p = -1;
        if (!(core >> c) || !(pkg >> p)) continue;
        cores.insert({p, c});
    }
    return (int)cores.size();
#endif
}

d3_npu::Impl::Impl(LM_Config config, npu_xclbin_manager* npu_instance, int max_length)
    : config(config), npu(npu_instance), MAX_L(max_length) {
    if (MAX_L < 1) throw std::invalid_argument("d3_npu: max_length must be positive");
    desc.build(this->config);
    sequence = std::make_unique<d3_npu_sequence>(desc);

    const int n = desc.num_hidden_layers;
    proj_weights.resize(n);
    rms_weights.resize(n);
    rope_rms_weights.resize(n);
    alpha_proj_cpu.resize(n);
    beta_proj_cpu.resize(n);
    // Weights are device BOs in every configuration: d3_desc's loaders (a frozen contract shared
    // with qwen3_5vl / minicpm_v) sync them to the device unconditionally.
    for (int i = 0; i < n; i++) {
        if (desc.is_attn_layer(i)) {
            proj_weights[i] = npu->create_bo_buffer<u8>(desc.attn_total_proj_weights_byte_size);
            rms_weights[i] = npu->create_bo_buffer<bf16>(desc.attn_rms_weights_byte_size / sizeof(bf16));
            rope_rms_weights[i] = npu->create_bo_buffer<bf16>(desc.RoPE_DIM + desc.DH * 2);
        } else {
            proj_weights[i] = npu->create_bo_buffer<u8>(desc.linear_total_proj_weights_byte_size);
            rms_weights[i] = npu->create_bo_buffer<bf16>(desc.linear_rms_weights_byte_size / sizeof(bf16));
            rope_rms_weights[i] = npu->create_bo_buffer<bf16>(desc.linear_ssm_weights_byte_size / sizeof(bf16));
            alpha_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
            beta_proj_cpu[i] = buffer<bf16>((size_t)desc.D * desc.linear_num_v_heads);
        }
    }
    final_norm = buffer<bf16>(desc.D);
    pctx = std::make_unique<d3_prefill_context>(npu, &desc, sequence.get(), this->config);
    pctx->set_scratch_rows(std::max(CACHE_ROWS, pack_rows));
}

void d3_npu::Impl::_trim() {
    pctx->trim();
    std::vector<bf16>().swap(features);
    if (residual.size() > (size_t)pctx->cache_rows * desc.D) residual = buffer<bf16>();
}

void d3_npu::Impl::load_weights(Q4NX& q4nx) {
    if (q4nx.has_tensor("model.embed_tokens.weight")) {
        embedding = std::make_unique<Embedding>(desc.vocab_size, desc.D);
        embedding->init_weights(&q4nx, "model.embed_tokens");
    } else {
        embedding_q8 = std::make_unique<embedding_q8_0>(desc.vocab_size, desc.D);
        embedding_q8->init_weights(q4nx, "lm_head.weight");
    }
    for (int i = 0; i < (int)desc.num_hidden_layers; i++) {
        if (desc.is_attn_layer(i)) {
            desc.load_attn_weights(i, q4nx, proj_weights[i], rms_weights[i], rope_rms_weights[i]);
        } else {
            desc.load_linear_weights(i, q4nx, proj_weights[i], rms_weights[i], rope_rms_weights[i],
                                     alpha_proj_cpu[i], beta_proj_cpu[i]);
        }
    }
    buffer<bf16> w;
    q4nx.load_weights(w, "model.norm.weight");
    memcpy(final_norm.data(), w.data(), desc.D * sizeof(bf16));
    loaded = true;
}

void d3_npu::Impl::_embed(int id, bf16* dst) {
    if (id < 0 || id >= (int)desc.vocab_size) throw std::out_of_range("d3_npu: token id out of range");
    buffer<bf16> row(dst, desc.D);
    if (embedding) embedding->forward(id, row);
    else embedding_q8->forward(id, row);
}

/// M1 per-row path: every prompt of the payload holds all of its images in order (image token runs only)
static int image_positions(const std::vector<int>& ids, const d3_image_payload& m, int* pos) {
    return d3_plan_positions(ids, m.image_token_id, -1, m.images.data(), m.images.size(), 0, pos);
}

void d3_npu::Impl::_check_images(const d3_image_payload& m, const char* who) const {
    size_t patches = 0;
    for (const auto& im : m.images) {
        if (im.grid_h <= 0 || im.grid_w <= 0 || im.grid_h % 2 || im.grid_w % 2)
            throw std::invalid_argument(std::string("d3_npu: ") + who + "image grids must be positive and even");
        patches += (size_t)im.grid_h * im.grid_w;
    }
    // per patch: 3 channels x 2 frames x 16 x 16
    if (m.pixels.size() != patches * 3 * 2 * 16 * 16)
        throw std::invalid_argument(std::string("d3_npu: ") + who + "image pixels do not match the image grids");
}

void d3_npu::Impl::_check_image_prompts(const std::vector<std::vector<int>>& ids, const d3_image_payload& m) const {
    _check_images(m, "");
    for (const auto& p : ids) d3_check_visual_runs(p, m.image_token_id, -1, m.images.data(), m.images.size(), "a prompt");
}

void d3_npu::Impl::_encode_pack(const std::vector<std::vector<int>>& ids, size_t first, size_t last,
                                std::vector<buffer<bf16>>& out) {
    PREFILL_PROF_INIT();
    std::vector<int> lens, states;
    for (size_t i = first; i < last; i++) {
        lens.push_back((int)ids[i].size());
        if (state_len) states.push_back((*state_len)[i]);
    }
    d3_batch b;
    b.build(lens, states);
    const int D = desc.D;
    if (residual.size() < (size_t)b.R_padded * D) residual = buffer<bf16>((size_t)b.R_padded * D);
    for (size_t i = first; i < last; i++) {
        const auto& s = b.segs[i - first];
        if (media) {
            // every prompt holds all images in order: image tokens take the merger rows in order
            const bf16* row = features.data();
            for (int l = 0; l < s.len; l++) {
                bf16* dst = residual.data() + (size_t)(s.begin + l) * D;
                if (ids[i][l] == media->image_token_id) {
                    memcpy(dst, row, (size_t)D * sizeof(bf16));
                    row += D;
                } else {
                    _embed(ids[i][l], dst);
                }
            }
            image_positions(ids[i], *media, b.input_idx.data() + (size_t)s.begin * 3);
        } else {
            for (int l = 0; l < s.len; l++) _embed(ids[i][l], residual.data() + (size_t)(s.begin + l) * D);
        }
    }
    D3_RIG_DUMP("emb", residual.data(), b.segs[0].len, D, D);
    PREFILL_PROF_TICK("encode:embed");

    for (int layer = 0; layer < (int)desc.num_hidden_layers; layer++) {
        pctx->forward(layer, b, residual, proj_weights[layer], rms_weights[layer], rope_rms_weights[layer],
                      alpha_proj_cpu[layer], beta_proj_cpu[layer]);
    }
    PREFILL_PROF_TICK("encode:layers");

    // Moved, never copied: in the default XRT build, FLM_DLL/include/buffer.hpp's copy
    // constructor does not share ownership, so a copy of `h` dangles once `h` goes out of
    // scope (the FLM_USE_HRX branch and FLM_Xclbin/host_common/buffer.hpp do share it).
    for (const auto& s : b.segs) {
        buffer<bf16> h(D);
        cpu_func::_rms_norm(h.data(), residual.data() + (size_t)(s.begin + s.len - 1) * D, final_norm.data(), D, 1, 0, 0);
        out.push_back(std::move(h));
    }
    D3_RIG_DUMP("final_norm.last", out[out.size() - b.segs.size()].data(), 1, D, D);
}

void d3_npu::Impl::_ensure_vision() {
    if (vision) return;
    try {
        auto v = std::make_unique<D3VisionEncoder>(config, npu);
        SafeTensors w(config.get<std::string>("vision_model_weight", ""));
        v->init_weights(w);
        vision = std::move(v);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("d3_npu: cannot load the vision tower (") + e.what() + ")");
    }
}

/// patches (before the 2 x 2 merge) one vision tower call may hold: 65,536 = the 16,384 merged tokens a request
/// may spend on videos. D3_VISION_MAX_PATCHES overrides it (a call holds whole images: one over the limit goes alone).
static size_t vision_max_patches() {
    static const size_t v = [] {
        const char* e = std::getenv("D3_VISION_MAX_PATCHES");
        const long n = e ? std::atol(e) : 0;
        return n > 0 ? (size_t)n : (size_t)65536;
    }();
    return v;
}

void d3_npu::Impl::_vision_rows(const d3_image_payload& m, std::vector<bf16>& rows) {
    if (const char* feed = std::getenv("D3_VISION_FEED")) {
        // tests: take the rows from a D3_VISION_DUMP-format file (per call int32 {images, tokens} + bf16 [tokens][D]) instead of
        // the tower, e.g. reference features: isolates the language path from the tower's error. Calls are consumed in order.
        static size_t cursor = 0;
        size_t tokens = 0;
        for (const auto& im : m.images) tokens += (size_t)im.grid_h * im.grid_w / 4;
        FILE* f = std::fopen(feed, "rb");
        if (!f) throw std::runtime_error("d3_npu: D3_VISION_FEED: cannot open the file");
        std::fseek(f, (long)cursor, SEEK_SET);
        int32_t head[2] = {0, 0};
        if (std::fread(head, sizeof(int32_t), 2, f) != 2 || (size_t)head[0] != m.images.size() || (size_t)head[1] != tokens) {
            std::fclose(f);
            throw std::runtime_error("d3_npu: D3_VISION_FEED: the next record does not match this call");
        }
        rows.assign(tokens * desc.D, bf16(0.0f));
        const size_t got = std::fread(rows.data(), sizeof(bf16), rows.size(), f);
        std::fclose(f);
        if (got != rows.size()) throw std::runtime_error("d3_npu: D3_VISION_FEED: short record");
        cursor += 8 + rows.size() * sizeof(bf16);
        return;
    }
    _ensure_vision();
    const bool timing = std::getenv("D3_VISION_TIMING") != nullptr;
    const size_t stride = vision->Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE;
    size_t tokens = 0;
    for (const auto& im : m.images) tokens += (size_t)im.grid_h * im.grid_w / 4;
    rows.assign(tokens * desc.D, bf16(0.0f));
    const size_t per_patch = (size_t)3 * 2 * 16 * 16;
    size_t first = 0, row0 = 0, pix0 = 0;
    while (first < m.images.size()) {
        // images [first, last) form one call
        size_t last = first, patches = 0;
        while (last < m.images.size()) {
            const size_t p = (size_t)m.images[last].grid_h * m.images[last].grid_w;
            if (last > first && patches + p > vision_max_patches()) break;
            patches += p;
            last++;
        }
        d3_image_payload part;
        const d3_image_payload* call = &m;
        if (first != 0 || last != m.images.size()) {
            part.images.assign(m.images.begin() + first, m.images.begin() + last);
            part.pixels.assign(m.pixels.begin() + pix0, m.pixels.begin() + pix0 + patches * per_patch);
            part.image_token_id = m.image_token_id;
            call = &part;
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<bf16>> f = vision->encode(*call);
        size_t part_tokens = 0;
        for (const auto& im : call->images) part_tokens += (size_t)im.grid_h * im.grid_w / 4;
        if (f.empty() || stride < (size_t)desc.D || f.back().size() < part_tokens * stride)
            throw std::runtime_error("d3_npu: the vision tower returned fewer rows than the images hold");
        for (size_t t = 0; t < part_tokens; t++)
            memcpy(rows.data() + (row0 + t) * desc.D, f.back().data() + t * stride, desc.D * sizeof(bf16));
        if (timing)
            std::fprintf(stderr, "vision_call,images=%zu,patches=%zu,tokens=%zu,ms=%.1f\n", last - first, patches, part_tokens,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (const char* dump = std::getenv("D3_VISION_DUMP")) {
            // debug / tests: per tower call int32 {images, tokens} then the merger rows bf16 [tokens][D], appended to the file
            if (FILE* df = std::fopen(dump, "ab")) {
                const int32_t head[2] = {(int32_t)(last - first), (int32_t)part_tokens};
                std::fwrite(head, sizeof(int32_t), 2, df);
                std::fwrite(rows.data() + row0 * desc.D, sizeof(bf16), part_tokens * desc.D, df);
                std::fclose(df);
            }
        }
        row0 += part_tokens;
        pix0 += patches * per_patch;
        first = last;
    }
}

void d3_npu::Impl::_prepare_media(const std::vector<std::vector<int>>& ids) {
    // `media` is set: check that every prompt holds its image runs, build the vision tower on first use and
    // run it; `features` holds the merger rows of all images in order
    _check_image_prompts(ids, *media);
    _vision_rows(*media, features);
}

std::vector<buffer<bf16>> d3_npu::Impl::encode_batch(const std::vector<std::vector<int>>& ids, const d3_image_payload* payload) {
    if (!loaded) throw std::runtime_error("d3_npu: load_weights() was not called");
    for (const auto& p : ids) {
        if (p.empty() || (int)p.size() > MAX_L) {
            throw std::length_error("d3_npu: prompt of " + std::to_string(p.size()) + " tokens, accepted 1.." + std::to_string(MAX_L));
        }
    }
    state_len = nullptr;
    if (payload && !payload->state_len.empty()) {
        if (payload->state_len.size() != ids.size())
            throw std::invalid_argument("d3_npu: state_len holds " + std::to_string(payload->state_len.size()) +
                                        " entries for " + std::to_string(ids.size()) + " prompts");
        state_len = &payload->state_len;
    }
    std::vector<buffer<bf16>> out;
    out.reserve(ids.size());
    media = (payload && !payload->images.empty()) ? payload : nullptr;
    if (media) _prepare_media(ids);
#ifdef D3_RIG
    d3_cpu::rig_active() = (ids.size() == 1);
    if (media) D3_RIG_DUMP("vision.feat", features.data(), features.size() / desc.D, desc.D, desc.D);
#endif
    // Greedy packing in input order: a pass holds prompts while their rows fit the capacity of
    // its first prompt's 512-row tiles (at least pack_rows). Packing never reorders prompts.
    size_t i = 0;
    while (i < ids.size()) {
        const int tile = d3_npu_sequence::MM_ROWS;
        const int first_len = (int)ids[i].size();
        int cap = (first_len + tile - 1) / tile * tile;
        if (pack_rows > cap) cap = pack_rows;
        size_t j = i + 1;
        int rows = first_len;
        if (pack_rows > 0) {
            while (j < ids.size() && rows + (int)ids[j].size() <= cap) rows += (int)ids[j++].size();
        }
        _encode_pack(ids, i, j, out);
        i = j;
    }
    _trim();
    media = nullptr;
    state_len = nullptr;
#ifdef D3_RIG
    d3_cpu::rig_active() = false;
#endif
    return out;
}


// ===============================================
// state-once
// ===============================================
// append the pass's positions to $D3_POS_DUMP (debug / tests): per segment int32 {kind (0 state, 1 question), index, len}
// followed by len x 3 int32 (t, h, w)
static void dump_positions(const char* path, int kind, size_t index, const int* pos, int len) {
    FILE* f = std::fopen(path, "ab");
    if (!f) return;
    const int32_t head[3] = {kind, (int32_t)index, len};
    std::fwrite(head, sizeof(int32_t), 3, f);
    std::fwrite(pos, sizeof(int32_t), (size_t)len * 3, f);
    std::fclose(f);
}

void d3_npu::Impl::_encode_state_pass(const d3_state_request& req, const std::vector<int>& state_pos, int next_pos,
                                      bool with_state, size_t first_q, size_t last_q, std::vector<buffer<bf16>>& out) {
    const int D = desc.D;
    const char* pos_dump = std::getenv("D3_POS_DUMP");
    auto visual = [&](int id, int image_id) { return id == image_id || (video_id >= 0 && id == video_id); };
    std::vector<int> lens;
    if (with_state) lens.push_back((int)req.state_ids.size());
    for (size_t i = first_q; i < last_q; i++) lens.push_back((int)req.questions[i].ids.size());
    d3_batch b;
    b.build_state_once(lens, with_state, next_pos, cache.get());
    if (residual.size() < (size_t)b.R_padded * D) residual = buffer<bf16>((size_t)b.R_padded * D);
    size_t seg = 0;
    if (with_state) {
        const auto& s = b.segs[seg++];
        const bf16* row = features.data();
        for (int l = 0; l < s.len; l++) {
            bf16* dst = residual.data() + (size_t)(s.begin + l) * D;
            if (media && visual(req.state_ids[l], media->image_token_id)) {
                memcpy(dst, row, (size_t)D * sizeof(bf16));
                row += D;
            } else {
                _embed(req.state_ids[l], dst);
            }
        }
        if (media) memcpy(b.input_idx.data() + (size_t)s.begin * 3, state_pos.data(), (size_t)s.len * 3 * sizeof(int));
        if (pos_dump) dump_positions(pos_dump, 0, 0, b.input_idx.data() + (size_t)s.begin * 3, s.len);
    }
    for (size_t i = first_q; i < last_q; i++) {
        const auto& s = b.segs[seg++];
        const auto& q = req.questions[i];
        if (q.image_ids.empty()) {
            for (int l = 0; l < s.len; l++) _embed(q.ids[l], residual.data() + (size_t)(s.begin + l) * D);
        } else {
            // images inside the block: their merger rows (one tower run per distinct image, shared by every use) and
            // the planner's positions, which start at the block's first position
            size_t k = 0, j = 0;
            for (int l = 0; l < s.len; l++) {
                bf16* dst = residual.data() + (size_t)(s.begin + l) * D;
                if (visual(q.ids[l], qimage_id)) {
                    const auto& g = qgrids[i][k];
                    const size_t n = (size_t)g.grid_h * g.grid_w / 4;
                    memcpy(dst, qfeatures.data() + (qrow[qmap[q.image_ids[k]]] + j) * D, (size_t)D * sizeof(bf16));
                    if (++j == n) { j = 0; k++; }
                } else {
                    _embed(q.ids[l], dst);
                }
            }
            d3_plan_positions(q.ids, qimage_id, video_id, qgrids[i].data(), qgrids[i].size(), next_pos,
                              b.input_idx.data() + (size_t)s.begin * 3);
        }
        if (pos_dump) dump_positions(pos_dump, 1, i, b.input_idx.data() + (size_t)s.begin * 3, s.len);
    }

    for (int layer = 0; layer < (int)desc.num_hidden_layers; layer++) {
        pctx->forward(layer, b, residual, proj_weights[layer], rms_weights[layer], rope_rms_weights[layer],
                      alpha_proj_cpu[layer], beta_proj_cpu[layer]);
    }

    seg = with_state ? 1 : 0;
    for (size_t i = first_q; i < last_q; i++) {
        const auto& s = b.segs[seg++];
        const auto& offs = req.questions[i].readout_from_end;
        const size_t n = offs.empty() ? 1 : offs.size();
        buffer<bf16> h(n * D);
        for (size_t r = 0; r < n; r++) {
            const int off = offs.empty() ? 0 : offs[r];
            cpu_func::_rms_norm(h.data() + r * D, residual.data() + (size_t)(s.begin + s.len - 1 - off) * D, final_norm.data(), D, 1, 0, 0);
        }
        out.push_back(std::move(h));
    }
}

static uint64_t fnv64(const void* data, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ull; }
    return h;
}

std::vector<buffer<bf16>> d3_npu::Impl::encode_state_once(const d3_state_request& req) {
    if (!loaded) throw std::runtime_error("d3_npu: load_weights() was not called");
    if (pctx->attn_block->causal) throw std::invalid_argument("d3_npu: encode_state_once does not run with set_causal(true)");
    const int S = (int)req.state_ids.size();
    if (S < 1) throw std::invalid_argument("d3_npu: encode_state_once needs a state block");
    const d3_image_payload& qi = req.question_images;
    const bool have_qi = !qi.images.empty();
    if (!have_qi && !qi.pixels.empty()) throw std::invalid_argument("d3_npu: question_images holds pixels but no images");
    if (have_qi) _check_images(qi, "question ");
    const int qid = have_qi ? qi.image_token_id : (req.media ? req.media->image_token_id : qi.image_token_id);
    const d3_image_payload* m = (req.media && !req.media->images.empty()) ? req.media : nullptr;
    if (m) _check_images(*m, "");
    // the state's runs: the state's own images / video groups (the pad ids of media), none without media
    if (m) d3_check_visual_runs(req.state_ids, m->image_token_id, req.video_token_id, m->images.data(), m->images.size(), "the state");
    else if (req.video_token_id >= 0)
        for (int id : req.state_ids)
            if (id == req.video_token_id) throw std::invalid_argument("d3_npu: the state holds video token runs but no media");
    std::vector<std::vector<d3_image>> grids(req.questions.size());
    for (size_t i = 0; i < req.questions.size(); i++) {
        const auto& q = req.questions[i];
        const std::string at = "question " + std::to_string(i) + ": ";
        if (q.ids.empty()) throw std::invalid_argument("d3_npu: " + at + "empty question block");
        if ((int64_t)S + (int64_t)q.ids.size() > MAX_L)
            throw std::length_error("d3_npu: " + at + "prompt of " + std::to_string((int64_t)S + (int64_t)q.ids.size()) +
                                    " tokens, accepted 1.." + std::to_string(MAX_L));
        for (int off : q.readout_from_end)
            if (off < 0 || off >= (int)q.ids.size())
                throw std::invalid_argument("d3_npu: " + at + "readout offset " + std::to_string(off) + " outside the block");
        for (int id : q.image_ids)
            if (id < 0 || id >= (int)qi.images.size())
                throw std::invalid_argument("d3_npu: " + at + "image id " + std::to_string(id) + " outside question_images");
        for (int id : q.image_ids) grids[i].push_back(qi.images[id]);
        if (!q.image_ids.empty() && !have_qi) throw std::invalid_argument("d3_npu: " + at + "image_ids without question_images");
        // pad runs of the block <-> its image_ids; a pad of the state's media id in a block without images is an error too
        bool pads = false;
        for (int id : q.ids) pads = pads || id == qid || id == (m ? m->image_token_id : qid) || (req.video_token_id >= 0 && id == req.video_token_id);
        if (pads && grids[i].empty()) throw std::invalid_argument("d3_npu: " + at + "image tokens inside a question block without image_ids");
        if (!grids[i].empty()) {
            if (m && m->image_token_id != qid)
                for (int id : q.ids) if (id == m->image_token_id) throw std::invalid_argument("d3_npu: " + at + "image tokens of the state's id inside a question block");
            d3_check_visual_runs(q.ids, qid, req.video_token_id, grids[i].data(), grids[i].size(), "a question block");
        }
    }
    if (req.questions.empty()) return {};

    // whatever happens below, nothing of this call outlives it
    struct Cleanup {
        d3_npu::Impl* e;
        ~Cleanup() {
            e->cache.reset(); e->media = nullptr; e->state_len = nullptr; e->video_id = -1;
            std::vector<bf16>().swap(e->qfeatures); e->qrow.clear(); e->qmap.clear(); e->qgrids.clear();
            e->_trim();
        }
    } cleanup{this};
    state_len = nullptr;
    video_id = req.video_token_id;
    qimage_id = qid;
    media = m;
    if (media) _vision_rows(*media, features);
    if (have_qi) {
        // distinct images only: equal grid and equal pixels give equal rows, and an option image shared by 100
        // questions costs one tower run
        const size_t per_patch = (size_t)3 * 2 * 16 * 16;
        std::vector<size_t> pix_off(qi.images.size() + 1, 0);
        for (size_t i = 0; i < qi.images.size(); i++) pix_off[i + 1] = pix_off[i] + (size_t)qi.images[i].grid_h * qi.images[i].grid_w * per_patch;
        qmap.assign(qi.images.size(), -1);
        std::vector<size_t> distinct;          // entries of the table that are the first of their kind
        std::multimap<uint64_t, size_t> seen;  // hash -> index into `distinct`
        for (size_t i = 0; i < qi.images.size(); i++) {
            uint64_t h = fnv64(&qi.images[i].grid_h, sizeof(int));
            h = fnv64(&qi.images[i].grid_w, sizeof(int), h);
            h = fnv64(qi.pixels.data() + pix_off[i], (pix_off[i + 1] - pix_off[i]) * sizeof(bf16), h);
            int found = -1;
            for (auto it = seen.equal_range(h); it.first != it.second; ++it.first) {
                const size_t d = distinct[it.first->second];
                if (qi.images[d].grid_h == qi.images[i].grid_h && qi.images[d].grid_w == qi.images[i].grid_w &&
                    !memcmp(qi.pixels.data() + pix_off[d], qi.pixels.data() + pix_off[i], (pix_off[i + 1] - pix_off[i]) * sizeof(bf16))) {
                    found = (int)it.first->second;
                    break;
                }
            }
            if (found < 0) {
                found = (int)distinct.size();
                distinct.push_back(i);
                seen.emplace(h, (size_t)found);
            }
            qmap[i] = found;
        }
        d3_image_payload uniq;
        const d3_image_payload* table = &qi;
        if (distinct.size() != qi.images.size()) {
            uniq.image_token_id = qi.image_token_id;
            for (size_t d : distinct) {
                uniq.images.push_back(qi.images[d]);
                uniq.pixels.insert(uniq.pixels.end(), qi.pixels.begin() + pix_off[d], qi.pixels.begin() + pix_off[d + 1]);
            }
            table = &uniq;
        }
        qrow.assign(table->images.size(), 0);
        size_t rows = 0;
        for (size_t u = 0; u < table->images.size(); u++) {
            qrow[u] = rows;
            rows += (size_t)table->images[u].grid_h * table->images[u].grid_w / 4;
        }
        _vision_rows(*table, qfeatures);
    }
    qgrids = std::move(grids);

    std::vector<int> state_pos;
    int next_pos = S;   // text: the question block continues at S
    if (media) {
        state_pos.assign((size_t)S * 3, 0);
        next_pos = d3_plan_positions(req.state_ids, media->image_token_id, video_id, media->images.data(), media->images.size(), 0, state_pos.data());
    }
    cache = std::make_unique<d3_state_cache>();
    cache->state_len = S;
    cache->layers.resize(desc.num_hidden_layers);

    // Pass 0: the state and the questions that fit into the padding of its last 512-row tile (and up to
    // pack_rows); later passes: questions only, packed greedily like prompts. Packing never reorders.
    const size_t N = req.questions.size();
    const int tile = d3_npu_sequence::MM_ROWS;
    std::vector<buffer<bf16>> out;
    out.reserve(N);
    size_t qi_next = 0;
    bool first = true;
    while (first || qi_next < N) {
        const int first_len = first ? S : (int)req.questions[qi_next].ids.size();
        int cap = (first_len + tile - 1) / tile * tile;
        if (pack_rows > cap) cap = pack_rows;
        size_t j = first ? qi_next : qi_next + 1;
        int rows = first_len;
        if (pack_rows > 0) {
            while (j < N && rows + (int)req.questions[j].ids.size() <= cap) rows += (int)req.questions[j++].ids.size();
        }
        _encode_state_pass(req, state_pos, next_pos, first, qi_next, j, out);
        qi_next = j;
        first = false;
    }
    return out;
}

// ===============================================
// Externals
// ===============================================
// The host worker pool (d3_par.hpp) lives while an engine does: the last engine to go joins its threads, so
// unloading the library never leaves a thread running its code.
static std::atomic<int> g_live_engines{0};

d3_npu::d3_npu(LM_Config config, npu_xclbin_manager* npu_instance, int max_length) {
    this->_impl = new Impl(config, npu_instance, max_length);
    g_live_engines++;
}

d3_npu::~d3_npu() {
    delete this->_impl;
    if (--g_live_engines == 0) d3_par::pool().shutdown();
}

void d3_npu::load_weights(Q4NX& q4nx) { this->_impl->load_weights(q4nx); }

// A request owns the array from its vision tower to its last layer when the prefill runs switches (d3_array_lock).
std::vector<buffer<bf16>> d3_npu::encode_batch(const std::vector<std::vector<int>>& ids, void* payload) {
    d3_array_lock array(this->_impl->pctx->ll.possible());
    return this->_impl->encode_batch(ids, static_cast<const d3_image_payload*>(payload));
}

std::vector<buffer<bf16>> d3_npu::encode_state_once(const d3_state_request& req) {
    d3_array_lock array(this->_impl->pctx->ll.possible());
    return this->_impl->encode_state_once(req);
}

extern "C" int d3_npu_abi_version() { return D3_NPU_ABI_VERSION; }
extern "C" int d3_npu_abi_minor() { return D3_NPU_ABI_MINOR; }
extern "C" unsigned d3_npu_capabilities() {
    return D3_CAP_STATE_ONCE | D3_CAP_READOUT_ROWS | D3_CAP_QUESTION_IMAGES | D3_CAP_VIDEO;
}

buffer<bf16> d3_npu::encode(const std::vector<int>& ids, void* payload) {
    return std::move(this->encode_batch({ids}, payload)[0]);   // a copy would dangle under XRT
}

int d3_npu::max_length() { return this->_impl->MAX_L; }

int d3_npu::hidden_size() { return (int)this->_impl->desc.D; }

void d3_npu::set_pack_rows(int rows) {
    this->_impl->pack_rows = rows < 0 ? 0 : rows;
    this->_impl->pctx->set_scratch_rows(std::max(Impl::CACHE_ROWS, this->_impl->pack_rows));
    this->_impl->_trim();
}

void d3_npu::set_causal(bool causal) { this->_impl->pctx->attn_block->causal = causal; }

void d3_vision_host_attention(bf16* out, const bf16* q, const bf16* k, const bf16* v, int L, int heads, int DH, int kv_stride,
                              float scale, std::vector<float>& kt_buf) {
    d3_cpu::attention(out, q, k, v, L, L, heads, heads, DH, 0, 0, kv_stride, kv_stride, scale, false, kt_buf);
}
