// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lfm3_npu_detail.hpp"
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <filesystem>

using namespace lfm3_cpu;

/// \file lfm3_decode_npu.cpp
/// NPU decode of the LFM3 engine. The decode layer xclbin (FLM_Xclbin/LiquidAI/lfm3_npu_bin) runs a whole layer for one
/// token. Conv layers run on it; attention layers still run on the host until their kernels land.

void lfm3_npu::Impl::init_decode() {
    if (!npu) return;
    // LFM3_DECODE: host | conv (conv layers + lm_head on the NPU, attention on the host) | full (default: attention too)
    std::string mode = "full";
    if (const char* e = std::getenv("LFM3_DECODE")) mode = e;
    if (mode == "host") return;
    const std::string path = utils::path_join(config.exec_path, "xclbins", config.model_name, "layer.xclbin");
    if (!std::filesystem::exists(path)) {
        header_print("FLM", "lfm3: no layer.xclbin in " + path + ", decoding on the host");
        return;
    }
    layer_mgr = npu->register_xclbin(path);
    conv_app = layer_mgr->create_app();
    attn_full_app = layer_mgr->create_app();
    attn_swa_app = layer_mgr->create_app();
    lm_app = layer_mgr->create_app();
    decode_attn = mode == "full";
    if (const char* e = std::getenv("LFM3_ATTN_NPU")) { const std::string k = e; attn_npu_full = k.find("full") != std::string::npos; attn_npu_swa = k.find("swa") != std::string::npos; }
    kv_owner.assign(desc.num_hidden_layers, 0);
    decode_conv = mode != "head";             // "head": only the lm_head runs on the NPU (isolates the layer stages for accuracy studies)
    dseq = std::make_unique<lfm3_npu_sequence>(desc, MAX_L);
    decode_verify = std::getenv("LFM3_DECODE_VERIFY") != nullptr;

    const uint32_t D = desc.D, vp = dseq->vocab_padded;
    d_x = conv_app.create_bo_buffer<bf16>(D);
    d_logits = lm_app.create_bo_buffer<bf16>(vp);
    d_final_norm = lm_app.create_bo_buffer<bf16>((size_t)D * lfm3_npu_sequence::NORMS_PER_LAYER);
    d_lm_w = lm_app.create_bo_buffer<u8>(lfm3_desc::qbytes(vp, D));
    dseq->gen_lm_head_seq(lm_app.seq());
    lm_app.update_ctrl_seq();
    lm_run = lm_app.create_run(d_logits, d_lm_w, d_x, d_final_norm);
    decode_npu = true;
    header_print("FLM", std::string("lfm3 decode: ") + (decode_attn ? "all layers" : "conv layers") + " + lm_head on the NPU" + (decode_verify ? " (verify against the host)" : ""));
}

/// The tied lm_head is a true Q4_1 (unsigned nibbles u, w = u*d + m). The decode matvec kernel reads signed nibbles and
/// adds min' * sum(x) per 32-column group, so store s = u - 8 (flip bit 3 of both nibbles of every byte) and
/// min' = m + 8 d. Padded rows are zero blocks (w = 0).
static void lfm3_lm_head_to_device(uint8_t* dst, const uint8_t* src_plain, size_t real_row_blocks, size_t padded_row_blocks, int cols) {
    const size_t cbs = cols / QB_COLS;
    bytes tmp(padded_row_blocks * cbs * QB_BYTES);
    std::memset(tmp.data(), 0, tmp.size());
    uint8_t* t = reinterpret_cast<uint8_t*>(tmp.data());
    std::memcpy(t, src_plain, real_row_blocks * cbs * QB_BYTES);
    #pragma omp parallel for num_threads(host_threads()) schedule(static)
    for (long b = 0; b < (long)(real_row_blocks * cbs); b++) {
        uint8_t* blk = t + (size_t)b * QB_BYTES;
        uint16_t* scale = reinterpret_cast<uint16_t*>(blk);
        uint16_t* mins = scale + 8 * 32;
        for (int i = 0; i < 8 * 32; i++) mins[i] = f2bf(bf2f(mins[i]) + 8.0f * bf2f(scale[i]));
        uint8_t* q = blk + 1024;
        for (int i = 0; i < 4096; i++) q[i] ^= 0x88;
    }
    lfm3_reorder_cpy(dst, tmp, cols);
}

void lfm3_npu::Impl::build_decode_weights() {
    const uint32_t D = desc.D, K = desc.conv_kernel, L = desc.num_hidden_layers;
    d_rms.assign(L, buffer<bf16>());
    d_taps.assign(L, buffer<bf16>());
    d_ring.assign(L, buffer<bf16>());
    d_rope.assign(L, buffer<bf16>());
    d_gate_w.assign(L, std::vector<float>());
    if (D % 16) throw std::runtime_error("lfm3_npu: hidden size must be a multiple of 16");
    d_gate_h.assign(D, 0.0f);
    for (uint32_t li = 0; li < L; li++) {
        const lfm3_layer_weights& w = layers[li];
        d_rms[li] = conv_app.create_bo_buffer<bf16>((size_t)4 * D);
        bf16* r = d_rms[li].data();
        std::memcpy(r + 0 * D, w.operator_norm.data(), D * sizeof(bf16));
        std::memcpy(r + 1 * D, w.post_operator_norm.data(), D * sizeof(bf16));
        std::memcpy(r + 2 * D, w.ffn_norm.data(), D * sizeof(bf16));
        std::memcpy(r + 3 * D, w.post_ffn_norm.data(), D * sizeof(bf16));
        d_rms[li].sync_to_device();
        if (w.is_attn()) {
            if (!decode_attn) continue;
            // cos/sin are rewritten every token; q_norm and k_norm are fixed. The DMA length is that of the conv taps (K x D).
            const uint32_t DH = desc.DH;
            d_rope[li] = attn_full_app.create_bo_buffer<bf16>((size_t)K * D);
            std::memset(d_rope[li].data(), 0, d_rope[li].size() * sizeof(bf16));
            std::memcpy(d_rope[li].data() + DH, w.q_norm.data(), DH * sizeof(bf16));
            std::memcpy(d_rope[li].data() + 2 * DH, w.k_norm.data(), DH * sizeof(bf16));
            d_rope[li].sync_to_device();
            // the 16-row gate projection as fp32: the per-token gate is 16 dot products of 2048, done in a few microseconds
            d_gate_w[li].resize((size_t)desc.num_heads * D);
            const uint16_t* gw = u16p(w.gate_proj.data());
            for (size_t i = 0; i < d_gate_w[li].size(); i++) d_gate_w[li][i] = bf2f(gw[i]);
        } else {
            d_taps[li] = conv_app.create_bo_buffer<bf16>((size_t)K * D);
            // host taps are [D][K]; the kernel wants K rows of D (row t = tap t, t = 0 the oldest input)
            const uint16_t* src = u16p(w.conv_taps.data());
            uint16_t* dst = u16p(d_taps[li].data());
            for (uint32_t t = 0; t < K; t++)
                for (uint32_t d = 0; d < D; d++) dst[(size_t)t * D + d] = src[(size_t)d * K + t];
            d_taps[li].sync_to_device();
            d_ring[li] = conv_app.create_bo_buffer<bf16>((size_t)(K - 1) * D);
            d_ring[li].memset((bf16)0.0f);
            d_ring[li].sync_to_device();
        }
    }
    std::memset(d_final_norm.data(), 0, d_final_norm.size() * sizeof(bf16));
    std::memcpy(d_final_norm.data(), final_norm.data(), (size_t)D * sizeof(bf16));
    d_final_norm.sync_to_device();
    lfm3_lm_head_to_device(reinterpret_cast<uint8_t*>(d_lm_w.data()), reinterpret_cast<const uint8_t*>(lm_head_blob.data()),
                           desc.vocab_size / QB_ROWS, dseq->vocab_padded / QB_ROWS, D);
    d_lm_w.sync_to_device();
}

// conv history: host rows are oldest first for the next position `ctx`; the device ring keeps the row of position t in slot t % R.
// The next position is t = ctx, so host row j (token ctx - R + j) sits in slot (ctx + j) % R.
void lfm3_npu::Impl::conv_host_to_ring() {
    const uint32_t D = desc.D, R = desc.conv_kernel - 1;
    for (size_t li = 0, ci = 0; li < layers.size(); li++) {
        if (layers[li].is_attn()) continue;
        const buffer<bf16>& hist = conv_hist[ci++];
        for (uint32_t j = 0; j < R; j++)
            std::memcpy(d_ring[li].data() + (size_t)((ctx + j) % R) * D, hist.data() + (size_t)j * D, (size_t)D * sizeof(bf16));
        d_ring[li].sync_to_device();
    }
    conv_state = conv_where::ring;
}

void lfm3_npu::Impl::conv_ring_to_host() {
    const uint32_t D = desc.D, R = desc.conv_kernel - 1;
    for (size_t li = 0, ci = 0; li < layers.size(); li++) {
        if (layers[li].is_attn()) continue;
        buffer<bf16>& hist = conv_hist[ci++];
        d_ring[li].sync_from_device();
        for (uint32_t j = 0; j < R; j++)
            std::memcpy(hist.data() + (size_t)j * D, d_ring[li].data() + (size_t)((ctx + j) % R) * D, (size_t)D * sizeof(bf16));
    }
    conv_state = conv_where::host;
}

/// cosine similarity of two bf16 rows
static double cos_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) { const double x = bf2f(a[i]), y = bf2f(b[i]); ab += x * y; aa += x * x; bb += y * y; }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

static double prof_gate = 0, prof_seqgen = 0, prof_run = 0;

static void kv_sync_report(const char* what, double ms) {
    static const bool prof = std::getenv("LFM3_PROFILE") != nullptr;
    if (prof) std::fprintf(stderr, "kv sync %s: %.1f ms\n", what, ms);
}

void lfm3_npu::Impl::kv_layer_to_host(size_t li) {
    if (kv_owner.size() <= li || kv_owner[li] != 1) return;
    kv_cache[li].sync_from_device();
    kv_owner[li] = 0;
}

void lfm3_npu::Impl::kv_layer_to_dev(size_t li) {
    if (kv_owner.size() <= li || kv_owner[li] != 0) return;
    kv_cache[li].sync_to_device();
    kv_owner[li] = 1;
}

void lfm3_npu::Impl::kv_to_host() {
    const auto t0 = std::chrono::steady_clock::now();
    bool any = false;
    for (size_t i = 0; i < kv_cache.size(); i++) if (desc.is_attn_layer(i) && kv_owner.size() > i && kv_owner[i] == 1) { kv_layer_to_host(i); any = true; }
    if (any) kv_sync_report("device->host", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void lfm3_npu::Impl::kv_to_dev() {
    const auto t0 = std::chrono::steady_clock::now();
    bool any = false;
    for (size_t i = 0; i < kv_cache.size(); i++) if (desc.is_attn_layer(i) && kv_owner.size() > i && kv_owner[i] == 0) { kv_layer_to_dev(i); any = true; }
    if (any) kv_sync_report("host->device", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

/// Instruction sequences of a token: they depend on the context length only (never on the layer), so one conv sequence and
/// one sequence per attention window kind serve all layers of a token.
void lfm3_npu::Impl::gen_decode_seqs(uint32_t L) {
    dseq->gen_conv_layer_seq(conv_app.seq(), L);
    conv_app.update_ctrl_seq();
    if (decode_attn) {
        bool need_full = false, need_swa = false;
        for (uint32_t li = 0; li < desc.num_hidden_layers; li++) {
            need_full |= layers[li].kind == lfm3_block_kind::full_attention && attn_npu_full;
            need_swa |= layers[li].kind == lfm3_block_kind::sliding_attention && attn_npu_swa;
        }
        if (need_full) { dseq->gen_attn_layer_seq(attn_full_app.seq(), L, 0); attn_full_app.update_ctrl_seq(); }
        if (need_swa) { dseq->gen_attn_layer_seq(attn_swa_app.seq(), L, desc.sliding_window); attn_swa_app.update_ctrl_seq(); }
    }
    seq_gen_L = (int)L;
}

buffer<bf16> lfm3_npu::Impl::decode_token(int id) {
    if (!weights_loaded) throw std::runtime_error("lfm3_npu: weights are not loaded");
    if ((uint32_t)(ctx + 1) > MAX_L) throw std::runtime_error("lfm3_npu: context would exceed MAX_L (" + std::to_string(MAX_L) + ")");
    if (id < 0 || (uint32_t)id >= desc.vocab_size) throw std::runtime_error("lfm3_npu: token id out of range");
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    static const bool trace = std::getenv("LFM3_DECODE_TRACE") != nullptr;
    static const bool verify_all = std::getenv("LFM3_VERIFY_ALL") != nullptr;   // print every layer, not only the first tokens
    const int D = desc.D;
    std::vector<uint16_t> x(D), h(D), mix(D), ref(D);
    std::memcpy(x.data(), u16p(embedding.w.data()) + (size_t)id * D, (size_t)D * sizeof(uint16_t));
    std::vector<uint16_t> rope_tab_store(2 * (size_t)desc.DH);
    uint16_t* rope_tab[2] = {rope_tab_store.data(), rope_tab_store.data() + desc.DH};               // cos|sin of this position per attention kind (full, sliding), filled on first use
    bool rope_valid[2] = {false, false};
    // x lives in two places: the host vector and d_x. host_valid / dev_valid say which copy is current.
    bool host_valid = true, dev_valid = false;
    static const bool no_residency = std::getenv("LFM3_NO_RESIDENCY") != nullptr;   // bisect: always copy x through the host
    auto need_host = [&] { if (!host_valid) { d_x.sync_from_device(); std::memcpy(x.data(), d_x.data(), (size_t)D * sizeof(uint16_t)); host_valid = true; } };
    auto need_dev = [&] { if (!dev_valid || no_residency) { std::memcpy(d_x.data(), x.data(), (size_t)D * sizeof(uint16_t)); d_x.sync_to_device(); dev_valid = true; } };

    if (decode_conv && conv_state == conv_where::host) conv_host_to_ring();
    if (!decode_conv) ensure_conv_host();
    const uint32_t L = (uint32_t)ctx + 1;
    if (seq_gen_L != (int)L) {
        const auto tg0 = clk::now();
        gen_decode_seqs(L);
        prof_seqgen += ms(tg0, clk::now());
    }

    for (uint32_t li = 0; li < desc.num_hidden_layers; li++) {
        const auto t0 = clk::now();
        const lfm3_layer_weights& w = layers[li];
        if (!w.is_attn() && decode_conv) {
            if (decode_verify) {
                // host reference: rebuild the host history from the device ring (the ring itself is not touched, so the NPU
                // layer below still sees the pre-token state), run the host layer on a copy of x
                need_host();
                ref = x;
                conv_ring_to_host();
                host_layer(li, ref.data(), h.data(), mix.data(), 1, ctx);
                conv_state = conv_where::ring;      // the host rows are advanced now and stale; the ring is authoritative
            }
            need_dev();
            if (trace) std::fprintf(stderr, "decode trace: conv layer %u pos %d start\n", li, ctx);
            conv_app(d_x, layers[li].proj, d_rms[li], d_taps[li], d_ring[li]);
            if (trace) std::fprintf(stderr, "decode trace: conv layer %u done\n", li);
            host_valid = false;
            if (decode_verify) {
                static int shown = 0;
                need_host();
                const double c = cos_bf16(x.data(), ref.data(), D);
                if (c < 0.999 || shown < 4 * 13 || verify_all) { shown++; std::fprintf(stderr, "decode verify: layer %u pos %d cos(npu, host) = %.6f\n", li, ctx, c); }
            }
            decode_stats_ms_npu += ms(t0, clk::now());
        } else if (attn_on_npu(w.kind)) {
            const uint32_t DH = desc.DH;
            need_host();                                    // the gate comes from the layer input
            if (decode_verify) {
                ref = x;
                kv_layer_to_host(li);                       // the host layer reads rows 0..ctx-1 and writes row ctx
                host_layer(li, ref.data(), h.data(), mix.data(), 1, ctx);
            }
            // verify: keep the K/V row the host layer just computed, to compare with the row the NPU writes
            std::vector<uint16_t> host_k(desc.DK), host_v(desc.DK);
            if (decode_verify) {
                for (uint32_t kh = 0; kh < desc.num_kv_heads; kh++) {
                    std::memcpy(&host_k[(size_t)kh * desc.DH], kv_cache[li].data() + desc.kv_index(false, kh, ctx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
                    std::memcpy(&host_v[(size_t)kh * desc.DH], kv_cache[li].data() + desc.kv_index(true, kh, ctx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
                }
            }
            const auto ta0 = clk::now();
            kv_layer_to_dev(li);
            // gate = bf16(sigmoid(bf16(h . Wg))) with h = operator_norm(x). No OpenMP here: 32k multiply-adds are far cheaper
            // than waking 16 threads, and under a server the wake-up dominated (11 times per token).
            uint16_t gates[16];
            {
                float* hf = d_gate_h.data();
                float ss = 0.0f;
                for (int i = 0; i < D; i++) { const float v = bf2f(x[i]); ss += v * v; }
                const float r = 1.0f / std::sqrt(ss / (float)D + desc.eps);
                const uint16_t* nw = u16p(w.operator_norm.data());
                for (int i = 0; i < D; i++) hf[i] = bf2f(f2bf(bf2f(nw[i]) * bf2f(f2bf(bf2f(x[i]) * r))));   // bf16 RMSNorm, as rms_norm_vec
                const float* gw = d_gate_w[li].data();
                for (uint32_t hd = 0; hd < desc.num_heads; hd++) {
                    __m512 a = _mm512_setzero_ps();
                    for (int k = 0; k < D; k += 16) a = _mm512_fmadd_ps(_mm512_loadu_ps(gw + (size_t)hd * D + k), _mm512_loadu_ps(hf + k), a);
                    gates[hd] = f2bf(sigmoidf(bf2f(f2bf(_mm512_reduce_add_ps(a)))));
                }
            }
            // cos/sin of this position, once per kind and token (rope tables are the same for every layer of a kind)
            {
                const int kidx = w.kind == lfm3_block_kind::sliding_attention ? 1 : 0;
                if (!rope_valid[kidx]) {
                    std::vector<float> cs(DH / 2), sn(DH / 2);
                    desc.rope_cos_sin(w.kind, ctx, cs.data(), sn.data());
                    for (uint32_t i = 0; i < DH / 2; i++) { rope_tab[kidx][i] = f2bf(cs[i]); rope_tab[kidx][DH / 2 + i] = f2bf(sn[i]); }
                    rope_valid[kidx] = true;
                }
                uint16_t* rp = u16p(d_rope[li].data());
                std::memcpy(rp, rope_tab[kidx], DH * sizeof(uint16_t));
                for (uint32_t i = 0; i < desc.num_heads; i++) rp[3 * DH + i] = gates[i];
                d_rope[li].sync_to_device();
            }
            const auto ta1 = clk::now();
            npu_app& attn_app = w.kind == lfm3_block_kind::sliding_attention ? attn_swa_app : attn_full_app;
            need_dev();
            if (trace) std::fprintf(stderr, "decode trace: attn layer %u pos %d start\n", li, ctx);
            attn_app(d_x, layers[li].proj, d_rms[li], d_rope[li], kv_cache[li]);
            if (trace) std::fprintf(stderr, "decode trace: attn layer %u done\n", li);
            host_valid = false;
            prof_gate += ms(ta0, ta1); prof_run += ms(ta1, clk::now());
            if (decode_verify) {
                kv_cache[li].sync_from_device();            // the NPU wrote row ctx into the device copy
                std::vector<uint16_t> npu_k(desc.DK), npu_v(desc.DK);
                for (uint32_t kh = 0; kh < desc.num_kv_heads; kh++) {
                    std::memcpy(&npu_k[(size_t)kh * desc.DH], kv_cache[li].data() + desc.kv_index(false, kh, ctx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
                    std::memcpy(&npu_v[(size_t)kh * desc.DH], kv_cache[li].data() + desc.kv_index(true, kh, ctx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
                }
                const double ck = cos_bf16(npu_k.data(), host_k.data(), desc.DK), cv = cos_bf16(npu_v.data(), host_v.data(), desc.DK);
                static int kvshown = 0;
                if (ck < 0.999 || cv < 0.999 || kvshown++ < 11 || verify_all)
                    std::fprintf(stderr, "decode verify: layer %u pos %d KV row cos(npu, host): K %.6f V %.6f\n", li, ctx, ck, cv);
                // continue from the NPU's cache, like a real run: the host row is dropped (device is authoritative again)
                kv_owner[li] = 1;
            }
            if (decode_verify) {
                static int shown = 0;
                need_host();
                const double c = cos_bf16(x.data(), ref.data(), D);
                if (c < 0.999 || shown < 4 * 11 || verify_all) { shown++; std::fprintf(stderr, "decode verify: layer %u (attn, %s) pos %d cos(npu, host) = %.6f\n", li, w.kind == lfm3_block_kind::sliding_attention ? "swa" : "full", ctx, c); }
            }
            decode_stats_ms_npu += ms(t0, clk::now());
        } else {
            // host layer: the conv history is not involved, so the ring can stay where it is
            need_host();
            kv_layer_to_host(li);
            host_layer(li, x.data(), h.data(), mix.data(), 1, ctx);
            dev_valid = false;
            decode_stats_ms_host += ms(t0, clk::now());
        }
    }

    // lm_head on the NPU; while it runs, build the sequences of the next token (they only depend on the context length)
    const auto t1 = clk::now();
    need_dev();
    static const bool no_overlap = std::getenv("LFM3_NO_OVERLAP") != nullptr;       // bisect: generate the next sequences after the lm_head
    lm_run.start();
    ctx += 1;
    note_context();
    decode_stats_tokens++;
    if (no_overlap) flm::npu::wait_for_completion(lm_run, "lfm3 lm_head");
    if ((uint32_t)(ctx + 1) <= MAX_L) {
        const auto tg0 = clk::now();
        gen_decode_seqs((uint32_t)ctx + 1);
        prof_seqgen += ms(tg0, clk::now());
    }
    if (!no_overlap) flm::npu::wait_for_completion(lm_run, "lfm3 lm_head");
    d_logits.sync_from_device();
    buffer<bf16> logits(desc.vocab_size);
    std::memcpy(logits.data(), d_logits.data(), (size_t)desc.vocab_size * sizeof(bf16));
    if (decode_verify) {
        // the host lm_head on the same final hidden state (x is the last layer's output; d_x still holds it)
        need_host();
        std::vector<uint16_t> last(D), hl(desc.vocab_size);
        rms_norm_vec(last.data(), x.data(), u16p(final_norm.data()), D, desc.eps);
        matmul(hl.data(), desc.vocab_size, lm_head, last.data(), D, 1);
        const double c = cos_bf16(u16p(logits.data()), hl.data(), desc.vocab_size);
        int a = 0, b = 0; float va = -1e30f, vb = -1e30f;
        for (uint32_t i = 0; i < desc.vocab_size; i++) { const float p = bf2f(u16p(logits.data())[i]), q = bf2f(hl[i]); if (p > va) { va = p; a = i; } if (q > vb) { vb = q; b = i; } }
        static int shown = 0;
        if (shown++ < 12 || c < 0.999) std::fprintf(stderr, "decode verify: lm_head pos %d cos(npu, host) = %.6f  top1 npu %d host %d %s\n", ctx - 1, c, a, b, a == b ? "" : "DIFFERENT");
    }
    decode_stats_ms_head += ms(t1, clk::now());
    static const bool prof = std::getenv("LFM3_PROFILE") != nullptr;
    if (prof && decode_stats_tokens % 32 == 0) {
        std::fprintf(stderr, "decode profile attn layers (ms/token): gate+rope %.2f  run %.2f  | sequence gen+upload (hidden behind the lm_head) %.2f\n",
                     prof_gate / decode_stats_tokens, prof_run / decode_stats_tokens, prof_seqgen / decode_stats_tokens);
        std::fprintf(stderr, "decode profile (%d tokens): npu layers %.1f ms/tok, host layers %.1f ms/tok, lm_head (incl. overlapped seq gen) %.1f ms/tok\n", decode_stats_tokens,
                     decode_stats_ms_npu / decode_stats_tokens, decode_stats_ms_host / decode_stats_tokens, decode_stats_ms_head / decode_stats_tokens);
    }
    return logits;
}
