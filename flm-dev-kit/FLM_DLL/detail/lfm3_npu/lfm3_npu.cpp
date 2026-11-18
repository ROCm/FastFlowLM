// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include "lfm3_npu_detail.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>

using namespace lfm3_cpu;

lfm3_npu::Impl::Impl(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L)
    : config(config), npu(npu_instance) {
    this->desc.build(config);
    this->MAX_L = std::max<uint32_t>((uint32_t)MAX_L, 1u);
    // MAX_L is a power of two like the other engines (the attention kernel's cache layout is sized by it)
    if (npu_instance) {
        uint32_t m = 1;
        while (m < this->MAX_L) m <<= 1;
        this->MAX_L = m;
    }

    // NPU prefill stages: LFM3_PREFILL = host | mm | mm+attn | mm+attn+swa (default: all of them when a device is given)
    if (npu_instance) {
        std::string mode = "mm+attn+swa";
        if (const char* e = std::getenv("LFM3_PREFILL")) mode = e;
        if (const char* e = std::getenv("LFM3_PREFILL_MIN")) npu_prefill_min = std::atoi(e);
        if (mode != "host") {
            const std::string dir = utils::path_join(config.exec_path, "xclbins", config.model_name);
            bool present = true;
            for (const char* f : {"mm.xclbin", "dequant.xclbin", "attn.xclbin"})
                present = present && std::filesystem::exists(utils::path_join(dir, f));
            if (!present) {
                header_print("FLM", "lfm3: mm/dequant/attn xclbins not found in " + dir + ", running prefill on the host");
            } else {
                pctx = std::make_unique<lfm3_prefill_ctx>(npu_instance, &desc, config, this->MAX_L);
                stage_attn_full = mode.find("attn") != std::string::npos;
                stage_attn_swa = mode.find("swa") != std::string::npos;
                header_print("FLM", "lfm3 prefill stages: mm" + std::string(stage_attn_full ? "+attn" : "") + std::string(stage_attn_swa ? "+swa" : ""));
            }
        }
    }

    init_decode();

    const uint32_t L = desc.num_hidden_layers;
    this->layers.resize(L);
    this->kv_cache.resize(L);
    this->conv_index.assign(L, -1);
    this->attn_mask.resize(L);
    for (uint32_t i = 0; i < L; i++) this->attn_mask[i] = desc.is_attn_layer(i);
    int nconv = 0;
    for (uint32_t i = 0; i < L; i++) {
        layers[i].kind = desc.kinds[i];
        if (desc.is_attn_layer(i)) {
            kv_cache[i] = alloc_kv(2 * desc.kv_cache_elems(this->MAX_L) + (size_t)KV_SLACK_TOKENS * (desc.DK / 2));
            kv_cache[i].memset((bf16)0.0f);
            if (kv_on_device()) kv_cache[i].sync_to_device();
        } else {
            conv_index[i] = nconv++;
        }
    }
    conv_hist.resize(nconv);
    for (auto& h : conv_hist) {
        h = buffer<bf16>((size_t)(desc.conv_kernel - 1) * desc.D);
        h.memset((bf16)0.0f);
    }
    embedding = Embedding(desc.vocab_size, desc.D);
}

lfm3_npu::Impl::~Impl() = default;

void lfm3_npu::Impl::load_weights(Q4NX& q4nx) {
    const int D = desc.D, I = desc.intermediate_size, DQ = desc.DQ, DK = desc.DK;
    this->embedding.init_weights(&q4nx, "model.token_embd");
    lfm3_load_bf16(q4nx, final_norm, "model.embedding_norm.weight", D);
    // the tied embedding is stored a second time as a real Q4_1 matrix (unsigned nibbles, non-zero mins), plain block order
    q4nx.load_weights(lm_head_blob, "lm_head.weight");
    lm_head.base = reinterpret_cast<const uint8_t*>(lm_head_blob.data());
    lm_head.rows = desc.vocab_size;
    lm_head.cols = D;
    lm_head.is_signed = false;
    lm_head.vblocks = 1;
    if (lm_head_blob.size() != lm_head.bytes()) throw std::runtime_error("lfm3: lm_head.weight has the wrong size");

    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        lfm3_layer_weights& w = layers[i];
        const std::string n = "model.layers." + std::to_string(i) + ".";
        lfm3_load_bf16(q4nx, w.operator_norm, n + "operator_norm.weight", D);
        lfm3_load_bf16(q4nx, w.post_operator_norm, n + "post_operator_norm.weight", D);
        lfm3_load_bf16(q4nx, w.ffn_norm, n + "ffn_norm.weight", D);
        lfm3_load_bf16(q4nx, w.post_ffn_norm, n + "post_ffn_norm.weight", D);

        w.proj = this->alloc_proj(desc.proj_bytes());
        uint8_t* proj = reinterpret_cast<uint8_t*>(w.proj.data());
        const size_t cap = w.proj.size();
        // FFN: up (w3) and gate (w1) as one interleaved band, then down (w2)
        lfm3_load_up_gate(q4nx, desc, proj + desc.upgate_off(), n + "feed_forward.w3.weight", n + "feed_forward.w1.weight");
        lfm3_load_into(q4nx, proj + desc.down_off(), cap - desc.down_off(), n + "feed_forward.w2.weight", D, I);
        const size_t slice = desc.up_gate_slice_bytes();
        for (int g = 0; g < 2; g++) {                      // g = 0: up, 1: gate
            lfm3_cpu::qmat& m = g == 0 ? w.w3 : w.w1;
            m = lfm3_view(w.proj, desc.upgate_off() + (size_t)g * slice, I, D);
            m.slice_rb = lfm3_desc::UP_GATE_SLICE_ROWS / lfm3_cpu::QB_ROWS;
            m.slice_stride = 2 * slice;
        }
        w.w2 = lfm3_view(w.proj, desc.down_off(), D, I);

        if (w.is_attn()) {
            lfm3_load_into(q4nx, proj + desc.q_off(), cap - desc.q_off(), n + "self_attn.q_proj.weight", DQ, D);
            lfm3_load_into(q4nx, proj + desc.k_off(), cap - desc.k_off(), n + "self_attn.k_proj.weight", DK, D);
            lfm3_load_into(q4nx, proj + desc.v_off(), cap - desc.v_off(), n + "self_attn.v_proj.weight", DK, D);
            lfm3_load_into(q4nx, proj + desc.slot1_off(), cap - desc.slot1_off(), n + "self_attn.o_proj.weight", D, DQ);
            w.q_proj = lfm3_view(w.proj, desc.q_off(), DQ, D);
            w.k_proj = lfm3_view(w.proj, desc.k_off(), DK, D);
            w.v_proj = lfm3_view(w.proj, desc.v_off(), DK, D);
            w.o_proj = lfm3_view(w.proj, desc.slot1_off(), D, DQ);
            lfm3_load_bf16(q4nx, w.q_norm, n + "self_attn.q_norm.weight", desc.DH);
            lfm3_load_bf16(q4nx, w.k_norm, n + "self_attn.k_norm.weight", desc.DH);
            lfm3_load_bf16(q4nx, w.gate_proj, n + "self_attn.gate_proj.weight", (size_t)desc.num_heads * D);
        } else {
            lfm3_load_into(q4nx, proj + desc.slot0_off(), cap, n + "conv.in_proj.weight", 3 * D, D);
            lfm3_load_into(q4nx, proj + desc.slot1_off(), cap - desc.slot1_off(), n + "conv.out_proj.weight", D, D);
            w.conv_in = lfm3_view(w.proj, desc.slot0_off(), 3 * D, D);
            w.conv_out = lfm3_view(w.proj, desc.slot1_off(), D, D);
            lfm3_load_bf16(q4nx, w.conv_taps, n + "conv.conv.weight", (size_t)D * desc.conv_kernel);
        }
        this->sync_proj(w.proj);
    }
    this->weights_loaded = true;
    if (decode_npu) build_decode_weights();
    this->clear_context();
}

// ---------------------------------------------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------------------------------------------

void lfm3_npu::Impl::ffn_block(const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n) {
    const int D = desc.D, I = desc.intermediate_size;
    std::vector<uint16_t> a((size_t)n * I), b((size_t)n * I);
    matmul(a.data(), I, w.w1, h, D, n);          // gate
    matmul(b.data(), I, w.w3, h, D, n);          // up
    // g = bf16( bf16(silu(a)) * b )
    #pragma omp parallel for num_threads(host_threads()) schedule(static)
    for (long i = 0; i < (long)n * I; i++) {
        const uint16_t s = f2bf(siluf(bf2f(a[i])));
        a[i] = f2bf(bf2f(s) * bf2f(b[i]));
    }
    matmul(out, D, w.w2, a.data(), I, n);
}

/// conv block core on the host: bcx = in_proj output [n][3D] (B | C | x); writes y = C * conv(B * x) [n][D] and advances the history
void lfm3_npu::Impl::conv_core(int layer, const lfm3_layer_weights& w, const uint16_t* bcx, uint16_t* y, int n) {
    const int D = desc.D, K = desc.conv_kernel;
    // all = [K-1 history rows][n rows of B*x]; output row t reads rows t .. t+K-1
    buffer<bf16>& hist = conv_hist[conv_index[layer]];
    const int H = K - 1;
    std::vector<uint16_t> all((size_t)(H + n) * D);
    std::memcpy(all.data(), u16p(hist.data()), (size_t)H * D * sizeof(uint16_t));
    for (int t = 0; t < n; t++) {
        const uint16_t* row = &bcx[(size_t)t * 3 * D];
        uint16_t* dst = &all[(size_t)(H + t) * D];
        for (int d = 0; d < D; d++) dst[d] = f2bf(bf2f(row[d]) * bf2f(row[2 * D + d]));   // B * x
    }
    const uint16_t* taps = u16p(w.conv_taps.data());            // [D][K]
    #pragma omp parallel for num_threads(host_threads()) schedule(static)
    for (int t = 0; t < n; t++) {
        const uint16_t* row = &bcx[(size_t)t * 3 * D];
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += bf2f(taps[(size_t)d * K + k]) * bf2f(all[(size_t)(t + k) * D + d]);
            const uint16_t conv = f2bf(acc);                      // the conv output is bf16
            y[(size_t)t * D + d] = f2bf(bf2f(row[D + d]) * bf2f(conv));   // C * conv
        }
    }
    // the next call's history is the last K-1 rows of `all`
    std::memcpy(u16p(hist.data()), &all[(size_t)n * D], (size_t)H * D * sizeof(uint16_t));
}

void lfm3_npu::Impl::conv_block(int layer, const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n) {
    const int D = desc.D;
    std::vector<uint16_t> bcx((size_t)n * 3 * D), y((size_t)n * D);
    matmul(bcx.data(), 3 * D, w.conv_in, h, D, n);
    conv_core(layer, w, bcx.data(), y.data(), n);
    matmul(out, D, w.conv_out, y.data(), D, n);
}

/// q/k norm + RoPE for the n new tokens (q: [n][DQ], k: [n][DK], rows contiguous) and the K/V rows into the cache
void lfm3_npu::Impl::attn_norm_rope_cache(int layer, const lfm3_layer_weights& w, uint16_t* q, uint16_t* k, const uint16_t* v, int n, int pos0) {
    const int DH = desc.DH, nh = desc.num_heads, nkv = desc.num_kv_heads, DQ = desc.DQ, DK = desc.DK;
    uint16_t* kv = u16p(kv_cache[layer].data());
    const uint32_t ML = this->MAX_L;
    #pragma omp parallel num_threads(host_threads())
    {
        std::vector<float> cosv(DH / 2), sinv(DH / 2);
        #pragma omp for schedule(static)
        for (int t = 0; t < n; t++) {
            desc.rope_cos_sin(w.kind, pos0 + t, cosv.data(), sinv.data());
            for (int hd = 0; hd < nh; hd++) {
                uint16_t* qh = &q[(size_t)t * DQ + (size_t)hd * DH];
                rms_norm_vec(qh, qh, u16p(w.q_norm.data()), DH, desc.eps);
                rope_head(qh, cosv.data(), sinv.data(), DH);
            }
            for (int kh = 0; kh < nkv; kh++) {
                uint16_t* kk = &k[(size_t)t * DK + (size_t)kh * DH];
                rms_norm_vec(kk, kk, u16p(w.k_norm.data()), DH, desc.eps);
                rope_head(kk, cosv.data(), sinv.data(), DH);
                std::memcpy(kv + desc.kv_index(false, kh, pos0 + t, 0, ML), kk, (size_t)DH * sizeof(uint16_t));
                std::memcpy(kv + desc.kv_index(true, kh, pos0 + t, 0, ML), &v[(size_t)t * DK + (size_t)kh * DH], (size_t)DH * sizeof(uint16_t));
            }
        }
    }
}

/// attention for the n new tokens on the host: q [n][DQ] (normed, rotated), gate [n][nh] (sigmoid applied), o [n][DQ] = gate * attention
void lfm3_npu::Impl::attn_host_core(int layer, const lfm3_layer_weights& w, const uint16_t* q, const uint16_t* gate, uint16_t* o, int n, int pos0) {
    const int DH = desc.DH, nh = desc.num_heads, nkv = desc.num_kv_heads, DQ = desc.DQ;
    const int rep = nh / nkv;
    const bool sliding = w.kind == lfm3_block_kind::sliding_attention;
    const float scale = 1.0f / std::sqrt((float)DH);
    const uint16_t* kv = u16p(kv_cache[layer].data());
    const uint32_t ML = this->MAX_L;
    #pragma omp parallel num_threads(host_threads())
    {
        std::vector<float> scores(pos0 + n), qf(DH), acc(DH);
        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (int t = 0; t < n; t++) {
            for (int hd = 0; hd < nh; hd++) {
                const int pos = pos0 + t;
                // window: key j is visible when pos - j < sliding_window
                const int lo = sliding ? std::max(0, pos - (int)desc.sliding_window + 1) : 0;
                const int kh = hd / rep;
                const uint16_t* qh = &q[(size_t)t * DQ + (size_t)hd * DH];
                for (int d = 0; d < DH; d++) qf[d] = bf2f(qh[d]);
                float mx = -INFINITY;
                for (int j = lo; j <= pos; j++) {
                    const uint16_t* kj = kv + desc.kv_index(false, kh, j, 0, ML);
                    float sc = 0.0f;
                    for (int d = 0; d < DH; d++) sc += qf[d] * bf2f(kj[d]);
                    sc *= scale;
                    scores[j - lo] = sc;
                    mx = std::max(mx, sc);
                }
                float sum = 0.0f;
                for (int j = lo; j <= pos; j++) { scores[j - lo] = std::exp(scores[j - lo] - mx); sum += scores[j - lo]; }
                for (int d = 0; d < DH; d++) acc[d] = 0.0f;
                for (int j = lo; j <= pos; j++) {
                    const float p = bf2f(f2bf(scores[j - lo] / sum));       // probabilities are cast to bf16
                    const uint16_t* vj = kv + desc.kv_index(true, kh, j, 0, ML);
                    for (int d = 0; d < DH; d++) acc[d] += p * bf2f(vj[d]);
                }
                const float g = bf2f(gate[(size_t)t * nh + hd]);
                for (int d = 0; d < DH; d++) o[(size_t)t * DQ + (size_t)hd * DH + d] = f2bf(bf2f(f2bf(acc[d])) * g);
            }
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------
// LFM3_ATTN_EMU=<bits>: accuracy study only. Runs the host attention with some of the NPU kernel's numerics, to see which
// step costs accuracy. 1: scores rounded to bf16 (raw dot, then scaled, both rounded); 2: Q and K quantized to block floating
// point (blocks of 8 along the head dim, shared exponent, 7-bit magnitude) before the dot; 4: probabilities as bf16 numerators
// (exp(s - max) rounded to bf16, normalized at the end); 8: P and V quantized to block floating point along the token dim
// (blocks of 8 tokens); 16: final 1/l rounded to bf16.
// ---------------------------------------------------------------------------------------------------------------
static float bfp_q(float x, float scale) { float q = std::nearbyint(x / scale); q = std::max(-127.0f, std::min(127.0f, q)); return q * scale; }
static void bfp_block(float* v, int n, int stride = 1) {   // quantize n values (stride apart) with one shared exponent
    float mx = 0; for (int i = 0; i < n; i++) mx = std::max(mx, std::fabs(v[i * stride]));
    if (mx == 0) return;
    int e; std::frexp(mx, &e);                               // mx = m * 2^e, m in [0.5, 1)
    const float scale = std::ldexp(1.0f, e - 7);             // 7 magnitude bits below the leading one
    for (int i = 0; i < n; i++) v[i * stride] = bfp_q(v[i * stride], scale);
}

void lfm3_npu::Impl::attn_host_core_emu(int layer, const lfm3_layer_weights& w, const uint16_t* q, const uint16_t* gate, uint16_t* o, int n, int pos0, int emu) {
    const int DH = desc.DH, nh = desc.num_heads, nkv = desc.num_kv_heads, DQ = desc.DQ;
    const int rep = nh / nkv;
    const bool sliding = w.kind == lfm3_block_kind::sliding_attention;
    const float scale = 1.0f / std::sqrt((float)DH);
    const float scale_b = bf2f(f2bf(scale));
    const uint16_t* kv = u16p(kv_cache[layer].data());
    const uint32_t ML = this->MAX_L;
    #pragma omp parallel num_threads(host_threads())
    {
        std::vector<float> scores(pos0 + n), qf(DH), kf(DH), acc(DH), pv;
        #pragma omp for schedule(dynamic, 4) collapse(2)
        for (int t = 0; t < n; t++) {
            for (int hd = 0; hd < nh; hd++) {
                const int pos = pos0 + t;
                const int lo = sliding ? std::max(0, pos - (int)desc.sliding_window + 1) : 0;
                const int cnt = pos - lo + 1;
                const int kh = hd / rep;
                const uint16_t* qh = &q[(size_t)t * DQ + (size_t)hd * DH];
                for (int d = 0; d < DH; d++) qf[d] = bf2f(qh[d]);
                if (emu & 2) for (int d = 0; d < DH; d += 8) bfp_block(&qf[d], 8);
                float mx = -INFINITY;
                for (int j = lo; j <= pos; j++) {
                    const uint16_t* kj = kv + desc.kv_index(false, kh, j, 0, ML);
                    for (int d = 0; d < DH; d++) kf[d] = bf2f(kj[d]);
                    if (emu & 2) for (int d = 0; d < DH; d += 8) bfp_block(&kf[d], 8);
                    float sc = 0.0f;
                    for (int d = 0; d < DH; d++) sc += qf[d] * kf[d];
                    if (emu & 1) { sc = bf2f(f2bf(sc)); sc = bf2f(f2bf(sc * scale_b)); } else sc *= scale;
                    scores[j - lo] = sc;
                    mx = std::max(mx, sc);
                }
                float sum = 0.0f;
                for (int j = 0; j < cnt; j++) {
                    float e = std::exp(scores[j] - mx);
                    if (emu & 4) e = bf2f(f2bf(e));
                    scores[j] = e; sum += e;
                }
                for (int d = 0; d < DH; d++) acc[d] = 0.0f;
                if (emu & 8) {
                    // P and V in blocks of 8 tokens: shared exponent per block of 8 tokens (P per row, V per column)
                    for (int j0 = 0; j0 < cnt; j0 += 8) {
                        const int m8 = std::min(8, cnt - j0);
                        float pb[8], vb[8];
                        for (int i = 0; i < m8; i++) pb[i] = (emu & 4) ? scores[j0 + i] : scores[j0 + i];
                        bfp_block(pb, m8);
                        for (int d = 0; d < DH; d++) {
                            for (int i = 0; i < m8; i++) vb[i] = bf2f(kv[desc.kv_index(true, kh, lo + j0 + i, d, ML)]);
                            bfp_block(vb, m8);
                            float a = 0; for (int i = 0; i < m8; i++) a += pb[i] * vb[i];
                            acc[d] += a;
                        }
                    }
                } else {
                    for (int j = 0; j < cnt; j++) {
                        const uint16_t* vj = kv + desc.kv_index(true, kh, lo + j, 0, ML);
                        for (int d = 0; d < DH; d++) acc[d] += scores[j] * bf2f(vj[d]);
                    }
                }
                float inv = 1.0f / sum;
                if (emu & 16) inv = bf2f(f2bf(inv));
                const float g = bf2f(gate[(size_t)t * nh + hd]);
                for (int d = 0; d < DH; d++) o[(size_t)t * DQ + (size_t)hd * DH + d] = f2bf(bf2f(f2bf(acc[d] * inv)) * g);
            }
        }
    }
}

void lfm3_npu::Impl::attention_block(int layer, const lfm3_layer_weights& w, const uint16_t* h, uint16_t* out, int n, int pos0) {
    const int D = desc.D, nh = desc.num_heads, DQ = desc.DQ, DK = desc.DK;
    std::vector<uint16_t> q((size_t)n * DQ), k((size_t)n * DK), v((size_t)n * DK), g((size_t)n * nh), o((size_t)n * DQ);
    matmul(q.data(), DQ, w.q_proj, h, D, n);
    matmul(k.data(), DK, w.k_proj, h, D, n);
    matmul(v.data(), DK, w.v_proj, h, D, n);
    // accuracy study: LFM3_NOISE_QK / LFM3_NOISE_V = relative sigma of multiplicative noise on the q, k / v projections
    {
        static const float nqk = std::getenv("LFM3_NOISE_QK") ? (float)std::atof(std::getenv("LFM3_NOISE_QK")) : 0.0f;
        static const float nv = std::getenv("LFM3_NOISE_V") ? (float)std::atof(std::getenv("LFM3_NOISE_V")) : 0.0f;
        auto noisy = [](std::vector<uint16_t>& x, float sigma, uint64_t seed) {
            uint64_t st = seed * 6364136223846793005ull + 1442695040888963407ull;
            for (auto& e : x) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                const double u1 = ((st >> 11) + 1) * (1.0 / 9007199254740993.0);
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                const double u2 = ((st >> 11) + 1) * (1.0 / 9007199254740993.0);
                const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
                e = f2bf(bf2f(e) * (1.0f + sigma * (float)z));
            }
        };
        static uint64_t call = 0;
        if (nqk > 0) { noisy(q, nqk, 7 + call); noisy(k, nqk, 1000003 + call); }
        if (nv > 0) noisy(v, nv, 99991 + call);
        call++;
    }
    matmul_dense(g.data(), nh, u16p(w.gate_proj.data()), nh, D, h, D, n);
    for (auto& x : g) x = f2bf(sigmoidf(bf2f(x)));          // gate = bf16(sigmoid(bf16(h . Wg)))
    attn_norm_rope_cache(layer, w, q.data(), k.data(), v.data(), n, pos0);
    static const int emu = std::getenv("LFM3_ATTN_EMU") ? std::atoi(std::getenv("LFM3_ATTN_EMU")) : 0;
    if (emu) attn_host_core_emu(layer, w, q.data(), g.data(), o.data(), n, pos0, emu);
    else attn_host_core(layer, w, q.data(), g.data(), o.data(), n, pos0);
    matmul(out, D, w.o_proj, o.data(), DQ, n);
}

// ---------------------------------------------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------------------------------------------

void lfm3_npu::Impl::host_layer(uint32_t li, uint16_t* x, uint16_t* h, uint16_t* mix, int n, int pos0) {
    const int D = desc.D;
    const lfm3_layer_weights& w = layers[li];
    for (int t = 0; t < n; t++) rms_norm_vec(&h[(size_t)t * D], &x[(size_t)t * D], u16p(w.operator_norm.data()), D, desc.eps);
    if (w.is_attn()) attention_block(li, w, h, mix, n, pos0);
    else conv_block(li, w, h, mix, n);
    for (int t = 0; t < n; t++) rms_norm_vec(&mix[(size_t)t * D], &mix[(size_t)t * D], u16p(w.post_operator_norm.data()), D, desc.eps);
    residual_add(x, mix, (size_t)n * D);

    for (int t = 0; t < n; t++) rms_norm_vec(&h[(size_t)t * D], &x[(size_t)t * D], u16p(w.ffn_norm.data()), D, desc.eps);
    ffn_block(w, h, mix, n);
    for (int t = 0; t < n; t++) rms_norm_vec(&mix[(size_t)t * D], &mix[(size_t)t * D], u16p(w.post_ffn_norm.data()), D, desc.eps);
    residual_add(x, mix, (size_t)n * D);
}

buffer<bf16> lfm3_npu::Impl::run(const int* ids, int n, bool want_logits) {
    if (!weights_loaded) throw std::runtime_error("lfm3_npu: weights are not loaded");
    if ((uint32_t)(ctx + n) > MAX_L) throw std::runtime_error("lfm3_npu: context would exceed MAX_L (" + std::to_string(MAX_L) + ")");
    ensure_conv_host();
    kv_to_host();
    const int D = desc.D;

    std::vector<uint16_t> x((size_t)n * D), h((size_t)n * D), mix((size_t)n * D);
    for (int t = 0; t < n; t++) {
        if (ids[t] < 0 || (uint32_t)ids[t] >= desc.vocab_size) throw std::runtime_error("lfm3_npu: token id out of range");
        std::memcpy(&x[(size_t)t * D], u16p(embedding.w.data()) + (size_t)ids[t] * D, (size_t)D * sizeof(uint16_t));
    }
    // LFM3_DUMP_DIR=<dir>: write the residual stream (bf16, n x D) after the embedding and after every layer, for the
    // first call only. Used to find the first layer that disagrees with models_simple/simple/lfm3_simple.py.
    static const char* dump_dir = std::getenv("LFM3_DUMP_DIR");
    auto dump = [&](const std::string& name) {
        if (!dump_dir || dump_done) return;
        std::FILE* f = std::fopen((std::string(dump_dir) + "/" + name + ".bf16").c_str(), "wb");
        if (f) { std::fwrite(x.data(), sizeof(uint16_t), x.size(), f); std::fclose(f); }
    };
    dump("embed");
    for (uint32_t li = 0; li < desc.num_hidden_layers; li++) {
        host_layer(li, x.data(), h.data(), mix.data(), n, ctx);
        dump("layer" + std::to_string(li));
    }
    if (dump_dir) dump_done = true;
    this->ctx += n;
    note_context();
    if (!want_logits) return buffer<bf16>();

    std::vector<uint16_t> last(D);
    rms_norm_vec(last.data(), &x[(size_t)(n - 1) * D], u16p(final_norm.data()), D, desc.eps);
    buffer<bf16> logits(desc.vocab_size);
    matmul(u16p(logits.data()), desc.vocab_size, lm_head, last.data(), D, 1);
    return logits;
}

buffer<bf16> lfm3_npu::Impl::forward(int id) {
    if (decode_npu) return decode_token(id);
    return run(&id, 1, true);
}

buffer<bf16> lfm3_npu::Impl::prefill(std::vector<int>& ids, void* payload) {
    if (ids.empty()) throw std::runtime_error("lfm3_npu: empty prefill");
    ensure_conv_host();
    kv_to_host();
    if (pctx && (int)ids.size() >= npu_prefill_min) return prefill_npu(ids);
    const int total = (int)ids.size();
    buffer<bf16> logits;
    for (int begin = 0; begin < total; begin += CHUNK) {
        const int n = std::min(CHUNK, total - begin);
        const bool last = begin + n == total;
        buffer<bf16> r = run(ids.data() + begin, n, last);
        if (last) logits = std::move(r);   // move: copy-assigning a buffer makes a non-owning view (XRT build)
    }
    return logits;
}

void lfm3_npu::Impl::set_context_length(int L) {
    this->ctx = L;
    note_context();
}

void lfm3_npu::Impl::clear_context() {
    conv_state = conv_where::host;
    for (auto& h : conv_hist) h.memset((bf16)0.0f);
    // a cleared context must look like a fresh engine: zero the K/V rows what earlier requests left behind
    if (kv_on_device() && dirty_rows > 0) {
        kv_to_dev();
        kv_rows_zero(kv_cache, kv_layout(), 0, dirty_rows, &attn_mask);
    }
    dirty_rows = 0;
    std::fill(kv_owner.begin(), kv_owner.end(), 0);
    this->ctx = 0;
}

buffer<bf16> lfm3_npu::Impl::get_k_cache(int layer_idx, int idx) {
    kv_to_host();
    if (layer_idx < 0 || (uint32_t)layer_idx >= desc.num_hidden_layers || !desc.is_attn_layer(layer_idx)) return buffer<bf16>(0);
    buffer<bf16> r(desc.DK);
    for (uint32_t h = 0; h < desc.num_kv_heads; h++)
        std::memcpy(r.data() + (size_t)h * desc.DH, kv_cache[layer_idx].data() + desc.kv_index(false, h, idx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
    return r;
}

buffer<bf16> lfm3_npu::Impl::get_v_cache(int layer_idx, int idx) {
    kv_to_host();
    if (layer_idx < 0 || (uint32_t)layer_idx >= desc.num_hidden_layers || !desc.is_attn_layer(layer_idx)) return buffer<bf16>(0);
    buffer<bf16> r(desc.DV);
    for (uint32_t h = 0; h < desc.num_kv_heads; h++)
        std::memcpy(r.data() + (size_t)h * desc.DH, kv_cache[layer_idx].data() + desc.kv_index(true, h, idx, 0, MAX_L), (size_t)desc.DH * sizeof(bf16));
    return r;
}

int lfm3_npu::Impl::checkpoint() {
    // attention rows are addressed by position, so only the conv history and the length need saving
    ensure_conv_host();
    conv_ckpt.resize(conv_hist.size());
    for (size_t i = 0; i < conv_hist.size(); i++) {
        conv_ckpt[i] = buffer<bf16>(conv_hist[i].size());
        std::memcpy(conv_ckpt[i].data(), conv_hist[i].data(), conv_hist[i].size() * sizeof(bf16));
    }
    ckpt_valid = true;
    ckpt_ctx = ctx;
    return ckpt_ctx;
}

int lfm3_npu::Impl::restore() {
    if (!ckpt_valid) {
        header_print("FLM", "No valid checkpoint found!");
        return -1;
    }
    for (size_t i = 0; i < conv_hist.size(); i++)
        std::memcpy(conv_hist[i].data(), conv_ckpt[i].data(), conv_hist[i].size() * sizeof(bf16));
    conv_state = conv_where::host;
    ctx = ckpt_ctx;
    note_context();
    return ckpt_ctx;
}

void lfm3_npu::Impl::update_max_length(uint32_t new_max) {
    if (new_max <= MAX_L) return;
    kv_to_host();
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        if (!desc.is_attn_layer(i)) continue;
        buffer<bf16> nkv = alloc_kv(2 * desc.kv_cache_elems(new_max) + (size_t)KV_SLACK_TOKENS * (desc.DK / 2));
        nkv.memset((bf16)0.0f);
        // each of the four bands moves to its new offset
        const size_t band = (size_t)MAX_L * (desc.DK / 2);
        for (uint32_t b = 0; b < 2; b++) {
            std::memcpy(nkv.data() + desc.k_band_off(b, new_max), kv_cache[i].data() + desc.k_band_off(b, MAX_L), band * sizeof(bf16));
            std::memcpy(nkv.data() + desc.v_band_off(b, new_max), kv_cache[i].data() + desc.v_band_off(b, MAX_L), band * sizeof(bf16));
        }
        if (kv_on_device()) nkv.sync_to_device();
        kv_cache[i] = std::move(nkv);   // move, never copy-assign: XRT buffer copies don't own the allocation
    }
    MAX_L = new_max;
    if (pctx) pctx->set_max_length(new_max);
    if (dseq) { dseq->set_max_length(new_max); seq_gen_L = -1; }
}

void lfm3_npu::Impl::dump_layer(int li, const uint16_t* rows, int n) {
    static const char* dump_dir = std::getenv("LFM3_DUMP_DIR");
    if (!dump_dir || dump_done) return;
    std::FILE* f = std::fopen((std::string(dump_dir) + "/layer" + std::to_string(li) + ".bf16").c_str(), "wb");
    if (f) { std::fwrite(rows, sizeof(uint16_t), (size_t)n * desc.D, f); std::fclose(f); }
    if (li + 1 == (int)desc.num_hidden_layers) dump_done = true;
}

// ---------------------------------------------------------------------------------------------------------------
// saved states for the multi-slot prefix cache (flm_state_api.h)
// ---------------------------------------------------------------------------------------------------------------

kv_row_layout lfm3_npu::Impl::kv_layout() const {
    kv_row_layout l;
    l.DK = desc.DK; l.DV = desc.DV; l.striped = true;
    l.k47 = desc.k_band_off(1, MAX_L);
    l.v03 = desc.v_band_off(0, MAX_L);
    l.v47 = desc.v_band_off(1, MAX_L);
    return l;
}

std::unique_ptr<hybrid_engine_state> lfm3_npu::Impl::new_state() {
    if (!kv_on_device()) return nullptr;       // the shared row helpers sync device buffers
    const size_t conv_layers = desc.num_conv_layers();
    return kv_rows_new_state(desc.num_attn_layers(), kv_layout(), conv_layers * (size_t)(desc.conv_kernel - 1) * desc.D * sizeof(bf16));
}

int lfm3_npu::Impl::read_state(hybrid_engine_state& state, int kv_from) {
    if (!kv_on_device()) return -1;
    header_print_r("FLM", "Reading state at context length " + std::to_string(ctx) + ", K/V rows from " + std::to_string(kv_from));
    ensure_conv_host();                        // the conv history lives in the device ring after NPU decode steps
    kv_to_dev();                               // rows the host wrote since the last sync reach the device; the helpers read the device copy
    const int length = kv_rows_read(state, kv_cache, kv_layout(), ctx, kv_from, -1, &attn_mask);
    if (length < 0) return -1;
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        if (attn_mask[i]) continue;
        const buffer<bf16>& hist = conv_hist[conv_index[i]];
        state.state[i] = buffer<bf16>(hist.size());
        std::memcpy(state.state[i].data(), hist.data(), hist.size() * sizeof(bf16));
    }
    return length;
}

int lfm3_npu::Impl::write_state_kv(const hybrid_engine_state& state) {
    if (!kv_on_device()) return -1;
    kv_to_dev();
    return kv_rows_write(state, kv_cache, kv_layout(), MAX_L, &attn_mask);
}

int lfm3_npu::Impl::write_state(const hybrid_engine_state& state) {
    if (!kv_on_device()) return -1;
    // check the conv blobs before anything is written
    bool ok = state.state.size() == kv_cache.size();
    for (uint32_t i = 0; ok && i < desc.num_hidden_layers; i++) {
        if (!attn_mask[i] && state.state[i].size() != conv_hist[conv_index[i]].size()) ok = false;
    }
    kv_to_dev();
    const int length = ok ? kv_rows_write(state, kv_cache, kv_layout(), MAX_L, &attn_mask) : -1;
    if (length < 0) {
        header_print_r("FLM", "No valid state to write");
        return -1;
    }
    header_print_r("FLM", "Writing state at context length " + std::to_string(length));
    for (uint32_t i = 0; i < desc.num_hidden_layers; i++) {
        if (attn_mask[i]) continue;
        buffer<bf16>& hist = conv_hist[conv_index[i]];
        std::memcpy(hist.data(), state.state[i].data(), hist.size() * sizeof(bf16));
    }
    conv_state = conv_where::host;
    // rows past the length hold what the requests since the state was read left there: put back the zeros of a fresh engine
    kv_rows_zero(kv_cache, kv_layout(), static_cast<size_t>(length), dirty_rows, &attn_mask);
    dirty_rows = static_cast<size_t>(length);
    set_context_length(length);
    return length;
}

// ===============================================
// Externals for lfm3_npu
// ===============================================
lfm3_npu::lfm3_npu(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L) {
    this->_impl = new Impl(config, npu_instance, MAX_L);
}
buffer<bf16> lfm3_npu::forward(int ids) { return this->_impl->forward(ids); }
buffer<bf16> lfm3_npu::prefill(std::vector<int>& ids, void* payload) {
    try {
        return this->_impl->prefill(ids, payload);
    } catch (const std::runtime_error& e) {
        header_print("FLM", e.what());
        throw;
    }
}
void lfm3_npu::set_context_length(int L) { this->_impl->set_context_length(L); }
void lfm3_npu::load_weights(Q4NX& q4nx) { this->_impl->load_weights(q4nx); }
void lfm3_npu::update_max_length(uint32_t MAX_L) { this->_impl->update_max_length(MAX_L); }
void lfm3_npu::clear_context() { this->_impl->clear_context(); }
buffer<bf16> lfm3_npu::get_k_cache(int layer_idx, int idx) { return this->_impl->get_k_cache(layer_idx, idx); }
buffer<bf16> lfm3_npu::get_v_cache(int layer_idx, int idx) { return this->_impl->get_v_cache(layer_idx, idx); }
int lfm3_npu::get_current_context_length() { return this->_impl->get_current_context_length(); }
int lfm3_npu::checkpoint() { return this->_impl->checkpoint(); }
int lfm3_npu::restore() { return this->_impl->restore(); }
std::unique_ptr<hybrid_engine_state> lfm3_npu::new_state() { return this->_impl->new_state(); }
int lfm3_npu::read_state(hybrid_engine_state& state, int kv_from) { return this->_impl->read_state(state, kv_from); }
int lfm3_npu::write_state(const hybrid_engine_state& state) { return this->_impl->write_state(state); }
int lfm3_npu::write_state_kv(const hybrid_engine_state& state) { return this->_impl->write_state_kv(state); }
extern "C" const flm_state_api* lfm3_npu_state_api(void* engine) { return hybrid_state_api<lfm3_npu>(engine); }
lfm3_npu::~lfm3_npu() { delete this->_impl; }
