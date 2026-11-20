// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file speculate/dflash2_draft.hpp
/// \brief The z-lab DFlash2 block-diffusion drafter (Qwen3.8-27B-DFlash2) over the base engine's kernels.
///
/// What DFlash2 is (reference: z-lab/dflash `DFlash2DraftModel`, dflash/model.py):
///   * 5 Qwen3-style decoder layers (32 q heads / 8 kv heads x 128, MLP 17408), sliding-window attention (2048),
///     no causal mask inside the block.
///   * It conditions on the TARGET: the residual stream after target layers 5 19 33 47 61 for every committed token
///     (`tap_buffer`), concatenated to 5 x 5120, projected by `fc` (25600 -> 5120) and normed by `hidden_norm`.
///     Those "context features" give every layer a K/V row per committed token (k_proj/v_proj of the features
///     directly, no input norm), cached across cycles.
///   * One pass drafts a whole block: the input is the target embedding of [anchor, mask x 7]; the 8 rows attend to
///     the context rows and to each other. The 7 outputs go through the TARGET's lm_head.
///   * Two-tap dynamic convolutions: around attention and around the MLP of every layer the 8 block rows are mixed
///     with their predecessor, `out[t] = sum_{o<2} (base[o] + dyn[t][o][group]) * x[t-o]`, where `dyn` is a linear
///     projection (5120 -> 1280) of the layer's normed input and `base` a learned [2][2][5120] table.
///   * A candidate selector turns the 7 logit rows into one coherent path: top-16 candidates per position,
///     `score = logit + <pred_cb[prev] * W h, succ_cb[cand]>`, greedy argmax, `prev = chosen`.
///
/// What runs where (no new bitstream):
///   * fc, k|v of the context rows (all 5 layers in one dispatch), and per layer q|k|v, o, up|gate, down:
///     `dequant_mm.xclbin`, the same fused Q4_K dequant + GEMM the base model's prefill uses. A dispatch is a
///     fixed 512-row tile, so the 8-row block costs a full tile's compute: 22 dispatches per cycle.
///   * the draft's lm_head: `lm_head_8.xclbin` with the draft's own final-norm gain (the bitstream norms its
///     input itself, so it gets the pre-norm hidden and `norm.weight` instead of `model.norm.weight`).
///   * attention, the convolutions, norms, rope, the 1280-wide dynamic-kernel projections (1280 is not a multiple
///     of the 512-wide array tile) and the selector: host.
///
/// Weights: `dflash_weight.q4nx` in the draft package (FLM_Q4NX_Converter/convert_dflash2.py): Q4_K for the big
/// matrices, BF16 for everything else.
#pragma once

#include "common/cpu_ref_common.hpp"
#include "common/cpu_simd.hpp"
#include "common/embedding.hpp"
#include "common/host_par.hpp"
#include "common/npu/lm_head8_op.hpp"
#include "common/tap_buffer.hpp"
#include "speculate/npu/proj_op.hpp"
#include "verify/npu/q4nx.hpp"

#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace qwen38 {

/// The draft's geometry, from the draft package's config.json.
struct dflash2_config {
    int   layers      = 5;
    int   D           = 5120;
    int   heads       = 32;
    int   kv_heads    = 8;
    int   head_dim    = 128;
    int   inter       = 17408;
    int   block       = 8;      ///< rows per draft pass: 1 anchor + (block - 1) drafts
    int   conv_group  = 16;
    int   conv_kernel = 2;
    int   sel_rank    = 256;
    int   sel_topk    = 16;
    int   window      = 2048;
    int   mask_token  = 0;
    int   vocab       = 0;
    float eps         = 1e-6f;
    float rope_theta  = 1e7f;
    float emb_scale   = 1.0f;
    std::vector<int> taps;

    static dflash2_config from_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("dflash2: cannot open " + path);
        const nlohmann::json j = nlohmann::json::parse(f);
        const nlohmann::json& d = j.at("dflash_config");
        dflash2_config c;
        c.layers      = j.at("num_hidden_layers").get<int>();
        c.D           = j.at("hidden_size").get<int>();
        c.heads       = j.at("num_attention_heads").get<int>();
        c.kv_heads    = j.at("num_key_value_heads").get<int>();
        c.head_dim    = j.at("head_dim").get<int>();
        c.inter       = j.at("intermediate_size").get<int>();
        c.window      = j.value("sliding_window", 2048);
        c.vocab       = j.at("vocab_size").get<int>();
        c.eps         = j.value("rms_norm_eps", 1e-6f);
        c.block       = d.at("block_size").get<int>();
        c.conv_group  = d.at("conv_group_size").get<int>();
        c.conv_kernel = d.at("conv_kernel_size").get<int>();
        c.sel_rank    = d.at("selector_rank").get<int>();
        c.sel_topk    = d.at("selector_top_k").get<int>();
        c.mask_token  = d.at("mask_token_id").get<int>();
        c.taps        = d.at("target_layer_ids").get<std::vector<int>>();
        if (j.contains("rope_parameters"))
            c.rope_theta = j["rope_parameters"].value("rope_theta", 1e7f);
        c.emb_scale   = d.value("input_embedding_scale", 1.0f);
        return c;
    }
};

class dflash2_draft {
public:
    static constexpr int ROWS = 8;   ///< the block; also lm_head_8's row count and the verify batch

    /// \param w       the draft's dflash_weight.q4nx, mmap'd; must outlive this object
    /// \param embed   the TARGET's token embedding (the draft has none)
    /// \param head8   the TARGET's lm_head on lm_head_8.xclbin
    /// \param proj_xclbin  dequant_mm.xclbin
    dflash2_draft(const dflash2_config& cfg, const cpu_ref::safetensors& w,
                  const cpu_ref::embedding_table* embed, q38npu::npu_lm_head_8* head8,
                  npu_xclbin_manager* npu, const std::string& proj_xclbin, int target_D)
        : c_(cfg), w_(w), embed_(embed), head8_(head8) {
        if (c_.block != ROWS)
            throw std::runtime_error("dflash2: block_size " + std::to_string(c_.block) + " != " + std::to_string(ROWS));
        if (c_.D != target_D || c_.D != head8_->dim())
            throw std::runtime_error("dflash2: draft width " + std::to_string(c_.D) + " != target " + std::to_string(target_D));
        if (c_.vocab != head8_->vocab())
            throw std::runtime_error("dflash2: draft vocab " + std::to_string(c_.vocab) + " != target " + std::to_string(head8_->vocab()));
        if (std::fabs(c_.eps - cpu_ref::q38::RMS_EPS) > 1e-9f)
            throw std::runtime_error("dflash2: rms eps differs from the lm_head_8 bitstream's");
        if (c_.heads * c_.head_dim != DQ() || c_.conv_kernel != 2 || (int)c_.taps.size() < 1)
            throw std::runtime_error("dflash2: unsupported geometry");
        _load_host_weights();
        _build_ops(npu, proj_xclbin);

        cache_cap_ = c_.window + 1024 + 64;
        kc_.assign(c_.layers, std::vector<float>((size_t)cache_cap_ * DKV()));
        vc_.assign(c_.layers, std::vector<float>((size_t)cache_cap_ * DKV()));
        logits_.resize((size_t)ROWS * c_.vocab);
        const char* d = std::getenv("FLM_DFLASH_DUMP");
        if (d && d[0]) dump_dir_ = d;
        const char* p = std::getenv("FLM_DFLASH_PROF");
        prof_on_ = p && p[0] && p[0] != '0';
        inv_freq_.resize(c_.head_dim / 2);
        for (int i = 0; i < c_.head_dim / 2; i++)
            inv_freq_[i] = std::pow((double)c_.rope_theta, -2.0 * i / (double)c_.head_dim);
    }

    const dflash2_config& config() const { return c_; }
    int DQ()  const { return c_.heads * c_.head_dim; }
    int DKV() const { return c_.kv_heads * c_.head_dim; }

    // ------------------------------------------------------------------ ledger
    /// Forget everything: the next cycles draft from whatever rows arrive after this.
    void reset() {
        pend_n_ = 0; pend_.clear(); pend_idx_.clear(); pend_pos_.clear();
        c_off_ = c_n_ = 0; c_idx_.clear(); c_pos_.clear();
    }

    /// Drop every row of a token index >= n (the engine rewound its committed log to n tokens).
    void truncate(int n) {
        while (pend_n_ > 0 && pend_idx_[pend_n_ - 1] >= n) pend_n_--;
        pend_.resize((size_t)pend_n_ * row_elems());
        pend_idx_.resize(pend_n_); pend_pos_.resize(pend_n_);
        while (c_n_ > 0 && c_idx_[c_off_ + c_n_ - 1] >= n) c_n_--;
        c_idx_.resize(c_off_ + c_n_); c_pos_.resize(c_off_ + c_n_);
    }

    /// Highest token index held, absorbed or pending; -1 when empty.
    int last_idx() const {
        if (pend_n_ > 0) return pend_idx_[pend_n_ - 1];
        if (c_n_ > 0) return c_idx_[c_off_ + c_n_ - 1];
        return -1;
    }

    int rows_held() const { return pend_n_ + c_n_; }

    /// Wall time and row count of the context fold the last draft() started with.
    uint64_t last_absorb_us() const { return last_absorb_us_; }
    int      last_absorb_rows() const { return last_absorb_rows_; }

    /// Take the taps of window rows [first_row, n_commit) as committed tokens `base_idx + r`.
    /// \param pos_t  [L] the rope position of every window row (channel 0 of the window's position ids)
    /// A row that does not follow the last held one means some commit path forgot to report rows: start over cold
    /// rather than draft from a context with a hole in it.
    void append(const tap_buffer& tb, int base_idx, const int32_t* pos_t, int n_commit) {
        const int r0 = std::max(tb.first_row, 0);
        const int n  = n_commit - r0;
        if (n <= 0) return;
        const int idx0 = base_idx + r0;
        if (last_idx() >= 0 && last_idx() + 1 != idx0) {
            DEBUG_BLOCK(1, header_print("FLM", "dflash2: token " + std::to_string(idx0) + " does not follow " +
                                              std::to_string(last_idx()) + " -- context reset"););
            reset();
        }
        const size_t re = row_elems();
        const size_t at = (size_t)pend_n_;
        pend_.resize((at + (size_t)n) * re);
        for (int r = 0; r < n; r++) {
            std::memcpy(pend_.data() + (at + r) * re, tb.row(r0 - tb.first_row + r), re * sizeof(bf16));
            pend_idx_.push_back(idx0 + r);
            pend_pos_.push_back(pos_t[r0 + r]);
        }
        pend_n_ += n;
        // The sliding window reaches back `window` rows from the next anchor; older pending rows are never read.
        const int keep = c_.window + 64;
        if (pend_n_ > keep) {
            const int drop = pend_n_ - keep;
            pend_.erase(pend_.begin(), pend_.begin() + (size_t)drop * re);
            pend_idx_.erase(pend_idx_.begin(), pend_idx_.begin() + drop);
            pend_pos_.erase(pend_pos_.begin(), pend_pos_.begin() + drop);
            pend_n_ = keep;
        }
    }

    // ------------------------------------------------------------------ draft
    /// k <= 7 draft tokens after `anchor` (the committed token that has no K/V row yet).
    /// \param anchor_idx  its token index = the number of committed tokens
    /// \param anchor_pos  its rope position
    std::vector<int> draft(int anchor, int anchor_idx, int anchor_pos, int k) {
        using clk = std::chrono::steady_clock;
        auto now = [] { return clk::now(); };
        auto us = [](clk::time_point a, clk::time_point b) {
            return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        const auto t0 = now();
        last_absorb_rows_ = pend_n_;
        _absorb(anchor_idx);
        const auto t1 = now();
        last_absorb_us_ = us(t0, t1);

        const int D = c_.D, H = c_.heads, hd = c_.head_dim;
        h_.assign((size_t)ROWS * D, 0.0f);
        {
            for (int t = 0; t < ROWS; t++) {
                embed_->embed_row(t == 0 ? anchor : c_.mask_token, h_.data() + (size_t)t * D);
                if (c_.emb_scale != 1.0f)
                    for (int i = 0; i < D; i++) h_[(size_t)t * D + i] *= c_.emb_scale;
            }
        }
        if (!dump_dir_.empty()) {
            _dump("noise_emb", h_.data(), (size_t)ROWS * D);
            const int meta[4] = {anchor, anchor_idx, anchor_pos, k};
            _dump_i("meta", meta, 4);
        }

        // ---- cache window per query ---------------------------------------
        for (int t = 0; t < ROWS; t++) {
            const int qi = anchor_idx + t;
            int lo = c_off_, hi = c_off_ + c_n_;       // first row with idx > qi - window
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (c_idx_[mid] > qi - c_.window) hi = mid; else lo = mid + 1;
            }
            lb_[t] = lo - c_off_;
        }

        xin_.resize((size_t)ROWS * D); xc_.resize((size_t)ROWS * D); dyn_.resize((size_t)ROWS * dynw());
        q_.resize((size_t)ROWS * DQ()); kn_.resize((size_t)ROWS * DKV()); vn_.resize((size_t)ROWS * DKV());
        ao_.resize((size_t)ROWS * DQ()); o_.resize((size_t)ROWS * D);
        uint64_t npu_us = 0, host_us = 0;
        for (int l = 0; l < c_.layers; l++) {
            const layer_w& L = lw_[l];
            // ---- attention block --------------------------------------------
            auto a = now();
            for (int t = 0; t < ROWS; t++)
                cpu_ref::rms_norm(h_.data() + (size_t)t * D, L.ln_in.data(), D, xin_.data() + (size_t)t * D, c_.eps);
            _mm_bf16(L.kproj_attn, dynw(), D, xin_.data(), dyn_.data());
            _conv(xin_.data(), dyn_.data(), 0, L.base_attn.data(), xc_.data());
            auto b = now(); host_us += us(a, b);

            const bf16* qkv = _mm_npu(*qkv_op_[l], xc_.data(), D);
            auto c = now(); npu_us += us(b, c);
            const int QW = DQ() + 2 * DKV();
            for (int t = 0; t < ROWS; t++) {
                q38simd::bf16_to_f32(q_.data() + (size_t)t * DQ(), qkv + (size_t)t * QW, DQ());
                q38simd::bf16_to_f32(kn_.data() + (size_t)t * DKV(), qkv + (size_t)t * QW + DQ(), DKV());
                q38simd::bf16_to_f32(vn_.data() + (size_t)t * DKV(), qkv + (size_t)t * QW + DQ() + DKV(), DKV());
                const int pos = anchor_pos + t;
                for (int hh = 0; hh < H; hh++) {
                    float* qh = q_.data() + (size_t)t * DQ() + (size_t)hh * hd;
                    cpu_ref::rms_norm(qh, L.q_norm.data(), hd, qh, c_.eps);
                    _rope(qh, pos);
                }
                for (int hh = 0; hh < c_.kv_heads; hh++) {
                    float* kh = kn_.data() + (size_t)t * DKV() + (size_t)hh * hd;
                    cpu_ref::rms_norm(kh, L.k_norm.data(), hd, kh, c_.eps);
                    _rope(kh, pos);
                }
            }
            _attention(l);
            auto d = now(); host_us += us(c, d);

            const bf16* ob = _mm_npu(*o_op_[l], ao_.data(), DQ());
            auto e = now(); npu_us += us(d, e);
            for (int t = 0; t < ROWS; t++) q38simd::bf16_to_f32(o_.data() + (size_t)t * D, ob + (size_t)t * D, D);
            _conv(o_.data(), dyn_.data(), 1, L.base_attn.data(), xc_.data());   // finish: second half of dyn
            for (size_t i = 0; i < (size_t)ROWS * D; i++) h_[i] += xc_[i];

            // ---- MLP block ----------------------------------------------------
            auto f = now();
            for (int t = 0; t < ROWS; t++)
                cpu_ref::rms_norm(h_.data() + (size_t)t * D, L.ln_post.data(), D, xin_.data() + (size_t)t * D, c_.eps);
            _mm_bf16(L.kproj_mlp, dynw(), D, xin_.data(), dyn_.data());
            _conv(xin_.data(), dyn_.data(), 0, L.base_mlp.data(), xc_.data());
            auto g = now(); host_us += us(f, g);

            const bf16* ug = _mm_npu(*ug_op_[l], xc_.data(), D);
            {
                // up|gate come woven in 512-wide bands (even band = up, odd = silu'd gate, the kernel does the silu)
                bf16* gx = dn_op_[l]->stage(ROWS);
                const size_t DI = (size_t)c_.inter, W2 = 2 * DI;
                for (int t = 0; t < ROWS; t++)
                    for (size_t p = 0; p < DI / 512; p++)
                        q38simd::bf16_mul_to_bf16(gx + (size_t)t * DI + p * 512, ug + (size_t)t * W2 + (2 * p + 1) * 512,
                                                  ug + (size_t)t * W2 + (2 * p) * 512, 512);
            }
            const bf16* mb = dn_op_[l]->run_staged(ROWS, wbo_);
            auto hh_ = now(); npu_us += us(g, hh_);
            for (int t = 0; t < ROWS; t++) q38simd::bf16_to_f32(o_.data() + (size_t)t * D, mb + (size_t)t * D, D);
            _conv(o_.data(), dyn_.data(), 1, L.base_mlp.data(), xc_.data());
            for (size_t i = 0; i < (size_t)ROWS * D; i++) h_[i] += xc_[i];
            host_us += us(hh_, now());
            if (!dump_dir_.empty()) _dump(("h_l" + std::to_string(l)).c_str(), h_.data(), (size_t)ROWS * D);
        }
        const auto t2 = now();

        // ---- logits from the target's lm_head, with the draft's own final norm -------------------------------
        head8_->run(h_.data(), ROWS, norm_w_.data(), ROWS, logits_.data(), c_.eps);
        const auto t3 = now();

        // ---- selector -----------------------------------------------------------------------------------------
        std::vector<int> out = _select(anchor, k);
        const auto t4 = now();
        if (!dump_dir_.empty()) {
            _dump("h_pre", h_.data(), (size_t)ROWS * D);
            _dump_i("drafts", out.data(), out.size());
            dump_n_++;
        }
        if (prof_on_) {
            prof_[0] += us(t0, t1); prof_[1] += npu_us; prof_[2] += host_us; prof_[3] += us(t2, t3); prof_[4] += us(t3, t4);
            prof_n_++;
            if (prof_n_ % 20 == 0) {
                std::fprintf(stderr, "[dflash2] per cycle (%llu): absorb %.1f ms, layers NPU %.1f ms, layers host %.1f ms, lm_head %.1f ms, selector %.1f ms, ctx rows %d\n",
                             (unsigned long long)prof_n_, prof_[0] / 1e3 / prof_n_, prof_[1] / 1e3 / prof_n_, prof_[2] / 1e3 / prof_n_,
                             prof_[3] / 1e3 / prof_n_, prof_[4] / 1e3 / prof_n_, c_n_);
            }
        }
        (void)t2;
        return out;
    }

private:
    struct layer_w {
        std::vector<float> ln_in, ln_post, q_norm, k_norm;
        std::vector<float> base_attn, base_mlp;            // [2][2][D]
        const uint16_t* kproj_attn = nullptr;               // [1280][D] bf16
        const uint16_t* kproj_mlp  = nullptr;
    };

    size_t row_elems() const { return (size_t)c_.taps.size() * (size_t)c_.D; }
    int dynw() const { return 2 * c_.conv_kernel * (c_.D / c_.conv_group); }   // 1280

    // ------------------------------------------------------------------ weights
    void _load_host_weights() {
        auto get = [&](const std::string& n) { return w_.f32("model." + n); };
        hnorm_w_ = get("hidden_norm.weight");
        norm_w_  = get("norm.weight");
        lw_.resize(c_.layers);
        for (int l = 0; l < c_.layers; l++) {
            const std::string p = "layers." + std::to_string(l) + ".";
            layer_w& L = lw_[l];
            L.ln_in     = get(p + "input_layernorm.weight");
            L.ln_post   = get(p + "post_attention_layernorm.weight");
            L.q_norm    = get(p + "self_attn.q_norm.weight");
            L.k_norm    = get(p + "self_attn.k_norm.weight");
            L.base_attn = get(p + "attention_conv.base_kernel");
            L.base_mlp  = get(p + "mlp_conv.base_kernel");
            L.kproj_attn = reinterpret_cast<const uint16_t*>(w_.raw("model." + p + "attention_conv.kernel_projection.weight"));
            L.kproj_mlp  = reinterpret_cast<const uint16_t*>(w_.raw("model." + p + "mlp_conv.kernel_projection.weight"));
        }
        sel_proj_ = reinterpret_cast<const uint16_t*>(w_.raw("model.candidate_selector.hidden_projection.weight"));
        pred_cb_  = reinterpret_cast<const uint16_t*>(w_.raw("model.candidate_selector.predecessor_codebook"));
        succ_cb_  = reinterpret_cast<const uint16_t*>(w_.raw("model.candidate_selector.successor_codebook"));
    }

    void _build_ops(npu_xclbin_manager* npu, const std::string& proj_xclbin) {
        using namespace q38npu;
        npu_app_manager* mgr = npu->register_xclbin(proj_xclbin);
        const int D = c_.D, DI = c_.inter, T = (int)c_.taps.size();
        const int QW = DQ() + 2 * DKV();

        // one weight BO, every matrix reorder_cpy'd in at a recorded byte offset
        size_t total = 0;
        const size_t off_fc = total;  total += q4k_bytes((size_t)D * T * D);
        const size_t off_kv = total;  total += q4k_bytes((size_t)c_.layers * 2 * DKV() * D);
        std::vector<size_t> off_qkv(c_.layers), off_o(c_.layers), off_ug(c_.layers), off_dn(c_.layers);
        for (int l = 0; l < c_.layers; l++) {
            off_qkv[l] = total; total += q4k_bytes((size_t)QW * D);
            off_o[l]   = total; total += q4k_bytes((size_t)D * DQ());
            off_ug[l]  = total; total += q4k_bytes((size_t)2 * DI * D);
            off_dn[l]  = total; total += q4k_bytes((size_t)D * DI);
        }

        npu_app first = mgr->create_app();
        wbo_ = first.create_bo_buffer<u8>(total);
        uint8_t* base = wbo_.data();
        auto stage = [&](size_t off, const std::string& name, int N, int K) -> size_t {
            const size_t bytes = q4k_bytes((size_t)N * K);
            if (w_.nbytes("model." + name) != bytes)
                throw std::runtime_error("dflash2: " + name + " is " + std::to_string(w_.nbytes("model." + name)) +
                                         " B, expected a [" + std::to_string(N) + "][" + std::to_string(K) + "] q4_k");
            reorder_cpy(base + off, w_.raw("model." + name), bytes, K);
            return off + bytes;
        };
        stage(off_fc, "fc.weight", D, T * D);
        {
            size_t o = off_kv;
            for (int l = 0; l < c_.layers; l++) {
                const std::string p = "layers." + std::to_string(l) + ".self_attn.";
                o = stage(o, p + "k_proj.weight", DKV(), D);
                o = stage(o, p + "v_proj.weight", DKV(), D);
            }
        }
        for (int l = 0; l < c_.layers; l++) {
            const std::string p = "layers." + std::to_string(l) + ".";
            size_t o = off_qkv[l];
            o = stage(o, p + "self_attn.q_proj.weight", DQ(), D);
            o = stage(o, p + "self_attn.k_proj.weight", DKV(), D);
            o = stage(o, p + "self_attn.v_proj.weight", DKV(), D);
            stage(off_o[l], p + "self_attn.o_proj.weight", D, DQ());
            // up|gate woven in 512-row bands, as the base model's draft head stages them
            const std::string un = "model." + p + "mlp.up_proj.weight", gn = "model." + p + "mlp.gate_proj.weight";
            const size_t chunk = (size_t)512 / 32 * (D / 256) * W_BLOCK_BYTES;
            const size_t tot = q4k_bytes((size_t)DI * D);
            if (w_.nbytes(un) != tot || w_.nbytes(gn) != tot || tot % chunk != 0)
                throw std::runtime_error("dflash2: up/gate of layer " + std::to_string(l) + " are not [" + std::to_string(DI) + "][" + std::to_string(D) + "] q4_k");
            uint8_t* dst = base + off_ug[l];
            for (size_t i = 0; i < tot / chunk; i++) {
                reorder_cpy(dst, w_.raw(un) + i * chunk, chunk, D); dst += chunk;
                reorder_cpy(dst, w_.raw(gn) + i * chunk, chunk, D); dst += chunk;
            }
            stage(off_dn[l], p + "mlp.down_proj.weight", D, DI);
        }
        wbo_.sync_to_device();

        fc_ = std::make_unique<dequant_mm_op>(std::move(first), D, T * D, off_fc, dequant_mm_act_none);
        kv_ = std::make_unique<dequant_mm_op>(mgr->create_app(), c_.layers * 2 * DKV(), D, off_kv, dequant_mm_act_none);
        for (int l = 0; l < c_.layers; l++) {
            qkv_op_.push_back(std::make_unique<dequant_mm_op>(mgr->create_app(), QW, D, off_qkv[l], dequant_mm_act_none));
            o_op_.push_back(std::make_unique<dequant_mm_op>(mgr->create_app(), D, DQ(), off_o[l], dequant_mm_act_none));
            ug_op_.push_back(std::make_unique<dequant_mm_op>(
                mgr->create_app(), 2 * DI, D, off_ug[l],
                std::vector<dequant_mm_pass>{{0, 2, dequant_mm_act_none}, {1, 2, dequant_mm_act_silu}}));
            dn_op_.push_back(std::make_unique<dequant_mm_op>(mgr->create_app(), D, DI, off_dn[l], dequant_mm_act_none));
        }
    }

    // ------------------------------------------------------------------ small kernels
    /// 8 rows of fp32 x through one NPU op -> bf16 y [512][Dout] (rows >= 8 are padding)
    const bf16* _mm_npu(q38npu::dequant_mm_op& op, const float* x, int Din) {
        bf16* xd = op.stage(ROWS);
        for (int t = 0; t < ROWS; t++) q38simd::f32_to_bf16(xd + (size_t)t * Din, x + (size_t)t * Din, (size_t)Din);
        return op.run_staged(ROWS, wbo_);
    }

    /// y[8][N] = x[8][K] . W[N][K]^T, W bf16 row-major on the host
    static void _mm_bf16(const uint16_t* W, int N, int K, const float* x, float* y) {
        const bool par = (size_t)N * (size_t)K >= q38npu::host_par_min();
#pragma omp parallel for schedule(static) if (par)
        for (int n = 0; n < N; n++) {
            const uint16_t* wr = W + (size_t)n * K;
#if defined(__AVX512F__)
            __m512 acc[ROWS];
            for (int t = 0; t < ROWS; t++) acc[t] = _mm512_setzero_ps();
            for (int k = 0; k < K; k += 16) {
                const __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(wr + k))), 16));
                for (int t = 0; t < ROWS; t++)
                    acc[t] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(x + (size_t)t * K + k), acc[t]);
            }
            for (int t = 0; t < ROWS; t++) y[(size_t)t * N + n] = _mm512_reduce_add_ps(acc[t]);
#else
            for (int t = 0; t < ROWS; t++) {
                float s = 0;
                for (int k = 0; k < K; k++) s += cpu_ref::bf16_to_f32(wr[k]) * x[(size_t)t * K + k];
                y[(size_t)t * N + n] = s;
            }
#endif
        }
    }

    /// Two-tap grouped dynamic convolution over the 8 block rows.
    /// `which` picks the half of `dyn` ([row][2][2][groups]) and of `base` ([2][2][D]).
    void _conv(const float* x, const float* dyn, int which, const float* base, float* out) const {
        const int D = c_.D, G = D / c_.conv_group, K = c_.conv_kernel;
        for (int t = 0; t < ROWS; t++) {
            const float* dt = dyn + (size_t)t * dynw() + (size_t)which * K * G;
            for (int c = 0; c < D; c++) {
                const int g = c / c_.conv_group;
                float acc = 0.0f;
                for (int o = 0; o < K && o <= t; o++)
                    acc += (base[((size_t)which * K + o) * D + c] + dt[(size_t)o * G + g]) * x[(size_t)(t - o) * D + c];
                out[(size_t)t * D + c] = acc;
            }
        }
    }

    void _rope(float* v, int pos) const {
        const int h2 = c_.head_dim / 2;
        for (int i = 0; i < h2; i++) {
            const double ang = (double)pos * inv_freq_[i];
            const float c = (float)std::cos(ang), s = (float)std::sin(ang);
            const float a = v[i], b = v[i + h2];
            v[i]      = a * c - b * s;
            v[i + h2] = b * c + a * s;
        }
    }

    // ------------------------------------------------------------------ context
    /// Fold every pending row into the per-layer K/V cache.
    void _absorb(int anchor_idx) {
        if (pend_n_ == 0) return;
        const int D = c_.D, T = (int)c_.taps.size();
        const size_t re = row_elems();
        // rows no query of this block (or any later one) can see
        int first = 0;
        while (first < pend_n_ && pend_idx_[first] <= anchor_idx - c_.window) first++;
        const int n = pend_n_ - first;
        if (n > 0 && !dump_dir_.empty()) {
            char path[512];
            std::snprintf(path, sizeof path, "%s/c%04d_ctx.u16", dump_dir_.c_str(), dump_n_);
            if (FILE* f = std::fopen(path, "wb")) { std::fwrite(pend_.data() + (size_t)first * re, 2, (size_t)n * re, f); std::fclose(f); }
            _dump_i("ctx_idx", pend_idx_.data() + first, (size_t)n);
            _dump_i("ctx_pos", pend_pos_.data() + first, (size_t)n);
        }
        if (n > 0) {
            const int hd = c_.head_dim, KV = c_.kv_heads, W2 = 2 * DKV();
            _cache_make_room(n);
            std::vector<float> feat((size_t)512 * D);
            for (int t0 = 0; t0 < n; t0 += 512) {
                const int rows = std::min(512, n - t0);
                // fc: [rows][T*D] bf16 straight into the op's x BO
                bf16* xd = fc_->stage(rows);
                std::memcpy(xd, pend_.data() + (size_t)(first + t0) * re, (size_t)rows * re * sizeof(bf16));
                const bf16* y = fc_->run_staged(rows, wbo_);
                for (int r = 0; r < rows; r++) {
                    q38simd::bf16_to_f32(feat.data() + (size_t)r * D, y + (size_t)r * D, D);
                    cpu_ref::rms_norm(feat.data() + (size_t)r * D, hnorm_w_.data(), D, feat.data() + (size_t)r * D, c_.eps);
                }
                if (!dump_dir_.empty() && t0 == 0) _dump("ctx_feat", feat.data(), (size_t)std::min(rows, 64) * D);
                // k|v of all layers
                bf16* xk = kv_->stage(rows);
                for (int r = 0; r < rows; r++) q38simd::f32_to_bf16(xk + (size_t)r * D, feat.data() + (size_t)r * D, D);
                const bf16* kvy = kv_->run_staged(rows, wbo_);
                for (int r = 0; r < rows; r++) {
                    const int pos = pend_pos_[first + t0 + r];
                    const size_t row = (size_t)(c_off_ + c_n_ + r);
                    for (int l = 0; l < c_.layers; l++) {
                        float* kd = kc_[l].data() + row * DKV();
                        float* vd = vc_[l].data() + row * DKV();
                        q38simd::bf16_to_f32(kd, kvy + (size_t)r * c_.layers * W2 + (size_t)l * W2, DKV());
                        q38simd::bf16_to_f32(vd, kvy + (size_t)r * c_.layers * W2 + (size_t)l * W2 + DKV(), DKV());
                        for (int h = 0; h < KV; h++) {
                            float* kh = kd + (size_t)h * hd;
                            cpu_ref::rms_norm(kh, lw_[l].k_norm.data(), hd, kh, c_.eps);
                            _rope(kh, pos);
                        }
                    }
                }
                for (int r = 0; r < rows; r++) {
                    c_idx_.resize(c_off_ + c_n_ + r + 1);
                    c_pos_.resize(c_off_ + c_n_ + r + 1);
                    c_idx_[c_off_ + c_n_ + r] = pend_idx_[first + t0 + r];
                    c_pos_[c_off_ + c_n_ + r] = pend_pos_[first + t0 + r];
                }
                c_n_ += rows;
            }
        }
        (void)T;
        pend_n_ = 0; pend_.clear(); pend_idx_.clear(); pend_pos_.clear();
        // rows the next anchors cannot see any more
        while (c_n_ > 0 && c_idx_[c_off_] <= anchor_idx - c_.window) { c_off_++; c_n_--; }
    }

    void _cache_make_room(int n) {
        if (c_off_ + c_n_ + n <= cache_cap_) return;
        // slide the live rows to the front
        const size_t KVw = (size_t)DKV();
        for (int l = 0; l < c_.layers; l++) {
            std::memmove(kc_[l].data(), kc_[l].data() + (size_t)c_off_ * KVw, (size_t)c_n_ * KVw * sizeof(float));
            std::memmove(vc_[l].data(), vc_[l].data() + (size_t)c_off_ * KVw, (size_t)c_n_ * KVw * sizeof(float));
        }
        std::vector<int32_t> ni(c_idx_.begin() + c_off_, c_idx_.begin() + c_off_ + c_n_);
        std::vector<int32_t> np(c_pos_.begin() + c_off_, c_pos_.begin() + c_off_ + c_n_);
        c_idx_.swap(ni); c_pos_.swap(np);
        c_off_ = 0;
        if (c_n_ + n > cache_cap_)
            throw std::runtime_error("dflash2: cache overflow");
    }

    /// Block attention for layer l into ao_: 8 queries x 32 heads over [context rows in window | the 8 block rows].
    void _attention(int l) {
        const int hd = c_.head_dim, H = c_.heads, G = H / c_.kv_heads;
        const float scale = 1.0f / std::sqrt((float)hd);
        const size_t KVw = (size_t)DKV();
        const float* Kc = kc_[l].data() + (size_t)c_off_ * KVw;
        const float* Vc = vc_[l].data() + (size_t)c_off_ * KVw;
        const bool par = (size_t)H * (size_t)(c_n_ + ROWS) * hd * ROWS >= q38npu::host_par_min();
#pragma omp parallel for schedule(static) if (par)
        for (int h = 0; h < H; h++) {
            const int g = h / G;
            std::vector<float> s((size_t)c_n_ + ROWS);
            for (int t = 0; t < ROWS; t++) {
                const float* qv = q_.data() + (size_t)t * DQ() + (size_t)h * hd;
                const int lb = lb_[t];
                const int nr = c_n_ - lb;
                float mx = -1e30f;
                for (int r = 0; r < nr; r++) {
                    const float* kr = Kc + (size_t)(lb + r) * KVw + (size_t)g * hd;
                    float d = 0;
                    for (int i = 0; i < hd; i++) d += qv[i] * kr[i];
                    s[r] = d * scale; mx = std::max(mx, s[r]);
                }
                for (int j = 0; j < ROWS; j++) {
                    const float* kr = kn_.data() + (size_t)j * KVw + (size_t)g * hd;
                    float d = 0;
                    for (int i = 0; i < hd; i++) d += qv[i] * kr[i];
                    s[nr + j] = d * scale; mx = std::max(mx, s[nr + j]);
                }
                float sum = 0;
                for (int r = 0; r < nr + ROWS; r++) { s[r] = std::exp(s[r] - mx); sum += s[r]; }
                const float inv = 1.0f / sum;
                float* ov = ao_.data() + (size_t)t * DQ() + (size_t)h * hd;
                for (int i = 0; i < hd; i++) ov[i] = 0;
                for (int r = 0; r < nr; r++) {
                    const float* vr = Vc + (size_t)(lb + r) * KVw + (size_t)g * hd;
                    const float p = s[r] * inv;
                    for (int i = 0; i < hd; i++) ov[i] += p * vr[i];
                }
                for (int j = 0; j < ROWS; j++) {
                    const float* vr = vn_.data() + (size_t)j * KVw + (size_t)g * hd;
                    const float p = s[nr + j] * inv;
                    for (int i = 0; i < hd; i++) ov[i] += p * vr[i];
                }
            }
        }
    }

    // ------------------------------------------------------------------ selector
    std::vector<int> _select(int anchor, int k) {
        const int D = c_.D, R = c_.sel_rank, V = c_.vocab, TK = c_.sel_topk;
        // normed block rows 1..7 -> rank-256 projection
        std::vector<float> hn((size_t)ROWS * D, 0.0f), proj((size_t)ROWS * R);
        for (int t = 1; t < ROWS; t++)
            cpu_ref::rms_norm(h_.data() + (size_t)t * D, norm_w_.data(), D, hn.data() + (size_t)t * D, c_.eps);
        _mm_bf16(sel_proj_, R, D, hn.data(), proj.data());

        std::vector<int> out;
        int prev = anchor;
        std::vector<float> pw(R);
        for (int j = 1; j <= std::min(k, ROWS - 1); j++) {
            const float* lg = logits_.data() + (size_t)j * V;
            // top-k candidates, unsorted
            float cv[64]; int ci[64];
            int n = 0, mn = 0;
            for (int v = 0; v < V; v++) {
                const float x = lg[v];
                if (n < TK) {
                    cv[n] = x; ci[n] = v; n++;
                    if (n == TK) { mn = 0; for (int i = 1; i < TK; i++) if (cv[i] < cv[mn]) mn = i; }
                } else if (x > cv[mn]) {
                    cv[mn] = x; ci[mn] = v;
                    mn = 0; for (int i = 1; i < TK; i++) if (cv[i] < cv[mn]) mn = i;
                }
            }
            const uint16_t* pc = pred_cb_ + (size_t)prev * R;
            for (int r = 0; r < R; r++) pw[r] = cpu_ref::bf16_to_f32(pc[r]) * proj[(size_t)j * R + r];
            int best = 0; float bs = -1e30f;
            for (int i = 0; i < TK; i++) {
                const uint16_t* sc = succ_cb_ + (size_t)ci[i] * R;
                float d = 0;
                for (int r = 0; r < R; r++) d += pw[r] * cpu_ref::bf16_to_f32(sc[r]);
                const float sc_ = cv[i] + d;
                // ties to the lower token id, like torch.argmax over a sorted-by-id candidate list would not guarantee,
                // but the logits are bf16-rounded on the reference side so exact ties are real; pick lowest id
                if (sc_ > bs || (sc_ == bs && ci[i] < ci[best])) { bs = sc_; best = i; }
            }
            prev = ci[best];
            out.push_back(prev);
            if (!dump_dir_.empty()) {
                _dump_i(("cand" + std::to_string(j)).c_str(), ci, TK);
                _dump(("candv" + std::to_string(j)).c_str(), cv, TK);
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ debug dumps
    void _dump(const char* name, const float* p, size_t n) const {
        char path[512];
        std::snprintf(path, sizeof path, "%s/c%04d_%s.f32", dump_dir_.c_str(), dump_n_, name);
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(p, 4, n, f); std::fclose(f); }
    }
    void _dump_i(const char* name, const int* p, size_t n) const {
        char path[512];
        std::snprintf(path, sizeof path, "%s/c%04d_%s.i32", dump_dir_.c_str(), dump_n_, name);
        if (FILE* f = std::fopen(path, "wb")) { std::fwrite(p, 4, n, f); std::fclose(f); }
    }

    // ------------------------------------------------------------------ state
    dflash2_config c_;
    const cpu_ref::safetensors& w_;
    const cpu_ref::embedding_table* embed_;
    q38npu::npu_lm_head_8* head8_;

    std::vector<float> hnorm_w_, norm_w_;
    std::vector<layer_w> lw_;
    const uint16_t *sel_proj_ = nullptr, *pred_cb_ = nullptr, *succ_cb_ = nullptr;

    buffer<u8> wbo_;
    std::unique_ptr<q38npu::dequant_mm_op> fc_, kv_;
    std::vector<std::unique_ptr<q38npu::dequant_mm_op>> qkv_op_, o_op_, ug_op_, dn_op_;

    // context rows not yet folded into the K/V cache: [n][taps][D] bf16
    std::vector<bf16> pend_;
    std::vector<int32_t> pend_idx_, pend_pos_;
    int pend_n_ = 0;
    // the per-layer K/V cache of context rows (k normed + roped, fp32), live rows [c_off_, c_off_ + c_n_)
    std::vector<std::vector<float>> kc_, vc_;
    std::vector<int32_t> c_idx_, c_pos_;
    int c_off_ = 0, c_n_ = 0, cache_cap_ = 0;

    // scratch
    std::vector<float> h_, xin_, xc_, dyn_, q_, kn_, vn_, ao_, o_, logits_;
    int lb_[ROWS] = {};
    std::vector<double> inv_freq_;

    uint64_t last_absorb_us_ = 0;
    int last_absorb_rows_ = 0;
    std::string dump_dir_;
    int dump_n_ = 0;
    bool prof_on_ = false;
    uint64_t prof_[5] = {0, 0, 0, 0, 0};
    uint64_t prof_n_ = 0;
};

} // namespace qwen38
