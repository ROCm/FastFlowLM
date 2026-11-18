// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// DFlash speculative decoding for gemma4e_dflash (draft / verify / commit; README.md).
//
// One cycle: dflash_draft (host draft on the draft context + 7 NPU lm_head dispatches) -> dflash_verify (n+1 ordinary
// forward() calls, one row each, logits and taps kept) -> the caller accepts k drafts -> dflash_commit(k+1).
// Rows of rejected tokens have already been written to the KV caches. Global layers need nothing (rows past the context
// length are masked); the sliding layers use linear window buffers (gemma4e_dflash_swa.hpp), where a rejected row never overwrites
// a position a later window needs. On commit the rejected rows of every layer are zeroed again so the caches look exactly like
// the ones of a plain decode.
#include "gemma4e_dflash_bo.hpp"
#include "gemma4e_dflash_spec_state.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <cmath>

void gemma4e_dflash::Impl::spec_create() {
    this->spec = new Spec();
    const char* e = std::getenv("FLM_DFLASH");
    this->spec->env_off = (e != nullptr && std::string(e) == "0");
    const char* v = std::getenv("FLM_DFLASH_VERIFY");
    this->spec->seq_forced = (v != nullptr && std::string(v) == "seq");
    const char* g = std::getenv("FLM_DFLASH_GUARD");
    this->spec->guard_off = (g != nullptr && std::string(g) == "0");
    if (const char* a = std::getenv("FLM_DFLASH_ASYNC_INJECT")) this->spec->inj_async = std::string(a) != "0";
    if (const char* d = std::getenv("FLM_DFLASH_DUMP")) {
        this->spec->dump_inj = std::fopen((std::string(d) + ".inj").c_str(), "wb");
        this->spec->dump_drf = std::fopen((std::string(d) + ".drf").c_str(), "wb");
    }
}

void gemma4e_dflash::Impl::spec_destroy() {
    if (this->spec != nullptr) {
        this->spec_join_inject();
        this->verify_destroy();
        if (this->spec->dump_inj) std::fclose(this->spec->dump_inj);
        if (this->spec->dump_drf) std::fclose(this->spec->dump_drf);
    }
    delete this->spec;
    this->spec = nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// hooks called from the plain engine paths
// ---------------------------------------------------------------------------------------------------------------------

bool gemma4e_dflash::Impl::spec_tap_mode() const { return this->spec != nullptr && this->spec->draft_loaded; }

int gemma4e_dflash::Impl::spec_tap_segment(int layer) const {
    int s = 0;
    for (int t = 0; t < kTaps; t++) if (layer > kTapLayers[t]) s = t + 1;
    return s;   // 0: layers 0..5, 1: 6..11, 2: 12..17, 3: 18..23, 4: 24..35, 5: 36..41
}

int gemma4e_dflash::Impl::spec_tap_index(int layer) const {
    for (int t = 0; t < kTaps; t++) if (layer == kTapLayers[t]) return t;
    return -1;
}

void gemma4e_dflash::Impl::spec_capture_tap(int t) {
    // x slot 0 holds the hidden state after layer kTapLayers[t]
    this->x.sync_from_device();
    std::memcpy(this->spec->tap_cur.data() + (size_t)t * D, this->x.data(), (size_t)D * sizeof(bf16));
}

void gemma4e_dflash::Impl::spec_join_inject() {
    Spec& S = *this->spec;
    if (!S.inj_thread.joinable()) return;
    auto t0 = clk::now();
    S.inj_thread.join();
    S.t_inj_wait += us_since(t0);
}

void gemma4e_dflash::Impl::spec_forward_begin(int id) {
    this->spec_join_inject();
    if (!this->spec->in_verify_seq) { this->spec->plain_since++; this->spec->fwd_t0 = clk::now(); }
    this->spec->fwd_pos = this->current_context_length;
    this->spec->ids.push_back(id);
}

void gemma4e_dflash::Impl::spec_forward_end() {
    Spec& S = *this->spec;
    if (!S.in_verify_seq) {          // a plain decode step: its cost feeds the guard
        const double ms = us_since(S.fwd_t0) / 1000.0;
        S.ema_t_plain = S.ema_t_plain > 0 ? 0.8 * S.ema_t_plain + 0.2 * ms : ms;
    }
    if (!S.draft_loaded) return;
    const long long p = S.fwd_pos;
    if (S.tap_hi != p) { S.tap_lo = S.tap_hi = p; }          // gap: the ring restarts here
    std::memcpy(S.tap_ring.data() + (size_t)(p % kRing) * kTaps * D, S.tap_cur.data(), (size_t)kTaps * D * sizeof(bf16));
    S.tap_hi = p + 1;
    if (S.tap_hi - S.tap_lo > kRing) S.tap_lo = S.tap_hi - kRing;
}

void gemma4e_dflash::Impl::spec_prefill_begin(const std::vector<int>& ids, int L) {
    this->spec_join_inject();
    Spec& S = *this->spec;
    S.ids.insert(S.ids.end(), ids.begin(), ids.begin() + L);
    S.prime_pending = true;
    S.rows_valid = 0;
    if (!S.draft_loaded) return;
    const long long pos0 = this->current_context_length;
    if (S.tap_hi != pos0) { S.tap_lo = S.tap_hi = pos0; }
}

void gemma4e_dflash::Impl::spec_prefill_tap(int layer, int L) {
    Spec& S = *this->spec;
    if (!S.draft_loaded) return;
    const int t = this->spec_tap_index(layer);
    if (t < 0) return;
    const long long pos0 = this->current_context_length;
    const int m = std::min(L, kRing);
    for (int i = L - m; i < L; i++) {
        const long long p = pos0 + i;
        std::memcpy(S.tap_ring.data() + ((size_t)(p % kRing) * kTaps + t) * D,
                    this->prefill_ctx->residual_row(i).data(), (size_t)D * sizeof(bf16));
    }
}

int gemma4e_dflash::Impl::spec_last_tap_layer() const { return kTapLayers[kTaps - 1]; }

bool gemma4e_dflash::Impl::spec_taps_wanted() const { return this->spec->draft_loaded; }

void gemma4e_dflash::Impl::spec_prefill_end(int L) {
    Spec& S = *this->spec;
    if (!S.draft_loaded) return;
    const long long pos0 = this->current_context_length;      // not advanced yet
    const long long first = pos0 + L - std::min(L, kRing);
    if (first > S.tap_hi) S.tap_lo = first;                    // the prompt was longer than the ring: older rows are gone
    S.tap_hi = pos0 + L;
    if (S.tap_hi - S.tap_lo > kRing) S.tap_lo = S.tap_hi - kRing;
}

// the context was cleared / rewound / replaced: ids may be unknown, taps and the draft context are gone (cold draft)
void gemma4e_dflash::Impl::spec_cold(int length) {
    this->spec_join_inject();
    Spec& S = *this->spec;
    S.ids.resize((size_t)std::max(length, 0), 0);
    S.ema_al = 2.6;
    S.plain_since = 0;
    S.tap_lo = S.tap_hi = std::max(length, 0);
    S.rows_valid = 0;
    S.prime_pending = true;
    if (S.draft_loaded) S.draft.reset();
}

// ---------------------------------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------------------------------

bool gemma4e_dflash::Impl::load_draft(const std::string& draft_dir) {
    this->spec_join_inject();
    Spec& S = *this->spec;
    if (S.env_off) {
        header_print("FLM", "gemma4e_dflash: FLM_DFLASH=0, speculation disabled");
        return false;
    }
    if (!S.draft.load(draft_dir)) {
        header_print("FLM", "gemma4e_dflash: draft package unusable, plain decoding");
        return false;
    }
    const auto& c = S.draft.cfg();
    if (c.D != D || c.n_taps != kTaps) {
        header_print("FLM", "gemma4e_dflash: draft does not match the target (hidden size / taps), plain decoding");
        return false;
    }
    S.tap_ring.assign((size_t)kRing * kTaps * D, bf16(0.0f));
    S.tap_cur.assign((size_t)kTaps * D, bf16(0.0f));
    S.logit_rows.assign((size_t)kBlock * this->vocab_size, bf16(0.0f));
    S.swa_layers.clear();
    for (int i = 0; i < non_skip_layers; i++) if (!attn_mask[i]) S.swa_layers.push_back(i);
    S.target_norm.assign((size_t)D, bf16(0.0f));
    std::memcpy(S.target_norm.data(), this->x.data() + D, (size_t)D * sizeof(bf16));   // loaded by load_weights()
    S.tap_lo = S.tap_hi = this->current_context_length;
    S.draft.reset();
    S.draft_loaded = true;
    // the segmented runlist (taps) replaces the single one from now on
    if (!this->npu->is_preemption_enabled()) {
        this->seg_runs.clear();
        for (int s = 0; s < kTaps + 1; s++) this->seg_runs.push_back(layer_app_manager->create_runlist());
    }
    this->set_context_length(this->current_context_length);
    header_print("FLM", "gemma4e_dflash: draft loaded from " + draft_dir);
    if (this->vocab_size == this->vocab_size_padded) this->verify_load();     // fused 8-row verify + lm_head_8 when the xclbins are there
    if (!this->verify_fused_ready() && !S.seq_forced) {
        // Without the fused verify a cycle is 8 sequential decode passes: about 3x slower than plain decoding at any acceptance
        // length, so do not speculate (preemption, missing verify xclbins). FLM_DFLASH_VERIFY=seq forces it for debugging.
        S.env_off = true;
        header_print("FLM", "gemma4e_dflash: fused 8-row verify not available, decoding without speculation (FLM_DFLASH_VERIFY=seq forces the sequential verify)");
    }
    return true;
}

bool gemma4e_dflash::Impl::draft_ready() const {
    const Spec& S = *this->spec;
    if (!(S.draft_loaded && !S.env_off && this->current_context_length > 0 &&
          (int)S.ids.size() == this->current_context_length &&
          this->current_context_length + kBlock <= this->MAX_L)) return false;
    if (S.guard_off || !this->verify_fused_ready()) return true;
    // Adaptive guard. A verify cycle costs about 114 + 0.0024*ctx ms, a plain step about 75 + 0.0022*ctx ms (both fitted on the Bolt E4B
    // measurements of FLM_Xclbin/Gemma4/verify_decoding/notes/e2e_results.md, verify with K/V read once per verify), so a cycle that yields AL
    // tokens only pays off when AL > t_cycle / t_plain: about 1.5 at 1k to 1.4 at 16k. The running mean of AL (reset per request, optimistic start)
    // decides; every 48 plain steps one cycle is let through to re-measure.
    // Costs: measured running means of this session when available (cycle = draft start .. commit end, plain = one forward()), the
    // fitted model otherwise. Measuring keeps the guard right when the verify gets faster (or slower) than the fit.
    const double ctx = this->current_context_length;
    const double t_plain = S.ema_t_plain > 0 ? S.ema_t_plain : 75.0 + 0.0022 * ctx;
    const double t_cycle = S.ema_t_cycle > 0 ? S.ema_t_cycle : 114.0 + 0.0024 * ctx;
    return S.ema_al * t_plain >= 0.95 * t_cycle || S.plain_since >= 48;     // 5% below break-even tolerated: n_keep per cycle is very noisy (1..8)
}

bool gemma4e_dflash::Impl::supports_speculation() const { return this->spec->draft_loaded && !this->spec->env_off; }

void gemma4e_dflash::Impl::set_stop_tokens(const std::vector<int>& ids) { this->spec->stop_ids = ids; }

int gemma4e_dflash::Impl::spec_vocab_size() const { return this->vocab_size; }

// ---------------------------------------------------------------------------------------------------------------------
// draft
// ---------------------------------------------------------------------------------------------------------------------

// absorb the taps of positions [draft.ctx_end(), ctx) into the draft context
void gemma4e_dflash::Impl::spec_sync_draft_ctx(int ctx) {
    this->spec_join_inject();
    Spec& S = *this->spec;
    long long start = S.draft.ctx_end();
    if (start < S.tap_lo || start > ctx) {          // never seen, fell out of the ring, or inconsistent: cold start
        S.draft.reset();
        start = S.tap_lo;
    }
    while (start < ctx) {
        const long long slot = start % kRing;
        static const int chunk = [] { const char* e = std::getenv("FLM_DFLASH_INJECT_CHUNK"); return e ? std::max(1, std::atoi(e)) : kRing; }();   // debug knob
        const int n = (int)std::min<long long>(std::min<long long>(ctx - start, kRing - slot), chunk);
        S.draft.inject(S.tap_ring.data() + (size_t)slot * kTaps * D, n, (int)start);
        if (S.dump_inj) {
            const long long p0 = start; const int nn = n;
            std::fwrite(&p0, sizeof(p0), 1, S.dump_inj); std::fwrite(&nn, sizeof(nn), 1, S.dump_inj);
            std::fwrite(S.tap_ring.data() + (size_t)slot * kTaps * D, sizeof(bf16), (size_t)n * kTaps * D, S.dump_inj);
            std::fflush(S.dump_inj);
        }
        start += n;
    }
}

int gemma4e_dflash::Impl::spec_lm_head_argmax(const bf16* hidden, const bf16* norm_w) {
    std::memcpy(this->x.data(), hidden, (size_t)D * sizeof(bf16));
    std::memcpy(this->x.data() + D, norm_w, (size_t)D * sizeof(bf16));
    this->x.sync_to_device();
    this->lm_head_run.start();
    flm::npu::wait_for_completion(this->lm_head_run, "lm_head_run");
    this->logits.sync_from_device();
    return argmax_bf16(this->logits_valid.data(), this->vocab_size);
}

int gemma4e_dflash::Impl::dflash_draft(int last_token, int* drafts) {
    Spec& S = *this->spec;
    if (!this->draft_ready()) return 0;
    if (this->is_preload_launched) { this->is_preload_launched = false; flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run"); }
    const int ctx = this->current_context_length;

    auto t0 = clk::now();
    S.cycle_t0 = t0; S.in_cycle = true;
    this->spec_sync_draft_ctx(ctx);
    if (S.prime_pending) {
        S.prime_us = (uint64_t)us_since(t0);
        S.t_prime += (long long)S.prime_us;
        S.prime_pending = false;
        t0 = clk::now();
    } else {
        S.prime_us = 0;
    }

    // block embeddings: row 0 = id_last, rows 1..7 = the mask token (the target's own, already x sqrt(D))
    // Verify preparation (compaction, instruction streams, the 42-run chain) runs on another thread. Started after the host draft by default,
    // so it overlaps the NPU draft-head dispatch and does not take a core from the 16 draft threads; FLM_DFLASH_PREP_EARLY=1: before the draft.
    static const bool prep_early = std::getenv("FLM_DFLASH_PREP_EARLY") != nullptr;
    if (prep_early && this->verify_fused_ready()) this->verify_prepare_async(ctx);
    std::vector<bf16> emb((size_t)kBlock * D);
    {
        buffer<bf16> r = this->embedding->forward(last_token);
        std::memcpy(emb.data(), r.data(), (size_t)D * sizeof(bf16));
        buffer<bf16> m = this->embedding->forward(S.draft.cfg().mask_id);
        for (int i = 1; i < kBlock; i++) std::memcpy(emb.data() + (size_t)i * D, m.data(), (size_t)D * sizeof(bf16));
    }
    std::vector<bf16> hidden((size_t)(kBlock - 1) * D);
    S.draft.draft(ctx, emb.data(), hidden.data());
    S.t_draft += us_since(t0);
    if (!prep_early && this->verify_fused_ready()) this->verify_prepare_async(ctx);

    t0 = clk::now();
    static const int rescore_k = [] { const char* e = std::getenv("FLM_DFLASH_RESCORE_K"); return e ? std::max(0, std::min(16, std::atoi(e))) : 8; }();
    std::vector<int> topk((size_t)(kBlock - 1) * std::max(rescore_k, 1));
    if (this->verify_draft_head(hidden.data(), S.draft.output_norm(), drafts, rescore_k ? topk.data() : nullptr, rescore_k)) {
        // The NPU head uses the Q4 lm_head; its argmax differs from the exact one on ~13% of the draft rows (near ties), which costs
        // ~2% acceptance. Re-score the NPU's top-K with the host Q8_0 embedding rows (tied weights) in fp32 and keep the best.
        if (rescore_k > 1) {
            const auto& c = S.draft.cfg();
            const bf16* onw = S.draft.output_norm();
            std::vector<float> out((size_t)D);
            for (int t = 0; t < kBlock - 1; t++) {
                const bf16* h = hidden.data() + (size_t)t * D;
                double ss = 0;
                for (int i = 0; i < (int)D; i++) { const float v = bf16_to_f32(h[i]); ss += (double)v * v; }
                const float sc = 1.0f / std::sqrt((float)(ss / D) + c.eps);
                for (int i = 0; i < (int)D; i++) out[i] = bf16_to_f32(h[i]) * sc * bf16_to_f32(onw[i]);
                float best = -1e30f; int bid = drafts[t];
                for (int j = 0; j < rescore_k; j++) {
                    const int id = topk[(size_t)t * rescore_k + j];
                    buffer<bf16> row = this->embedding->forward(id);
                    float acc = 0;
                    for (int i = 0; i < (int)D; i++) acc += bf16_to_f32(row[i]) * out[i];
                    if (acc > best) { best = acc; bid = id; }
                }
                drafts[t] = bid;
            }
        }
    } else {     // no lm_head_8: 7 single-row lm_head dispatches
        for (int i = 0; i < kBlock - 1; i++) {
            drafts[i] = this->spec_lm_head_argmax(hidden.data() + (size_t)i * D, S.draft.output_norm());
        }
        // put the target's final norm back (it reaches the device with the next x.sync_to_device())
        std::memcpy(this->x.data() + D, S.target_norm.data(), (size_t)D * sizeof(bf16));
    }
    S.t_draft_head += us_since(t0);
    if (S.dump_drf) {
        const int hdr[2] = {ctx, last_token};
        std::fwrite(hdr, sizeof(int), 2, S.dump_drf); std::fwrite(drafts, sizeof(int), kBlock - 1, S.dump_drf);
        std::fwrite(hidden.data(), sizeof(bf16), hidden.size(), S.dump_drf);
        std::fflush(S.dump_drf);
    }
    return kBlock - 1;
}

// ---------------------------------------------------------------------------------------------------------------------
// verify and commit
// ---------------------------------------------------------------------------------------------------------------------

const bf16* gemma4e_dflash::Impl::dflash_verify(int last_token, const int* drafts, int n_drafts) {
    this->spec_join_inject();
    const bf16* r = this->dflash_verify_impl(last_token, drafts, n_drafts);
    // the fused verify starts the injection itself between its taps and its lm_head_8 dispatch; the sequential path starts it here
    if (r != nullptr) this->spec_async_inject(this->spec->verify_ctx0, n_drafts + 1);
    return r;
}

// Overlap: inject the taps of every verified row on a background thread (it runs during the verify's lm_head_8 and the caller's
// sampling); dflash_commit() joins and truncates to the kept rows. No-op if already running or if the draft context is not exactly
// at ctx0 (a cold draft is rebuilt by spec_sync_draft_ctx instead).
void gemma4e_dflash::Impl::spec_async_inject(int ctx0, int rows) {
    Spec& S = *this->spec;
    if (!S.inj_async || S.inj_thread.joinable()) return;
    if (!(S.draft.ctx_end() == ctx0 && S.tap_hi >= (long long)ctx0 + rows && ctx0 >= S.tap_lo)) return;
    S.inj_thread = std::thread([this, ctx0, rows] {
        Spec& T = *this->spec;
        long long start = ctx0;
        while (start < (long long)ctx0 + rows) {
            const long long slot = start % kRing;
            const int n = (int)std::min<long long>((long long)ctx0 + rows - start, kRing - slot);
            T.draft.inject(T.tap_ring.data() + (size_t)slot * kTaps * D, n, (int)start);
            start += n;
        }
    });
}

const bf16* gemma4e_dflash::Impl::dflash_verify_impl(int last_token, const int* drafts, int n_drafts) {
    Spec& S = *this->spec;
    const int ctx0 = this->current_context_length;
    if (!S.draft_loaded || n_drafts < 0 || n_drafts > kBlock - 1 || ctx0 + n_drafts + 1 > this->MAX_L) return nullptr;
    if (!this->verify_fused_ready()) { S.verify_rows = n_drafts + 1; return this->verify_seq_impl(last_token, drafts, n_drafts); }
    S.verify_ctx0 = ctx0;
    S.verify_n = n_drafts;
    if (!S.verify_check) {
        auto t0 = clk::now();
        const bf16* r = this->verify_fused(last_token, drafts, n_drafts);
        if (r == nullptr) { S.verify_rows = n_drafts + 1; return this->verify_seq_impl(last_token, drafts, n_drafts); }
        S.rows_valid = n_drafts + 1;
        S.t_verify += us_since(t0);
        return r;
    }
    // check mode: sequential first (reference), roll back, then the fused pass; compare the rows that matter
    auto t0 = clk::now();
    const int V = this->vocab_size;
    const bf16* a = this->verify_seq_impl(last_token, drafts, n_drafts);
    std::vector<bf16> ref((size_t)(n_drafts + 1) * V);
    std::memcpy(ref.data(), a, ref.size() * sizeof(bf16));
    S.ids.resize((size_t)ctx0);
    S.tap_hi = std::min<long long>(S.tap_hi, ctx0);
    this->set_context_length(ctx0);
    const bf16* b = this->verify_fused(last_token, drafts, n_drafts);
    S.rows_valid = n_drafts + 1;
    S.t_verify += us_since(t0);
    if (b == nullptr) return nullptr;
    for (int i = 0; i <= n_drafts; i++) {
        double dot = 0, na = 0, nb = 0;
        for (int j = 0; j < V; j++) {
            const double x = bf16_to_f32(ref[(size_t)i * V + j]), y = bf16_to_f32(b[(size_t)i * V + j]);
            dot += x * y; na += x * x; nb += y * y;
        }
        const double cs = dot / (std::sqrt(na * nb) + 1e-30);
        const int ia = argmax_bf16(ref.data() + (size_t)i * V, V), ib = argmax_bf16(b + (size_t)i * V, V);
        S.chk_rows++;
        S.chk_cos_sum += cs;
        S.chk_min_cos = std::min(S.chk_min_cos, cs);
        if (ia != ib) {
            S.chk_argmax_diff++;
            // margin of the sequential logits between its argmax and the fused argmax; 2 bf16 ulps of the top logit = near tie
            const float la = bf16_to_f32(ref[(size_t)i * V + ia]), lb = bf16_to_f32(ref[(size_t)i * V + ib]);
            const float ulp = std::ldexp(1.0f, std::ilogb(std::max(std::fabs(la), 1e-6f)) - 7);
            if (la - lb > 2.0f * ulp) S.chk_argmax_diff_big++;
            if (std::getenv("FLM_DFLASH_TRACE")) {
                std::fprintf(stderr, "verify-check ctx %d row %d: seq argmax %d (%.3f) fused argmax %d (seq logit %.3f) margin %.3f cos %.5f\n", ctx0, i, ia, la, ib, lb, la - lb, cs);
            }
        }
    }
    return b;
}

const bf16* gemma4e_dflash::Impl::verify_seq_impl(int last_token, const int* drafts, int n_drafts) {
    Spec& S = *this->spec;
    const int ctx0 = this->current_context_length;
    if (!S.draft_loaded || n_drafts < 0 || n_drafts > kBlock - 1 || ctx0 + n_drafts + 1 > this->MAX_L) return nullptr;
    auto t0 = clk::now();
    if (this->is_preload_launched) { this->is_preload_launched = false; flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run"); }
    // Make room for the whole batch BEFORE it runs and never compact inside it: a commit may roll the context back to ctx0+1, and
    // the window of the next token then still needs positions down to ctx0-510, which a compaction at a later length would have
    // dropped. Compacting here keeps every row from floor16(ctx0 - 512) on, which covers any rollback (gemma4e_dflash_swa.hpp).
    if (swa_needs_compaction(ctx0 + n_drafts + 1, this->swa_base)) {
        this->swa_compact(ctx0);
        // the instruction sequences of the first verify row were generated for the OLD base (they bake the append/replay row offsets in)
        this->set_context_length(ctx0);
    }
    S.verify_ctx0 = ctx0;
    S.verify_n = n_drafts;
    S.verify_rows = n_drafts + 1;
    const int V = this->vocab_size;
    S.in_verify_seq = true;
    for (int i = 0; i <= n_drafts; i++) {
        const int tok = (i == 0) ? last_token : drafts[i - 1];
        buffer<bf16> lg = this->forward(tok);                       // also pushes the tap row of position ctx0+i
        std::memcpy(S.logit_rows.data() + (size_t)i * V, lg.data(), (size_t)this->vocab_size * sizeof(bf16));
    }
    S.in_verify_seq = false;
    S.rows_valid = n_drafts + 1;
    S.t_verify += us_since(t0);
    return S.logit_rows.data();
}

void gemma4e_dflash::Impl::dflash_commit(int n_keep) {
    this->spec_join_inject();
    Spec& S = *this->spec;
    auto t0 = clk::now();
    const int ctx0 = S.verify_ctx0;
    const int n = S.verify_n;
    if (S.rows_valid == 0 || n_keep < 1 || n_keep > n + 1) return;
    if (this->is_preload_launched) { this->is_preload_launched = false; flm::npu::wait_for_completion(this->pre_load_run, "pre_load_run"); }
    // rows ctx0+n_keep .. ctx0+n were rejected. With the linear sliding-window buffers (gemma4e_dflash_swa.hpp) they never overwrite
    // a position a later window needs (a row p sits at p - base, and the window of any later token starts at or after its own
    // position - 511), so nothing has to be saved or restored; what has to happen is that every row past the context becomes zero
    // again: the attention kernels read whole 16-row chunks, so stale finite rows would change the softmax max / rounding
    // relative to a plain decode even though they are masked
    const int written = std::max(S.verify_rows, n + 1);          // rows the verify wrote: n+1 (sequential) or 8 (fused, padding rows included)
    if (n_keep < written) {
        // zero the rows [ctx0+n_keep, ctx0+written) of every K/V buffer and flush just those byte ranges (a full-buffer sync of the 20 sliding
        // layers costs milliseconds per cycle); host writes of whole rows need no invalidate first
        auto zero_range = [&](buffer<bf16>& kv, size_t plane_off_elems, size_t row0, size_t row1, size_t dk) {
            std::memset(kv.data() + plane_off_elems + row0 * dk, 0, (row1 - row0) * dk * sizeof(bf16));
            gemma4e_dflash_bo::flush_range(kv, (plane_off_elems + row0 * dk) * sizeof(bf16), (row1 - row0) * dk * sizeof(bf16));
        };
        const size_t r0 = (size_t)ctx0 + n_keep, r1 = (size_t)ctx0 + written;
        for (size_t li = 0; li < S.swa_layers.size(); li++) {
            buffer<bf16>& kv = this->kv_caches[S.swa_layers[li]];
            const size_t b0 = r0 - (size_t)this->swa_base, b1 = std::min<size_t>(r1 - (size_t)this->swa_base, (size_t)kSwaBufferRows);
            if (b1 > b0) {
                zero_range(kv, 0, b0, b1, (size_t)SWA_DK);
                zero_range(kv, (size_t)kSwaBufferRows * SWA_DK, b0, b1, (size_t)SWA_DV);
            }
        }
        const kv_row_layout lay = this->kv_layout();
        for (int i = 0; i < non_skip_layers; i++) {
            if (!attn_mask[i]) continue;
            buffer<bf16>& kv = this->kv_caches[i];
            zero_range(kv, 0, r0, r1, (size_t)lay.DK);
            zero_range(kv, (size_t)lay.v_offset, r0, r1, (size_t)lay.DV);
        }
    }
    const int new_ctx = ctx0 + n_keep;
    if (this->verify_fused_ready()) {
        // the decode streams are only needed by a plain step: regenerate lazily in forward()
        this->current_context_length = new_ctx;
        this->dirty_rows = std::max(this->dirty_rows, (size_t)(ctx0 + written));
        this->decode_seq_stale = true;
    } else {
        this->set_context_length(new_ctx);
    }
    this->swa_check_zero("commit", new_ctx);
    S.ids.resize((size_t)new_ctx);
    S.tap_hi = std::min<long long>(S.tap_hi, new_ctx);
    S.draft.truncate(new_ctx);           // rows injected ahead for rejected drafts
    S.rows_valid = 0;
    // slow average while speculating (n_keep varies 1..8 from cycle to cycle), a heavy weight for the single cycle of a probe
    const double alpha = S.plain_since >= 48 ? 0.3 : 0.1;
    S.ema_al = (1.0 - alpha) * S.ema_al + alpha * n_keep;
    S.plain_since = 0;
    if (S.in_cycle) {
        const double ms = (us_since(S.cycle_t0) - (double)S.prime_us) / 1000.0;     // prompt priming is prefill work, not cycle cost
        S.ema_t_cycle = S.ema_t_cycle > 0 ? 0.8 * S.ema_t_cycle + 0.2 * ms : ms;
        S.in_cycle = false;
    }
    S.cycles++;
    S.tokens += n_keep;
    S.accepted += n_keep - 1;
    S.drafted += n;
    S.t_commit += us_since(t0);
}

const bf16* gemma4e_dflash::Impl::last_speculation_logits(int& rows, int& vocab) const {
    rows = this->spec->rows_valid;
    vocab = this->vocab_size;
    return this->spec->logit_rows.data();
}

uint64_t gemma4e_dflash::Impl::last_speculation_prime_us() const { return this->spec->prime_us; }

// ---------------------------------------------------------------------------------------------------------------------
// greedy convenience path (also what the engine-level tests drive)
// ---------------------------------------------------------------------------------------------------------------------

std::vector<int> gemma4e_dflash::Impl::speculate(int last_token, int max_draft) {
    Spec& S = *this->spec;
    if (!this->supports_speculation() || max_draft <= 0) return {};
    if (!this->draft_ready()) return {};
    const int k = std::min(max_draft, kBlock - 1);
    int d[kBlock];
    if (this->dflash_draft(last_token, d) != kBlock - 1) return {};
    const int V = this->vocab_size;
    const bf16* rows = this->dflash_verify(last_token, d, k);
    if (rows == nullptr) return {};

    auto t0 = clk::now();
    int am[kBlock];
    #pragma omp parallel for schedule(static, 1) num_threads(kBlock)
    for (int i = 0; i <= k; i++) am[i] = argmax_bf16(rows + (size_t)i * V, this->vocab_size);
    int acc = 0;
    while (acc < k && am[acc] == d[acc]) acc++;
    int commit = acc + 1;                                             // tokens am[0..acc]
    for (int j = 0; j < commit; j++) {
        if (std::find(S.stop_ids.begin(), S.stop_ids.end(), am[j]) != S.stop_ids.end()) { commit = j + 1; break; }
    }
    S.t_accept += us_since(t0);
    this->dflash_commit(commit);        // rows kept: id_last + (commit-1) drafts; the last returned token has no row yet
    return std::vector<int>(am, am + commit);
}

// ---------------------------------------------------------------------------------------------------------------------
// statistics
// ---------------------------------------------------------------------------------------------------------------------

double gemma4e_dflash::Impl::mean_accepted_length() const {
    return spec->cycles ? (double)spec->tokens / (double)spec->cycles : 0.0;
}
double gemma4e_dflash::Impl::draft_hit_rate() const {
    return spec->drafted ? (double)spec->accepted / (double)spec->drafted : 0.0;
}
int gemma4e_dflash::Impl::speculation_cycles() const { return (int)spec->cycles; }

std::string gemma4e_dflash::Impl::speculation_stats() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << "dflash: " << spec->cycles << " cycles, " << spec->tokens << " tokens, AL "
      << mean_accepted_length() << ", draft hit " << std::setprecision(1) << draft_hit_rate() * 100.0 << "%";
    return o.str();
}

std::string gemma4e_dflash::Impl::speculation_timing() const {
    const Spec& S = *spec;
    const double c = S.cycles ? (double)S.cycles : 1.0;
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << "dflash per cycle (ms): draft " << S.t_draft / c / 1000.0
      << ", draft lm_head " << S.t_draft_head / c / 1000.0 << ", verify " << S.t_verify / c / 1000.0 << ", accept "
      << S.t_accept / c / 1000.0 << ", commit " << S.t_commit / c / 1000.0 << " (inject wait " << S.t_inj_wait / c / 1000.0 << ")"
      << "; prime total " << S.t_prime / 1000.0 << " ms";
    if (S.fused_cycles > 0) {
        const double f = (double)S.fused_cycles;
        o << "\n  fused verify per cycle (ms): inputs " << S.t_v_prep / f / 1000.0 << ", sequences+runlist " << S.t_v_seq / f / 1000.0 << ", 42 layers "
          << S.t_v_run / f / 1000.0 << ", taps " << S.t_v_taps / f / 1000.0 << ", lm_head_8 " << S.t_v_lm / f / 1000.0
          << " (preparation work " << S.t_v_prep_work / f / 1000.0 << ", overlapped)";
    }
    if (S.chk_rows > 0) {
        o << "\n  verify check: " << S.chk_rows << " rows, argmax differs on " << S.chk_argmax_diff << " (" << S.chk_argmax_diff_big
          << " beyond 2 bf16 ulps), mean logits cosine " << std::setprecision(6) << S.chk_cos_sum / (double)S.chk_rows << ", min " << S.chk_min_cos;
    }
    return o.str();
}

void gemma4e_dflash::Impl::reset_speculation_stats() {
    Spec& S = *spec;
    S.cycles = S.tokens = S.accepted = S.drafted = 0;
    S.t_draft = S.t_draft_head = S.t_verify = S.t_accept = S.t_commit = S.t_prime = S.t_inj_wait = 0;
    S.t_v_prep = S.t_v_seq = S.t_v_run = S.t_v_taps = S.t_v_lm = S.fused_cycles = S.t_v_prep_work = 0;
    S.chk_rows = S.chk_argmax_diff = S.chk_argmax_diff_big = 0;
    S.chk_cos_sum = 0;
    S.chk_min_cos = 1.0;
}
