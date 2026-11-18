// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#ifndef __TEST_VERIFY8_HPP__
#define __TEST_VERIFY8_HPP__
// S6.2: 8-row layer test rig.
//
//   oracle  = 8 sequential single-token dispatches of the decode xclbin (the R2 equivalence: "only the input is 8 tokens")
//   path 2  = `--mode oracle`: the same 8 sequential dispatches again (validates the harness: must be bit-identical)
//             `--mode verify`: verify_layer_hook() (implemented by the 8-row design, see below)
//
// Inputs: layer input row t = token embedding row of a deterministic token id (the engine's layer 0 input; for layers > 0 the
// embedding rows stand in for hidden states: realistic magnitude, deterministic, and identical for oracle and path 2, so the
// comparison is exact in what it measures). PLI vectors come from the token ids as in decoding_layer::operator().
// KV seeding: the first c rows come from running the decode xclbin c single-token steps. Skip layers (L35/L40) read the cache
// of their owner layer (22 for SWA, 23 for global); the owner runs in lock step and its cache BO is copied into the skip layer.
#include "decoding_layer.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <functional>
#include <vector>

struct v8_opts {
    std::string mode = "oracle";                  // oracle | verify
    std::vector<int> layers = {0, 5, 40, 35};
    std::vector<int> cursors = {0, 1, 7, 8, 9, 14, 15, 16, 30, 40, 56, 500, 505, 511, 512, 520, 1000};
    int repeat = 1;
    bool poison = false;
    int maxl = 1536;
    // The decode xclbin is NOT bit-reproducible across different predecessor dispatches (measured: the same input after a different
    // previous token differs by 1-2 bf16 ulp, cosine 0.99996), so the oracle-vs-oracle floor is the pass criterion, not exact equality.
    // The strict part is `repeat`: N identical 8-row runs from the same state must be bit-identical.
    float ulp_tol = 6.0f;                         // max-abs error allowed, in bf16 ulps of the row max
    float cos_tol = 0.9998f;
};

/// Implemented by the 8-row verify design (weak: the oracle-only build links the stub, which returns 2 = NOT IMPLEMENTED).
/// Contract: the layer's cache is in the state "cursor rows seeded" (ring layout for SWA exactly as the decode host leaves it).
/// Run ONE 8-row dispatch with inputs x_rows[0..7] (token ids ids[0..7], positions cursor..cursor+7), write the 8 output rows
/// (D bf16 each) to out_rows and leave the 8 appended K/V rows in layer.kv_cache (not for skip layers).
/// Return 0 on success, 2 = not implemented, anything else = failure.
using verify_hook_fn = std::function<int(decoding_layer&, int, const std::vector<buffer<bf16>>&, const std::vector<int>&, std::vector<buffer<bf16>>&)>;
inline verify_hook_fn g_verify_hook;   // set by host.cpp (--fused-xclbin installs the fused8 driver of host/test_fused8.hpp)
inline int verify_layer_hook(decoding_layer& layer, int cursor, const std::vector<buffer<bf16>>& x_rows, const std::vector<int>& ids,
                             std::vector<buffer<bf16>>& out_rows) {
    return g_verify_hook ? g_verify_hook(layer, cursor, x_rows, ids, out_rows) : 2;
}

namespace v8 {

struct row_metrics { double cos = 1, rel_l2 = 0, max_abs = 0, ulp = 0; size_t n_diff = 0; bool nan = false; };

inline float bf2f(bf16 v) { return (float)v; }

inline row_metrics compare_rows(const bf16* a, const bf16* ref, size_t n) {
    row_metrics m;
    double dot = 0, na = 0, nr = 0, err2 = 0, rmax = 0;
    for (size_t i = 0; i < n; i++) {
        float x = bf2f(a[i]), r = bf2f(ref[i]);
        if (std::isnan(x) || std::isinf(x)) { m.nan = true; }
        if (std::memcmp(&a[i], &ref[i], sizeof(bf16)) != 0) m.n_diff++;
        dot += (double)x * r; na += (double)x * x; nr += (double)r * r;
        double e = std::fabs((double)x - r); err2 += e * e; m.max_abs = std::max(m.max_abs, e);
        rmax = std::max(rmax, (double)std::fabs(r));
    }
    m.cos = (na > 0 && nr > 0) ? dot / (std::sqrt(na) * std::sqrt(nr)) : ((na == 0 && nr == 0) ? 1.0 : 0.0);
    m.rel_l2 = nr > 0 ? std::sqrt(err2 / nr) : 0;
    int e = rmax > 0 ? (int)std::floor(std::log2(rmax)) : 0;
    double ulp = std::ldexp(1.0, e - 7);           // bf16: 8 significant bits
    m.ulp = ulp > 0 ? m.max_abs / ulp : 0;
    return m;
}

struct agg { double min_cos = 1, max_ulp = 0, max_rel = 0; size_t n_diff = 0; bool nan = false;
    void add(const row_metrics& m) { min_cos = std::min(min_cos, m.cos); max_ulp = std::max(max_ulp, m.ulp);
        max_rel = std::max(max_rel, m.rel_l2); n_diff += m.n_diff; nan |= m.nan; } };

inline std::vector<buffer<bf16>> copy_rows_alloc(int n, int d) { std::vector<buffer<bf16>> v; for (int i = 0; i < n; i++) v.emplace_back(d); return v; }

inline std::vector<int> make_ids(int n) {
    std::vector<int> ids(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < n; i++) { s = s * 6364136223846793005ull + 1442695040888963407ull; ids[i] = 1000 + (int)((s >> 33) % 200000); }
    return ids;
}

inline void poison_rows(decoding_layer& L, int first_row) {
    // fill K/V rows >= first_row (positions; ring layout for SWA) with NaN on the device copy
    L.kv_cache.sync_from_device();
    const bf16 nanv = bf16(std::nanf(""));
    const int rows = L.kv_rows();
    const size_t half = L.kv_cache_size / 2, row = L.DK;
    // Kernel contract (notes/attn8_L8_results.md): the V mmul works on bfp16 blocks of 8 keys, so padding rows of V must be finite
    // and a large value must not share an 8-row block with a valid key; NaN in V is never tolerated (0 x NaN).  The engine guarantees
    // zero rows past the context.  So: K rows >= first_row -> NaN; V rows from the next 8-row boundary -> 1e4 (bf16 0x461C).
    const bf16 bigv = [] { uint16_t u = 0x461C; bf16 b; std::memcpy(&b, &u, 2); return b; }();
    const int v_from = (first_row + 7) / 8 * 8;
    if (std::getenv("V8_POISON_DEBUG")) {   // what the rows between the K and the V poison boundaries hold (must be finite / small)
        float vmax = 0; int nonfin = 0;
        for (int r = first_row; r < v_from && r < rows; r++)
            for (size_t j = 0; j < row; j++) { float v = (float)L.kv_cache.data()[half + (size_t)r * row + j]; if (!std::isfinite(v)) nonfin++; else vmax = std::max(vmax, std::fabs(v)); }
        std::printf("poison_rows: first_row %d v_from %d: V rows in between max |v| %g, non-finite %d\n", first_row, v_from, vmax, nonfin);
    }
    for (int r = first_row; r < rows; r++) {
        for (size_t j = 0; j < row; j++) { L.kv_cache.data()[(size_t)r * row + j] = nanv; if (r >= v_from) L.kv_cache.data()[half + (size_t)r * row + j] = bigv; }
    }
    L.kv_cache.sync_to_device();
}

} // namespace v8

inline int test_verify8(model_list::model_t model, npu_app_manager* mgr, LM_Config& cfg, Q4NX& q4nx, const v8_opts& o) {
    using namespace v8;
    int failures = 0;
    const int maxc = *std::max_element(o.cursors.begin(), o.cursors.end());
    const int nids = maxc + 8 + 8;
    std::vector<int> ids = make_ids(nids);
    std::vector<int> cursors = o.cursors; std::sort(cursors.begin(), cursors.end());

    header_print("info", "verify8 mode=" + o.mode + " layers=" + std::to_string(o.layers.size()) + " cursors=" + std::to_string(cursors.size()) +
                 " repeat=" + std::to_string(o.repeat) + " poison=" + std::to_string(o.poison) + " L_TOKENS=" + std::to_string(L_TOKENS));
    std::printf("\n%-5s %-7s %-4s | %-9s %-9s %-7s | %-9s %-7s | %-6s %-9s %s\n", "layer", "kind", "c", "out_cos", "out_ulp", "out_diff", "kv_cos", "kv_ulp", "rep", "us/disp", "verdict");

    for (int layer_idx : o.layers) {
        auto L = std::make_unique<decoding_layer>(model, cfg, mgr, layer_idx, o.maxl);
        std::unique_ptr<decoding_layer> owner;
        const bool skip = L->shall_skip_kv;
        if (skip) owner = std::make_unique<decoding_layer>(model, cfg, mgr, L->final_kv_cache_layer, o.maxl);
        L->load_weights(q4nx);
        if (owner) owner->load_weights(q4nx);
        const std::string kind = std::string(L->is_swa_layer ? "swa" : "glob") + (skip ? "-skip" : "");
        const int D = L->D;

        // inputs: row t = embedding of ids[t]
        std::vector<buffer<bf16>> xin;
        for (int t = 0; t < nids; t++) { buffer<bf16> r(D); buffer<bf16> e = L->embedding(ids[t]); memcpy(r.data(), e.data(), D * sizeof(bf16)); xin.push_back(std::move(r)); }

        L->set_context_length(0);
        if (owner) owner->set_context_length(0);
        int pos = 0;   // steps already run on the sequence layer (owner for skip layers)
        decoding_layer& seqL = skip ? *owner : *L;
        auto step = [&](decoding_layer& ly, int t) { buffer<bf16> out = ly(xin[t], ids[t]); buffer<bf16> r(D); memcpy(r.data(), out.data(), D * sizeof(bf16)); return r; };

        for (int c : cursors) {
            try {
                auto vprobe = [&](const char* tag) {
                    if (!std::getenv("V8_POISON_DEBUG") || skip) return;
                    L->kv_cache.sync_from_device();
                    const size_t half = L->kv_cache_size / 2, row = L->DK; float vmax = 0; int r_at = -1;
                    for (int r = 0; r < std::min(64, L->kv_rows()); r++) for (size_t j = 0; j < row; j++) { float v = std::fabs((float)L->kv_cache.data()[half + r * row + j]); if (v > vmax) { vmax = v; r_at = r; } }
                    std::printf("vprobe %s c=%d: max |V| rows 0..63 = %g at row %d\n", tag, c, vmax, r_at);
                };
                vprobe("before seed");
                while (pos < c) { step(seqL, pos); pos++; }
                vprobe("after seed");
                // state: seqL holds rows 0..c-1
                std::vector<uint8_t> snap;
                if (!skip) { L->kv_cache.sync_from_device(); snap.assign((uint8_t*)L->kv_cache.data(), (uint8_t*)L->kv_cache.data() + L->kv_cache.size() * sizeof(bf16)); }
                if (skip) {   // owner must hold rows 0..c+7 (written by the owner in the same chain); copy its cache into the skip layer
                    while (pos < c + 8) { step(*owner, pos); pos++; }
                    owner->kv_cache.sync_from_device();
                    memcpy(L->kv_cache.data(), owner->kv_cache.data(), std::min(L->kv_cache.size(), owner->kv_cache.size()) * sizeof(bf16));
                    L->kv_cache.sync_to_device();
                    L->set_context_length(c);
                }
                auto restore = [&]() {
                    if (!skip) { memcpy(L->kv_cache.data(), snap.data(), snap.size()); L->kv_cache.sync_to_device(); }
                    L->set_context_length(c);
                };
                // ---- oracle: 8 sequential dispatches (clean cache) ----
                auto ref_out = copy_rows_alloc(8, D); std::vector<buffer<bf16>> ref_k = copy_rows_alloc(8, L->DK), ref_v = copy_rows_alloc(8, L->DK);
                double us_sum = 0;
                for (int i = 0; i < 8; i++) {
                    auto r = step(*L, c + i); memcpy(ref_out[i].data(), r.data(), D * sizeof(bf16)); us_sum += L->last_dispatch_us;
                    if (!skip) { L->kv_cache.sync_from_device(); ref_k[i] = L->kv_row(0, c + i); ref_v[i] = L->kv_row(1, c + i); }
                }
                restore();
                // ---- path 2 (repeated) ----
                agg out_a, kv_a; bool rep_ok = true; double us2 = 0; int ndisp = 0;
                std::vector<buffer<bf16>> first_out;
                for (int rep = 0; rep < o.repeat; rep++) {
                    if (o.poison) { poison_rows(*L, std::min(L->kv_rows(), L->is_swa_layer ? std::min(c + 8, L->kv_rows()) : c + 8)); }
                    auto out = copy_rows_alloc(8, D); std::vector<buffer<bf16>> k2 = copy_rows_alloc(8, L->DK), v2 = copy_rows_alloc(8, L->DK);
                    if (o.mode == "oracle") {
                        for (int i = 0; i < 8; i++) {
                            auto r = step(*L, c + i); memcpy(out[i].data(), r.data(), D * sizeof(bf16)); us2 += L->last_dispatch_us; ndisp++;
                            if (!skip) { L->kv_cache.sync_from_device(); k2[i] = L->kv_row(0, c + i); v2[i] = L->kv_row(1, c + i); }
                        }
                    } else {
                        std::vector<buffer<bf16>> xr(xin.begin() + c, xin.begin() + c + 8); std::vector<int> id8(ids.begin() + c, ids.begin() + c + 8);
                        auto t0 = std::chrono::steady_clock::now();
                        int rc = verify_layer_hook(*L, c, xr, id8, out);
                        us2 += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); ndisp++;
                        if (rc == 2) { std::printf("verify_layer_hook: NOT IMPLEMENTED (build the verify design's hook)\n"); return 3; }
                        if (rc != 0) throw std::runtime_error("verify_layer_hook failed rc=" + std::to_string(rc));
                        if (!skip) { L->kv_cache.sync_from_device(); for (int i = 0; i < 8; i++) { k2[i] = L->kv_row(0, c + i); v2[i] = L->kv_row(1, c + i); } }
                    }
                    for (int i = 0; i < 8; i++) {
                        out_a.add(compare_rows(out[i].data(), ref_out[i].data(), D));
                        if (!skip) { kv_a.add(compare_rows(k2[i].data(), ref_k[i].data(), L->DK)); kv_a.add(compare_rows(v2[i].data(), ref_v[i].data(), L->DK)); }
                    }
                    vprobe("after verify");
                    if (rep == 0) first_out = std::move(out);
                    else for (int i = 0; i < 8; i++) if (std::memcmp(out[i].data(), first_out[i].data(), D * sizeof(bf16)) != 0) rep_ok = false;
                    restore();
                }
                const bool pass_out = !out_a.nan && out_a.min_cos >= o.cos_tol && out_a.max_ulp <= o.ulp_tol;
                const bool pass_kv = skip || (!kv_a.nan && kv_a.min_cos >= o.cos_tol && kv_a.max_ulp <= o.ulp_tol);
                const bool ok = pass_out && pass_kv && rep_ok;
                if (!ok) failures++;
                std::printf("%-5d %-7s %-4d | %-9.6f %-9.3f %-7zu | %-9.6f %-7.3f | %-6s %-9.0f %s\n", layer_idx, kind.c_str(), c, out_a.min_cos, out_a.max_ulp, out_a.n_diff,
                            skip ? 1.0 : kv_a.min_cos, skip ? 0.0 : kv_a.max_ulp, rep_ok ? "same" : "DIFF", ndisp ? us2 / ndisp / (o.mode == "oracle" ? 1 : 1) : 0.0,
                            ok ? (out_a.n_diff == 0 ? "PASS(exact)" : "PASS") : "FAIL");
                std::printf("      oracle us/dispatch (mean of 8): %.0f\n", us_sum / 8);
                std::fflush(stdout);
            } catch (const std::exception& e) {
                std::printf("%-5d %-7s %-4d | ABORT: %s\n", layer_idx, kind.c_str(), c, e.what());
                return 4;
            }
        }
        L.reset(); owner.reset();
    }
    std::printf("\nverify8 %s: %d failing cursor(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

#endif
