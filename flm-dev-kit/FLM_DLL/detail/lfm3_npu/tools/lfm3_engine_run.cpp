// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Host harness for the LFM3 engine: load a package (config.json + model.q4nx), prefill token ids, write the last-position
// logits and optionally run greedy decode. Needs no NPU: the engine runs its host path when no device is given.
//
//   lfm3_engine_run --pkg <dir> --ids-file <file> [--logits-out <file>] [--greedy N] [--max-length L] [--npu --xclbin-root <dir>]
//     ids-file: whitespace separated token ids (one prompt)
//     logits-out: float32 little endian, vocab values
//   prints: load_ms, prefill_ms, decode tok/s, and the greedy token ids on one line ("greedy: ...")
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "models/lfm3/lfm3_npu.hpp"
#include "npu_utils/npu_utils.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    std::string pkg, ids_file, logits_out;
    int greedy = 0, max_length = 4096;
    bool split_prefill = false, use_npu = false;
    std::string xroot, teacher_file, trace_out;
    int grow_to = 0;
    bool ckpt_test = false, state_test = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--pkg") pkg = next();
        else if (a == "--ids-file") ids_file = next();
        else if (a == "--logits-out") logits_out = next();
        else if (a == "--greedy") greedy = std::stoi(next());
        else if (a == "--max-length") max_length = std::stoi(next());
        else if (a == "--state-test") state_test = true;              // saved-state API: read, decode, write back, decode again; split restore after clear_context
        else if (a == "--ckpt-test") ckpt_test = true;                // checkpoint after the prefill, decode N, restore, decode N again: must match
        else if (a == "--grow") grow_to = std::stoi(next());          // after the prefill, update_max_length(N) (cache re-layout) before decoding
        else if (a == "--teacher") teacher_file = next();             // after the prompt, feed these token ids through decode (forward) and ...
        else if (a == "--logits-trace") trace_out = next();           // ... write the logits of every step (float32, V per step) here
        else if (a == "--npu") use_npu = true;                     // run the NPU prefill stages (LFM3_PREFILL selects them)
        else if (a == "--xclbin-root") xroot = next();            // directory that holds xclbins/<package name>/
        else if (a == "--split-prefill") split_prefill = true;   // prefill in two calls (prefix-cache style) to test continuation
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pkg.empty() || ids_file.empty()) { std::fprintf(stderr, "usage: %s --pkg DIR --ids-file F [--logits-out F] [--greedy N] [--max-length L]\n", argv[0]); return 2; }

    std::vector<int> ids;
    { std::ifstream f(ids_file); int v; while (f >> v) ids.push_back(v); }
    if (ids.empty()) { std::fprintf(stderr, "no ids in %s\n", ids_file.c_str()); return 2; }

    LM_Config cfg;
    cfg.model_path = pkg;
    cfg.model_name = std::filesystem::path(pkg).filename().string();
    cfg.exec_path = xroot;
    { std::ifstream cf(std::filesystem::path(pkg) / "config.json"); cfg._json_config = nlohmann::json::parse(cf); }
    cfg._json_config["is_vlm"] = false;
    cfg._json_config["is_audio"] = false;

    auto t = std::chrono::steady_clock::now();
    Q4NX q4nx(pkg);
    std::unique_ptr<flm_rt::device> device;
    std::unique_ptr<npu_xclbin_manager> npu;
    if (use_npu) {
        device = std::make_unique<flm_rt::device>(0);
        npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, device.get(), false);
    }
    lfm3_npu engine(cfg, npu.get(), max_length);
    engine.load_weights(q4nx);
    std::fprintf(stderr, "load_ms,%.0f\n", ms_since(t));

    t = std::chrono::steady_clock::now();
    buffer<bf16> logits;
    if (split_prefill && ids.size() > 1) {
        const size_t cut = ids.size() / 2;
        std::vector<int> a(ids.begin(), ids.begin() + cut), b(ids.begin() + cut, ids.end());
        engine.prefill(a);
        logits = engine.prefill(b);
    } else {
        logits = engine.prefill(ids);
    }
    std::fprintf(stderr, "prefill_ms,%.0f,tokens,%zu\n", ms_since(t), ids.size());

    if (grow_to > 0) engine.update_max_length(grow_to);
    const size_t V = logits.size();
    if (!logits_out.empty()) {
        std::vector<float> f(V);
        for (size_t i = 0; i < V; i++) f[i] = (float)logits[i];
        std::ofstream fo(logits_out, std::ios::binary);
        fo.write(reinterpret_cast<const char*>(f.data()), V * sizeof(float));
    }
    auto argmax = [&](buffer<bf16>& l) {
        size_t best = 0;
        float bv = (float)l[0];
        for (size_t i = 1; i < l.size(); i++) { float v = (float)l[i]; if (v > bv) { bv = v; best = i; } }
        return (int)best;
    };
    if (!teacher_file.empty()) {
        // teacher forcing: the logits after each forced token, for comparing two engines step by step without greedy divergence
        std::vector<int> forced;
        { std::ifstream f(teacher_file); int v; while (f >> v) forced.push_back(v); }
        std::ofstream fo(trace_out, std::ios::binary);
        t = std::chrono::steady_clock::now();
        for (int tk : forced) {
            buffer<bf16> l = engine.forward(tk);
            std::vector<float> f(V);
            for (size_t i = 0; i < V; i++) f[i] = (float)l[i];
            fo.write(reinterpret_cast<const char*>(f.data()), V * sizeof(float));
        }
        std::fprintf(stderr, "teacher_ms,%.0f,tokens,%zu\n", ms_since(t), forced.size());
        return 0;
    }
    if (state_test) {
        // Saved-state API (flm_state_api.h), engine level. Each gate prints one PASS/FAIL line; exit status 0 only if all pass.
        auto dec = [&](int first, int n) { std::vector<int> o; int tk = first; for (int i = 0; i < n; i++) { o.push_back(tk); buffer<bf16> l = engine.forward(tk); tk = argmax(l); } return o; };
        int failed = 0;
        auto gate = [&](const char* name, bool ok, const std::string& detail = "") { std::printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str()); if (!ok) failed++; };
        auto cosine = [&](buffer<bf16>& a, buffer<bf16>& b) { double ab = 0, aa = 0, bb = 0; for (size_t i = 0; i < a.size(); i++) { double x = (float)a[i], y = (float)b[i]; ab += x * y; aa += x * x; bb += y * y; } return ab / (std::sqrt(aa * bb) + 1e-30); };
        const int first = argmax(logits);
        const int len = engine.get_current_context_length();
        auto st = engine.new_state();
        if (!st) { std::fprintf(stderr, "no state support (needs a device)\n"); return 2; }
        gate("S0 read_state returns the context length", engine.read_state(*st, 0) == len);
        std::vector<int> base = dec(first, greedy);                       // baseline: decode from the full prefill

        // S1 round trip: back to the read point (rows decoded since must not leak: rows past the length are zeroed)
        gate("S1 write_state returns the read length", engine.write_state(*st) == len);
        gate("S1 decode after write_state is identical", dec(first, greedy) == base);

        // S2 split restore: the shared head (write_state_kv) under a state that keeps only its tail (kv_from)
        const int cut = len / 2;
        auto tail = engine.new_state();
        engine.write_state(*st);
        engine.read_state(*tail, cut);
        gate("S2 tail state records kv_from", tail->kv_from == cut && tail->context_length == len);
        engine.clear_context();
        std::vector<int> head_ids(ids.begin(), ids.begin() + cut);
        engine.prefill(head_ids);
        auto head = engine.new_state();
        engine.read_state(*head, 0);
        engine.clear_context();
        const int w1 = engine.write_state_kv(*head);
        const int w2 = engine.write_state(*tail);
        gate("S2 write_state_kv / write_state lengths", w1 == cut && w2 == len);
        gate("S2 decode after split restore is identical", dec(first, greedy) == base);

        // S3 state read in the middle of decoding (after NPU steps: conv ring and KV rows live on the device)
        engine.write_state(*st);
        dec(first, 6);                                                    // 6 steps: base[0..5] were fed, base[6] is next
        auto mid = engine.new_state();
        const int mid_len = engine.read_state(*mid, 0);
        gate("S3 mid-decode state length", mid_len == len + 6 && base.size() >= 16);
        engine.write_state(*mid);
        std::vector<int> cont_a = dec(base[6], 10);
        engine.write_state(*mid);
        std::vector<int> cont_b = dec(base[6], 10);
        gate("S3 resume from a mid-decode state is repeatable", cont_a == cont_b);
        gate("S3 resume from a mid-decode state continues the baseline", cont_a == std::vector<int>(base.begin() + 6, base.begin() + 16));

        // S4 a state is independent of MAX_L: read it, grow the cache, write it back
        engine.write_state(*st);
        engine.update_max_length(max_length * 2);
        gate("S4 write_state after update_max_length", engine.write_state(*st) == len);
        gate("S4 decode after growth is identical", dec(first, greedy) == base);

        // S5 resume numerics: cold prefill vs (state of the first half + prefill of the rest) agree like two chunked prefills do
        engine.clear_context();
        buffer<bf16> cold = engine.prefill(ids);
        engine.clear_context();
        std::vector<int> a_ids(ids.begin(), ids.begin() + cut), b_ids(ids.begin() + cut, ids.end());
        engine.prefill(a_ids);
        auto half = engine.new_state();
        engine.read_state(*half, 0);
        engine.clear_context();
        engine.write_state(*half);
        buffer<bf16> warm = engine.prefill(b_ids);
        const double c = cosine(cold, warm);
        gate("S5 warm resume logits vs cold prefill (cos >= 0.998, same top-1)", c >= 0.998 && argmax(cold) == argmax(warm), "cos=" + std::to_string(c));

        // S6 an empty / foreign state is rejected without touching the engine
        auto empty = engine.new_state();
        gate("S6 write_state of an unfilled state is rejected", engine.write_state(*empty) == -1);
        std::printf("%s\n", failed ? "STATE_TESTS_FAILED" : "STATE_TESTS_OK");
        return failed ? 1 : 0;
    }
    if (ckpt_test) {
        const int first = argmax(logits);
        const int at = engine.checkpoint();
        std::vector<int> runs[2];
        for (int r = 0; r < 2; r++) {
            int tk = first;
            for (int i = 0; i < greedy; i++) { runs[r].push_back(tk); buffer<bf16> l = engine.forward(tk); tk = argmax(l); }
            if (r == 0) { const int back = engine.restore(); std::fprintf(stderr, "checkpoint at %d, restored to %d\n", at, back); }
        }
        std::printf("run1:"); for (int v : runs[0]) std::printf(" %d", v); std::printf("\nrun2:"); for (int v : runs[1]) std::printf(" %d", v);
        std::printf("\n%s\n", runs[0] == runs[1] ? "CKPT_OK" : "CKPT_MISMATCH");
        return runs[0] == runs[1] ? 0 : 1;
    }
    std::vector<int> out;
    t = std::chrono::steady_clock::now();
    int tok = argmax(logits);
    for (int i = 0; i < greedy; i++) {
        out.push_back(tok);
        if (tok == 124900) break;               // <|im_end|>
        logits = engine.forward(tok);
        tok = argmax(logits);
    }
    if (greedy > 0) {
        const double ms = ms_since(t);
        std::fprintf(stderr, "decode_ms,%.0f,tokens,%zu,tok_s,%.2f\n", ms, out.size(), out.size() * 1000.0 / ms);
        std::printf("greedy:");
        for (int v : out) std::printf(" %d", v);
        std::printf("\n");
    }
    return 0;
}
