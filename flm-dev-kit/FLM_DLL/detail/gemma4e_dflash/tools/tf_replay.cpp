// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file tf_replay.cpp
/// \brief Acceptance-length rig for gemma4e_dflash: teacher-forced and free-running replays of fixed token sequences.
///
/// Input: seqs.json = [{"id", "cat", "prompt_ids":[..], "gen_ids":[..]}] (llama.cpp greedy continuations of the prompts, thinking on).
///   mode tf   : prefill prompt_ids, then cycles at pos0 = |prompt|, ...: draft after tokens[pos0], k = matching prefix of the drafts with the
///               forced continuation, verify the FORCED tokens (so the target taps follow the sequence), commit k+1. Same rule as
///               tools/ref/dflash_ref.py replay_al, so the AL is directly comparable with the offline sweep (HF BF16 taps).
///               Also counts how often the NPU target's own argmax equals the forced token (target agreement).
///   mode free : prefill prompt_ids, then greedy speculative decoding (speculate()) for n_tokens: the AL of the real loop on these prompts.
/// With FLM_DFLASH_DUMP=<prefix> the engine writes the injected taps and the drafts (see gemma4e_dflash_spec_state.hpp) for the python analysis.
/// Output: one line per sequence and an "ALL" line; per-cycle trace as JSON lines with TF_TRACE=<file>.
///
/// Build: see README.md, "Tools"
/// Run:   tf_replay <model_dir> <exec_path> <draft_dir> <seqs.json> <idx list e.g. 0-13> <tf|free> [n_tokens=400]
#include "models/gemma4e_dflash/gemma4e_dflash.hpp"
#include "lm_config.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
static float bf2f(bf16 v) { uint16_t h; std::memcpy(&h, &v, 2); uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static int argmax(const bf16* p, int n) { int b = 0; float bv = bf2f(p[0]); for (int i = 1; i < n; i++) { float v = bf2f(p[i]); if (v > bv) { bv = v; b = i; } } return b; }

static std::vector<int> parse_idx(const std::string& s) {
    std::vector<int> r; size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
        std::string t = s.substr(i, j - i); size_t d = t.find('-');
        if (d == std::string::npos) r.push_back(std::atoi(t.c_str()));
        else for (int a = std::atoi(t.substr(0, d).c_str()), b = std::atoi(t.substr(d + 1).c_str()); a <= b; a++) r.push_back(a);
        i = j + 1;
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s model_dir exec_path draft_dir seqs.json idx tf|free [n_tokens]\n", argv[0]); return 2; }
    const std::string model_dir = argv[1], exec_path = argv[2], draft_dir = argv[3], seqs_path = argv[4], mode = argv[6];
    const std::vector<int> idx = parse_idx(argv[5]);
    const int n_tokens = argc > 7 ? std::atoi(argv[7]) : 400;

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = std::getenv("TEST_XCLBIN_FOLDER") ? std::getenv("TEST_XCLBIN_FOLDER") : "Nemotron-3.5-Bolt-E4B-DFlash-NPU2";
    config.exec_path = exec_path;
    {
        std::ifstream f(model_dir + "/config.json");
        f >> config._json_config;
        std::string vw = config._json_config.value("vision_model_weight", std::string());
        std::string aw = config._json_config.value("audio_model_weight", std::string());
        config._json_config["is_vlm"] = !vw.empty();
        config._json_config["is_audio"] = !aw.empty();
        if (!vw.empty()) config._json_config["vision_model_weight"] = model_dir + "/" + vw;
        if (!aw.empty()) config._json_config["audio_model_weight"] = model_dir + "/" + aw;
        std::ifstream pf(model_dir + "/processor_config.json");
        if (pf) { nlohmann::json pj; pf >> pj; config._json_config["processor_config"] = pj; }
    }
    nlohmann::json seqs;
    { std::ifstream f(seqs_path); f >> seqs; }

    flm_rt::device npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);
    gemma4e_dflash engine(config, &npu_mgr, std::getenv("TF_MAXL") ? std::atoi(std::getenv("TF_MAXL")) : 4096);
    { Q4NX q4nx(model_dir); engine.load_weights(q4nx); }
    if (!engine.load_draft(draft_dir)) { std::fprintf(stderr, "draft not loaded\n"); return 1; }
    engine.set_stop_tokens({1, 106});
    const int V = engine.vocab_size();
    FILE* tr = std::getenv("TF_TRACE") ? std::fopen(std::getenv("TF_TRACE"), "w") : nullptr;

    long long all_cyc = 0, all_emit = 0, all_tgt = 0, all_tgt_ok = 0;
    double all_ms = 0; long long all_tok = 0;
    for (int si : idx) {
        if (si < 0 || si >= (int)seqs.size()) continue;
        const auto& s = seqs[si];
        std::vector<int> P = s["prompt_ids"].get<std::vector<int>>();
        std::vector<int> G = s["gen_ids"].get<std::vector<int>>();
        std::vector<int> toks = P; toks.insert(toks.end(), G.begin(), G.end());
        const int T = (int)toks.size(), PL = (int)P.size();
        engine.clear_context();
        engine.reset_speculation_stats();
        std::vector<int> pids = P;
        buffer<bf16> lg = engine.prefill(pids, nullptr);
        long long cyc = 0, emit = 0, tgt = 0, tgt_ok = 0;
        auto t0 = clk::now();
        if (mode == "tf") {
            tgt++; tgt_ok += argmax(lg.data(), V) == toks[PL];
            int pos0 = PL;
            while (pos0 + 8 < T) {
                int d[7];
                if (engine.dflash_draft(toks[pos0], d) != 7) { std::fprintf(stderr, "seq %d: draft declined at %d (guard? set FLM_DFLASH_GUARD=0)\n", si, pos0); break; }
                int k = 0;
                while (k < 7 && d[k] == toks[pos0 + 1 + k]) k++;
                const bf16* rows = engine.dflash_verify(toks[pos0], &toks[pos0 + 1], 7);
                if (!rows) { std::fprintf(stderr, "seq %d: verify declined\n", si); break; }
                int am[8];
                for (int i = 0; i < 8; i++) am[i] = argmax(rows + (size_t)i * V, V);
                for (int i = 0; i <= k; i++) { tgt++; tgt_ok += am[i] == toks[pos0 + 1 + i]; }
                if (tr) {
                    std::fprintf(tr, "{\"seq\":%d,\"pos0\":%d,\"id_last\":%d,\"k\":%d,\"d\":[%d,%d,%d,%d,%d,%d,%d],\"am\":[%d,%d,%d,%d,%d,%d,%d,%d]}\n", si, pos0, toks[pos0], k,
                                 d[0], d[1], d[2], d[3], d[4], d[5], d[6], am[0], am[1], am[2], am[3], am[4], am[5], am[6], am[7]);
                }
                engine.dflash_commit(k + 1);
                cyc++; emit += k + 1; pos0 += k + 1;
            }
        } else {
            std::vector<int> seq{argmax(lg.data(), V)};
            int declines = 0;
            while ((int)seq.size() < n_tokens && seq.back() != 1 && seq.back() != 106) {
                std::vector<int> got = engine.speculate(seq.back(), 7);
                if (got.empty()) { declines++; seq.push_back(argmax(engine.forward(seq.back()).data(), V)); }
                else for (int t : got) seq.push_back(t);
            }
            cyc = engine.speculation_cycles();
            emit = (long long)(engine.mean_accepted_length() * cyc + 0.5);
            all_tok += (long long)seq.size() - 1;
            if (declines) std::printf("  declines %d\n", declines);
        }
        const double ms = ms_since(t0);
        all_ms += ms;
        all_cyc += cyc; all_emit += emit; all_tgt += tgt; all_tgt_ok += tgt_ok;
        std::printf("%s %d %s %s P=%d T=%d cycles=%lld AL=%.3f", mode.c_str(), si, s["id"].get<std::string>().c_str(), s["cat"].get<std::string>().c_str(), PL, T,
                    cyc, cyc ? (double)emit / cyc : 0.0);
        if (mode == "tf") std::printf(" target_agree=%.3f", tgt ? (double)tgt_ok / tgt : 0.0);
        std::printf(" %.0f ms\n", ms);
        std::printf("   %s\n", engine.speculation_timing().c_str());
        std::fflush(stdout);
    }
    std::printf("%s ALL cycles=%lld AL=%.3f", mode.c_str(), all_cyc, all_cyc ? (double)all_emit / all_cyc : 0.0);
    if (mode == "tf") std::printf(" target_agree=%.3f", all_tgt ? (double)all_tgt_ok / all_tgt : 0.0);
    else std::printf(" tok/s=%.2f", all_tok * 1000.0 / std::max(all_ms, 1e-9));
    std::printf("\n");
    if (tr) std::fclose(tr);
    return 0;
}
