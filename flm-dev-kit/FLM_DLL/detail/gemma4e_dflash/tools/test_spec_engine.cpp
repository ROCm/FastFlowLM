// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file test_spec_engine.cpp
/// \brief Engine-level test rig for gemma4e_dflash speculation (README.md, "Tools").
///
/// For every prompt: prefill, then (A) plain greedy decode of N tokens and (B) the same N tokens through speculate().
/// The two streams must be identical (a position where the plain run's top-2 logit margin is below 1 bf16 ulp of the
/// top-1 value is a tie and is reported, not failed). Prints acceptance length, per-phase timing, tok/s of both runs.
///
/// Lives in tools/ so the Makefile's *.cpp glob misses it. Build (from FLM_DLL/detail/gemma4e_dflash):
///   g++-13 -O3 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -std=c++20 -I../../include -I../q4nx -I$XRT_ROOT/include \
///       tools/test_spec_engine.cpp -L../../build/lib -lgemma4e_dflash -lxrt_coreutil \
///       -Wl,-rpath,$PWD/../../build/lib -o ../../build/test_spec_engine
///   HRX: add -DFLM_USE_HRX -I$HRX_HOME/include/hrx -I$HRX_HOME/include, link the HRX engine build and -L$HRX_HOME/lib -lhrx
///   instead of -lxrt_coreutil (work/scripts/build_engine_hrx.sh builds the engine into build/lib_hrx).
/// Run:
///   test_spec_engine <model_dir> <exec_path> <draft_dir|-> <ids.json> <n_prompts> <n_tokens> [skip_plain=0]
/// exec_path holds xclbins/<basename(model_dir)>/ ; ids.json = [{"id":..,"ids":[..]}] (chat-templated prompts).
#include "models/gemma4e_dflash/gemma4e_dflash.hpp"
#include "lm_config.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
static bool is_stop(int t) { return t == 1 || t == 106; }   // <eos>, <end_of_turn>
static float bf2f(bf16 v) { uint16_t h; std::memcpy(&h, &v, 2); uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }

static float ulp_bf16(float v) { return std::ldexp(1.0f, (int)std::floor(std::log2(std::max(std::fabs(v), 1e-30f))) - 7); }
struct top2 { int id; float v1, v2; };
static top2 top2_of(const bf16* p, int n) {
    top2 r{0, bf2f(p[0]), -1e30f};
    for (int i = 1; i < n; i++) {
        float v = bf2f(p[i]);
        if (v > r.v1) { r.v2 = r.v1; r.v1 = v; r.id = i; }
        else if (v > r.v2) r.v2 = v;
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: %s model_dir exec_path draft_dir|- ids.json n_prompts n_tokens [skip_plain]\n", argv[0]); return 2; }
    const std::string model_dir = argv[1], exec_path = argv[2], draft_dir = argv[3], ids_path = argv[4];
    const int n_prompts = std::atoi(argv[5]), n_tokens = std::atoi(argv[6]);
    const bool skip_plain = argc > 7 && std::atoi(argv[7]) == 1;
    const bool restore_test = argc > 7 && std::atoi(argv[7]) == 3;   // 3: checkpoint/restore scenario (cold draft), see below
    const bool repeat_plain = argc > 7 && std::atoi(argv[7]) == 2;   // 2: run the plain stream twice and compare (determinism check)

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = std::getenv("TEST_XCLBIN_FOLDER") ? std::getenv("TEST_XCLBIN_FOLDER") : model_dir.substr(model_dir.find_last_of('/') + 1);
    config.exec_path = exec_path;
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
        f >> config._json_config;
        // same normalisation as the runtime's LM_Config::_normalize_multi_modal
        std::string vw = config._json_config.value("vision_model_weight", std::string());
        std::string aw = config._json_config.value("audio_model_weight", std::string());
        config._json_config["is_vlm"] = !vw.empty();
        config._json_config["is_audio"] = !aw.empty();
        if (!vw.empty()) config._json_config["vision_model_weight"] = model_dir + "/" + vw;
        if (!aw.empty()) config._json_config["audio_model_weight"] = model_dir + "/" + aw;
        std::ifstream pf(model_dir + "/processor_config.json");
        if (pf) { nlohmann::json pj; pf >> pj; config._json_config["processor_config"] = pj; }
    }
    nlohmann::json prompts;
    { std::ifstream f(ids_path); f >> prompts; }

    flm_rt::device npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);
    gemma4e_dflash engine(config, &npu_mgr, 4096);
    {
        Q4NX q4nx(model_dir);
        engine.load_weights(q4nx);
    }
    bool have_draft = false;
    if (draft_dir != "-") have_draft = engine.load_draft(draft_dir);
    std::printf("engine ready, draft %s\n", have_draft ? "loaded" : "NOT loaded");
    engine.set_stop_tokens({1, 106});   // <eos>, <end_of_turn>: a test stream normally never reaches them in n_tokens

    const int V = engine.vocab_size();
    if (restore_test) {
        std::vector<int> ids = prompts[0]["ids"].get<std::vector<int>>();
        auto decode_spec = [&](std::vector<int>& seq, int upto) {
            while ((int)seq.size() < upto) {
                std::vector<int> got = engine.speculate(seq.back(), 7);
                if (got.empty()) got = {top2_of(engine.forward(seq.back()).data(), V).id};
                for (int t : got) seq.push_back(t);
            }
        };
        // reference: plain
        engine.clear_context();
        std::vector<int> pi_ids = ids;
        std::vector<int> plain{top2_of(engine.prefill(pi_ids, nullptr).data(), V).id};
        while ((int)plain.size() < 90) plain.push_back(top2_of(engine.forward(plain.back()).data(), V).id);
        // speculative with a checkpoint after ~30 tokens, a run to 60, a restore, and a second run to 60
        engine.clear_context();
        std::vector<int> sids = ids;
        std::vector<int> seq{top2_of(engine.prefill(sids, nullptr).data(), V).id};
        decode_spec(seq, 30);
        // the last token has no row yet; the checkpoint covers every row before it
        const int ck_len = engine.checkpoint();
        const std::vector<int> at_ck(seq.begin(), seq.end());
        const int ids_before = engine.get_current_context_length();
        decode_spec(seq, 60);
        std::vector<int> first(seq.begin(), seq.end());
        engine.restore();
        std::printf("restore: context %d -> %d (checkpoint %d)\n", (int)ids_before, engine.get_current_context_length(), ck_len);
        // re-enter at the token the checkpoint left off: it is at_ck.back() truncated to the same position
        std::vector<int> seq2(at_ck.begin(), at_ck.end());
        std::printf("draft ready after restore: %s (taps gone: cold draft expected to still draft)\n", engine.draft_ready() ? "yes" : "no");
        decode_spec(seq2, 60);
        int d1 = -1, d2 = -1;
        for (int i = 0; i < 60; i++) { if (d1 < 0 && first[i] != plain[i]) d1 = i; if (d2 < 0 && seq2[i] != plain[i]) d2 = i; }
        std::printf("spec run vs plain: %s; run after restore vs plain: %s\n", d1 < 0 ? "identical" : ("diverge at " + std::to_string(d1)).c_str(),
                    d2 < 0 ? "identical" : ("diverge at " + std::to_string(d2)).c_str());
        std::printf("%s\n", engine.speculation_stats().c_str());
        return (d1 < 0 && d2 < 0) ? 0 : 1;
    }

    int total_mismatch = 0, total_ties = 0, total_tokens_plain = 0, total_tokens_spec = 0;
    double t_plain = 0, t_spec = 0, t_prime = 0;
    long long cyc_total = 0, tok_total = 0;

    for (int pi = 0; pi < n_prompts && pi < (int)prompts.size(); pi++) {
        std::vector<int> ids = prompts[pi]["ids"].get<std::vector<int>>();
        std::printf("\n== prompt %d (%s): %zu tokens ==\n", pi, prompts[pi]["id"].get<std::string>().c_str(), ids.size());

        std::vector<int> plain; std::vector<float> margin, ulp1;   // margin[i] = top1 - top2 logit when plain[i] was chosen; ulp1 = bf16 ulp of top1
        if (!skip_plain) {
            engine.clear_context();
            std::vector<int> pids = ids;
            buffer<bf16> lg = engine.prefill(pids, nullptr);
            top2 t = top2_of(lg.data(), V);
            plain.push_back(t.id); margin.push_back(t.v1 - t.v2); ulp1.push_back(ulp_bf16(t.v1));
            auto t0 = clk::now();
            while ((int)plain.size() < n_tokens && !is_stop(plain.back())) {
                buffer<bf16> l = engine.forward(plain.back());
                top2 q = top2_of(l.data(), V);
                plain.push_back(q.id); margin.push_back(q.v1 - q.v2); ulp1.push_back(ulp_bf16(q.v1));
            }
            double dt = ms_since(t0);
            t_plain += dt; total_tokens_plain += (int)plain.size() - 1;
            std::printf("plain: %zu tokens in %.1f ms = %.2f tok/s (decode only)\n", plain.size() - 1, dt, (plain.size() - 1) * 1000.0 / dt);
            if (repeat_plain) {
                engine.clear_context();
                std::vector<int> p2ids = ids;
                buffer<bf16> l2 = engine.prefill(p2ids, nullptr);
                std::vector<int> pl2{top2_of(l2.data(), V).id};
                while ((int)pl2.size() < n_tokens && !is_stop(pl2.back())) pl2.push_back(top2_of(engine.forward(pl2.back()).data(), V).id);
                int first = -1;
                for (size_t i = 0; i < plain.size() && i < pl2.size(); i++) if (pl2[i] != plain[i]) { first = (int)i; break; }
                std::printf("plain vs plain (determinism): %s\n", first < 0 ? "identical" : ("first divergence at token " + std::to_string(first)).c_str());
            }
        }

        if (!have_draft) continue;
        engine.clear_context();
        engine.reset_speculation_stats();
        std::vector<int> sids = ids;
        buffer<bf16> lg;
        if (std::getenv("TEST_PROMPT_BY_FORWARD")) {     // feed the prompt one token at a time: taps then come from the decode path
            for (size_t i = 0; i + 1 < sids.size(); i++) engine.forward(sids[i]);
            lg = engine.forward(sids.back());
        } else {
            lg = engine.prefill(sids, nullptr);
        }
        std::vector<int> spec;
        spec.push_back(top2_of(lg.data(), V).id);
        if (const char* vl = std::getenv("TEST_VERIFY_LOOP")) {   // verify-only loop: dummy drafts, commit one row per cycle (isolates verify + lm_head_8)
            const int n_loop = std::atoi(vl);
            int last = spec.back();
            for (int c = 0; c < n_loop; c++) {
                int d[7] = {ids[1 % ids.size()], ids[2 % ids.size()], ids[3 % ids.size()], ids[4 % ids.size()], ids[5 % ids.size()], ids[6 % ids.size()], ids[7 % ids.size()]};
                auto tv = clk::now();
                const bf16* rows = engine.dflash_verify(last, d, 7);
                if (rows == nullptr) { std::printf("verify loop: declined at cycle %d\n", c); return 1; }
                const double ms = ms_since(tv);
                last = top2_of(rows, V).id;
                engine.dflash_commit(1);
                std::printf("verify loop cycle %d: ctx %d verify %.1f ms next %d\n", c, engine.get_current_context_length(), ms, last);
                std::fflush(stdout);
                if (const char* sl = std::getenv("TEST_LOOP_SLEEP_MS")) { struct timespec ts{0, (long)std::atoi(sl) * 1000000L}; nanosleep(&ts, nullptr); }
            }
            std::printf("%s\n", engine.speculation_stats().c_str());
            return 0;
        }
        auto t0 = clk::now();
        int declines = 0;
        while ((int)spec.size() < n_tokens && !is_stop(spec.back())) {
            std::vector<int> got;
            if (std::getenv("FLM_DFLASH_TRACE")) {          // explicit draft/verify/commit with a per-cycle dump
                int d[7];
                if (engine.dflash_draft(spec.back(), d) == 7) {
                    const bf16* rows = engine.dflash_verify(spec.back(), d, 7);
                    int am[8]; for (int i = 0; i < 8; i++) am[i] = top2_of(rows + (size_t)i * V, V).id;
                    int acc = 0; while (acc < 7 && am[acc] == d[acc]) acc++;
                    std::printf("  cyc: last %d drafts [%d %d %d %d %d %d %d] target [%d %d %d %d %d %d %d %d] acc %d\n", spec.back(), d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                                am[0], am[1], am[2], am[3], am[4], am[5], am[6], am[7], acc);
                    engine.dflash_commit(acc + 1);
                    got.assign(am, am + acc + 1);
                }
            } else {
                got = engine.speculate(spec.back(), 7);
            }
            if (got.empty()) {                       // decline: plain step
                declines++;
                buffer<bf16> l = engine.forward(spec.back());
                spec.push_back(top2_of(l.data(), V).id);
            } else {
                for (int t : got) spec.push_back(t);
            }
        }
        double dt = ms_since(t0);
        // a batch may run past the stop token (the engine cuts at stop_ids; the trace path does not): cut at the first stop
        for (size_t i = 0; i < spec.size(); i++) if (is_stop(spec[i])) { spec.resize(i + 1); break; }
        if ((int)spec.size() > n_tokens) spec.resize(n_tokens);
        const int ns = (int)spec.size();
        t_spec += dt; total_tokens_spec += ns - 1;
        t_prime += engine.last_speculation_prime_us() / 1000.0;
        cyc_total += engine.speculation_cycles(); tok_total += (long long)(engine.mean_accepted_length() * engine.speculation_cycles() + 0.5);
        std::printf("spec : %d tokens in %.1f ms = %.2f tok/s; %s; declines %d\n", ns - 1, dt, (ns - 1) * 1000.0 / dt,
                    engine.speculation_stats().c_str(), declines);
        std::printf("       %s\n", engine.speculation_timing().c_str());
        if (const char* dump = std::getenv("TEST_DUMP_STREAMS")) {   // one line per prompt: id and the speculative token stream (diff two builds/runtimes)
            std::ofstream df(dump, std::ios::app);
            df << prompts[pi]["id"].get<std::string>();
            for (int t : spec) df << ' ' << t;
            df << '\n';
        }

        if (!skip_plain) {
            int mism = 0, ties = 0;
            const int ncmp = (int)std::min(plain.size(), spec.size());
            for (int i = 0; i < ncmp; i++) {
                if (plain[i] == spec[i]) continue;
                // the streams diverge here; everything after is incomparable. A tie in the plain logits excuses the first divergence.
                bool tie = margin[i] <= ulp1[i];             // top-2 within one bf16 ulp of the top-1 value
                std::printf("       FIRST DIVERGENCE at token %d: plain %d spec %d, plain top1-top2 margin %.4f%s\n", i, plain[i], spec[i], margin[i], tie ? " (TIE, excused)" : "");
                if (tie) ties++; else mism++;
                break;
            }
            total_mismatch += mism; total_ties += ties;
            if (mism == 0 && ties == 0) std::printf("       streams identical (%d tokens compared, plain %zu spec %zu)\n", ncmp, plain.size(), spec.size());
        }
    }

    std::printf("\n==== summary ====\n");
    if (!skip_plain) std::printf("plain  : %d tokens, %.2f tok/s\n", total_tokens_plain, total_tokens_plain * 1000.0 / std::max(t_plain, 1e-9));
    if (have_draft) {
        std::printf("spec   : %d tokens, %.2f tok/s, mean AL %.2f\n", total_tokens_spec, total_tokens_spec * 1000.0 / std::max(t_spec, 1e-9),
                    cyc_total ? (double)tok_total / (double)cyc_total : 0.0);
        std::printf("stream mismatches: %d, excused ties: %d\n", total_mismatch, total_ties);
    }
    return total_mismatch ? 1 : 0;
}
