// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file test_swa_stream.cpp
/// \brief Engine-level rig for the sliding-window K/V buffer change (linear window buffer with host compaction).
///
/// The same driver is compiled against the UNMODIFIED base engine (-DENGINE_NPU, class gemma4e_npu with the 512-row ring) and against
/// gemma4e_dflash (linear window buffer); the greedy token streams must be identical. Stop tokens are ignored (the argmax over the
/// other tokens is taken) so a stream can run for thousands of steps and cross several buffer compactions.
///
/// Modes (argv[6]): stream   prefill, then n decode steps; writes the token ids to <out>
///                  state    prefill, decode m, read_state, decode k (S1), write_state, decode k (S2), compare S1/S2;
///                           then checkpoint at the same length, decode k (S3), restore, decode k (S4), compare
/// Build (from FLM_DLL/detail/gemma4e_dflash):
///   g++-13 -O3 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -std=c++20 -I../../include -I../q4nx -I$XRT_ROOT/include tools/test_swa_stream.cpp \
///       -L../../build/lib -lgemma4e_dflash -lq4_npu_eXpress -lxrt_coreutil -Wl,-rpath,$PWD/../../build/lib -o ../../build/test_swa_stream_dflash
///   (same with -DENGINE_NPU and -lgemma4e_npu for the oracle build)
/// Run: test_swa_stream <model_dir> <exec_path> <ids.json> <prompt_index> <n_tokens> <stream|state> <out|-> [max_l=12288]
#ifdef ENGINE_NPU
#include "models/gemma4e/gemma4e_npu.hpp"
using engine_t = gemma4e_npu;
#else
#include "models/gemma4e_dflash/gemma4e_dflash.hpp"
using engine_t = gemma4e_dflash;
#endif
#include "lm_config.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static float bf2f(bf16 v) { uint16_t h; std::memcpy(&h, &v, 2); uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static bool is_stop(int t) { return t == 1 || t == 106; }
static int argmax_nostop(const bf16* p, int n) {
    int best = -1; float bv = -1e30f;
    for (int i = 0; i < n; i++) { if (is_stop(i)) continue; float v = bf2f(p[i]); if (v > bv) { bv = v; best = i; } }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: %s model_dir exec_path ids.json prompt_index n_tokens stream|state out|- [max_l]\n", argv[0]); return 2; }
    const std::string model_dir = argv[1], exec_path = argv[2], ids_path = argv[3];
    const int pidx = std::atoi(argv[4]), n_tokens = std::atoi(argv[5]);
    const std::string mode = argv[6], out_path = argv[7];
    const int max_l = argc > 8 ? std::atoi(argv[8]) : 12288;

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = model_dir.substr(model_dir.find_last_of('/') + 1);
    config.exec_path = exec_path;
    {
        std::ifstream f(model_dir + "/config.json");
        if (!f) { std::fprintf(stderr, "cannot read config.json\n"); return 1; }
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
    nlohmann::json prompts;
    { std::ifstream f(ids_path); f >> prompts; }
    std::vector<int> ids = prompts[pidx]["ids"].get<std::vector<int>>();

    xrt::device npu_dev(0);
    npu_xclbin_manager npu_mgr(npu_device::device_npu2, &npu_dev, false);
    engine_t engine(config, &npu_mgr, max_l);
    { Q4NX q4nx(model_dir); engine.load_weights(q4nx); }

    auto decode = [&](std::vector<int>& seq, int n) {
        for (int i = 0; i < n; i++) { auto lg = engine.forward(seq.back()); seq.push_back(argmax_nostop(lg.data(), (int)lg.size())); }
    };

    engine.clear_context();
    std::vector<int> pids = ids;
    auto t0 = clk::now();
    auto lg0 = engine.prefill(pids, nullptr);
    std::vector<int> seq{argmax_nostop(lg0.data(), (int)lg0.size())};
    const double t_prefill = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("prompt %d tokens, prefill %.2f s\n", (int)ids.size(), t_prefill);

    if (mode == "stream") {
        t0 = clk::now();
        decode(seq, n_tokens - 1);
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("decoded %d tokens in %.2f s = %.2f tok/s; context %d\n", (int)seq.size(), dt, (seq.size() - 1) / dt, engine.get_current_context_length());
        if (out_path != "-") { std::ofstream o(out_path); for (int t : seq) o << t << "\n"; }
        return 0;
    }
    // state mode
    const int m = n_tokens, k = 120;
    decode(seq, m);
    auto st = engine.new_state();
    if (!st) { std::printf("new_state unavailable\n"); return 3; }
    const int ctx_at_state = engine.get_current_context_length();
    const std::vector<int> seq_at_state = seq;
    const int rl = engine.read_state(*st, 0);
    std::printf("read_state at %d -> %d\n", ctx_at_state, rl);
    std::vector<int> s1 = seq; decode(s1, k);
    const int wl = engine.write_state(*st);
    std::printf("write_state -> %d (context %d)\n", wl, engine.get_current_context_length());
    std::vector<int> s2 = seq_at_state; decode(s2, k);
    int d12 = -1; for (size_t i = 0; i < s1.size() && d12 < 0; i++) if (s1[i] != s2[i]) d12 = (int)i;
    std::printf("state round trip (decode after read vs after write): %s\n", d12 < 0 ? "identical" : ("DIVERGE at " + std::to_string(d12)).c_str());
    // checkpoint / restore at the same length
    engine.write_state(*st);
    const int ck = engine.checkpoint();
    std::vector<int> s3 = seq_at_state; decode(s3, k);
    const int rs = engine.restore();
    std::printf("checkpoint %d restore %d (context %d)\n", ck, rs, engine.get_current_context_length());
    std::vector<int> s4 = seq_at_state; decode(s4, k);
    int d34 = -1; for (size_t i = 0; i < s3.size() && d34 < 0; i++) if (s3[i] != s4[i]) d34 = (int)i;
    std::printf("checkpoint/restore: %s\n", d34 < 0 ? "identical" : ("DIVERGE at " + std::to_string(d34)).c_str());
    int d13 = -1; for (size_t i = 0; i < s1.size() && d13 < 0; i++) if (s1[i] != s3[i]) d13 = (int)i;
    std::printf("s1 vs s3: %s\n", d13 < 0 ? "identical" : ("DIVERGE at " + std::to_string(d13)).c_str());
    if (out_path != "-") { std::ofstream o(out_path); for (int t : s1) o << t << "\n"; }
    return (d12 < 0 && d34 < 0 && d13 < 0) ? 0 : 1;
}
