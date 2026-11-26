// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_engine_run.cpp
/// \brief drive libinkling2_npu.so through the causal_lm interface over reference prompts
///
/// Usage: inkling2_engine_run PKG_DIR GOLDEN_DIR NAME [NAME ...] [--ctx N] [--new N] [--no-state] [--dump DIR]
///   GOLDEN_DIR/NAME.{ids,new_ids,first_logits} as for inkling2_host_run; --dump writes the
///   greedy ids to DIR/NAME.got_ids.
/// Per prompt: prefill, first-step logits against the reference (KL, top-20, argmax), greedy
/// decode of --new tokens (default: the reference length) and its match with the reference.
/// Unless --no-state, then checks that a saved state continues exactly like the live engine
/// (bit-identical logits over 8 decode steps):
///   A checkpoint() after the prompt, decode, restore(), decode again
///   B read_state(kv_from 0), clear_context(), write_state()
///   C a state split in two: s1 at half the prompt (kv_from 0), s2 at the end with
///     kv_from = half; clear_context(), write_state_kv(s1), write_state(s2); compared with
///     the engine that prefilled the same two halves
/// Exit status 1 if any state check differs.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#endif

#include "lm_config.hpp"
#include "models/inkling2/inkling2_npu.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

using clk = std::chrono::steady_clock;
static constexpr int VOCAB = 200058;
static constexpr int NCHECK = 8;

static double sec_since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static std::vector<int> read_ids(const std::string& path) {
    std::ifstream f(path);
    std::vector<int> v;
    int x;
    while (f >> x) v.push_back(x);
    return v;
}

static std::vector<float> to_f32(const buffer<bf16>& b) {
    std::vector<float> v(VOCAB);
    for (int i = 0; i < VOCAB; i++) v[i] = float(b[i]);
    return v;
}

static int argmax(const std::vector<float>& l) { return (int)(std::max_element(l.begin(), l.end()) - l.begin()); }

static std::vector<int> topk(const std::vector<float>& l, int k) {
    std::vector<int> idx(l.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    idx.resize(k);
    return idx;
}

/// \brief greedy decode n tokens from `next`; the logits of every step are appended to `trace`
static std::vector<int> greedy(inkling2_npu& e, int next, int n, std::vector<std::vector<float>>* trace = nullptr) {
    std::vector<int> out;
    for (int i = 0; i < n; i++) {
        out.push_back(next);
        auto l = to_f32(e.forward(next));
        if (trace) trace->push_back(l);
        next = argmax(l);
    }
    return out;
}

static bool same(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::memcmp(a[i].data(), b[i].data(), VOCAB * sizeof(float))) return false;
    return true;
}

// user + system seconds of every thread of the process (0 where unsupported)
static double cpu_seconds() {
#ifndef _WIN32
    rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 + u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
#else
    return 0;
#endif
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s PKG_DIR GOLDEN_DIR NAME [NAME ...] [--ctx N] [--new N] [--no-state] [--dump DIR]\n", argv[0]);
        return 2;
    }
    std::string pkg = argv[1], gdir = argv[2];
    std::vector<std::string> names;
    int ctx = 0, n_new = -1;
    bool state = true;
    std::string dump;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--new") && i + 1 < argc) n_new = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--no-state")) state = false;
        else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else names.push_back(argv[i]);
    }
    if (!ctx) {
        for (const auto& n : names) ctx = std::max<int>(ctx, (int)read_ids(gdir + "/" + n + ".ids").size() + 1024);
    }

    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    LM_Config cfg;
    cfg.model_path = pkg;
    cfg.model_name = std::filesystem::path(pkg).filename().string();
    cfg.exec_path = std::filesystem::current_path().string();   // kernels in ./xclbins/<model_name>/
    {
        std::ifstream cf(std::filesystem::path(pkg) / "config.json");
        cfg._json_config = nlohmann::json::parse(cf);
        cfg._json_config["is_vlm"] = false;
        cfg._json_config["is_audio"] = false;
    }
    auto t = clk::now();
    Q4NX q4nx(pkg);
    inkling2_npu e(cfg, &npu, ctx);
    e.load_weights(q4nx);
    std::printf("load %.2f s, ctx %d\n", sec_since(t), ctx);

    int failures = 0;
    for (const auto& name : names) {
        const std::string base = gdir + "/" + name;
        std::vector<int> ids = read_ids(base + ".ids");
        const std::vector<int> ref_new = read_ids(base + ".new_ids");
        std::vector<float> ref(VOCAB);
        {
            std::ifstream f(base + ".first_logits", std::ios::binary);
            f.read(reinterpret_cast<char*>(ref.data()), VOCAB * sizeof(float));
            if (!f) {
                std::fprintf(stderr, "cannot read %s.first_logits\n", base.c_str());
                return 2;
            }
        }
        const int n = n_new >= 0 ? n_new : (int)ref_new.size();

        e.clear_context();
        const double c0 = cpu_seconds();
        t = clk::now();
        auto first = to_f32(e.prefill(ids));
        const double tp = sec_since(t);
        const double cp = cpu_seconds() - c0;
        auto lse = [](const std::vector<float>& l) {
            const float mx = *std::max_element(l.begin(), l.end());
            double s = 0;
            for (float v : l) s += std::exp((double)v - mx);
            return mx + std::log(s);
        };
        const double la = lse(first), lb = lse(ref);
        double kl = 0;
        for (int i = 0; i < VOCAB; i++) kl += std::exp(ref[i] - lb) * ((ref[i] - lb) - (first[i] - la));
        const auto a = topk(first, 20), b = topk(ref, 20);
        int overlap = 0;
        for (int x : a) overlap += (int)std::count(b.begin(), b.end(), x);

        // the states are taken right after the prompt, before decoding moves the engine on
        std::unique_ptr<hybrid_engine_state> full, half_state, tail;
        if (state) {
            e.checkpoint();
            full = e.new_state();
            e.read_state(*full, 0);
        }
        t = clk::now();
        const auto got = greedy(e, a[0], n);
        const double td = sec_since(t);
        int match = 0;
        while (match < (int)std::min(got.size(), ref_new.size()) && got[match] == ref_new[match]) match++;
        std::printf("%s: prefill cpu %.2f s (%.1f cores busy)\n", name.c_str(), cp, cp / tp);
        std::printf("%s: %zu tok, prefill %.1f tok/s, KL %.5f, top20 %d/20, argmax %d (ref %d), decode %.1f ms/tok, "
                    "greedy %d/%zu ref\n",
                    name.c_str(), ids.size(), ids.size() / tp, kl, overlap, a[0], b[0], 1000 * td / std::max(n, 1), match,
                    ref_new.size());
        if (!dump.empty()) {
            std::ofstream f(dump + "/" + name + ".got_ids");
            for (int x : got) f << x << "\n";
        }
        if (!state) continue;

        // live reference trace: restore the prompt state (A), then decode NCHECK steps
        std::vector<std::vector<float>> live, other;
        if (e.restore() != (int)ids.size()) {
            std::printf("  A restore: wrong length\n");
            failures++;
        }
        const auto again = greedy(e, a[0], NCHECK, &live);
        const int m = std::min(n, NCHECK);
        const bool okA = std::equal(again.begin(), again.begin() + m, got.begin());
        // B: write the full state into a cleared engine
        e.clear_context();
        if (e.write_state(*full) != (int)ids.size()) failures++;
        greedy(e, a[0], NCHECK, &other);
        const bool okB = same(live, other);
        // C: two states; the first carries rows [0, h), the second [h, L)
        const int h = (int)ids.size() / 2;
        bool okC = true;
        if (h > 0) {
            e.clear_context();
            std::vector<int> head(ids.begin(), ids.begin() + h), rest(ids.begin() + h, ids.end());
            e.prefill(head);
            half_state = e.new_state();
            e.read_state(*half_state, 0);
            e.prefill(rest);
            tail = e.new_state();
            e.read_state(*tail, h);
            // the split prefill runs other chunk shapes than the whole one, so its own decode
            // (not `live`) is what the restored engine must reproduce
            std::vector<std::vector<float>> live_split;
            greedy(e, a[0], NCHECK, &live_split);
            e.clear_context();
            const bool w1 = e.write_state_kv(*half_state) == h;
            const bool w2 = e.write_state(*tail) == (int)ids.size();
            other.clear();
            greedy(e, a[0], NCHECK, &other);
            okC = w1 && w2 && same(live_split, other);
        }
        std::printf("  state: A restore %s, B write_state %s, C split at %d %s\n", okA ? "ok" : "DIFFERS",
                    okB ? "ok" : "DIFFERS", h, okC ? "ok" : "DIFFERS");
        failures += !okA + !okB + !okC;
    }
    return failures ? 1 : 0;
}
