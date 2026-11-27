// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_host_run.cpp
/// \brief run a reference prompt through the host model and compare with its golden
///
/// Usage: inkling2_host_run model.q4nx DIR NAME [--ctx N] [--new N] [--chunk N]
///   DIR/NAME.ids          prompt token ids, text, one per line
///   DIR/NAME.new_ids      the reference greedy continuation
///   DIR/NAME.first_logits the reference logits after the prompt, raw f32 [vocab]
/// Prints the first-step logit error and top-k overlap, then decodes greedily and
/// reports the first divergence from the reference continuation.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "../inkling2_host.hpp"

using namespace inkling2_host;
using clk = std::chrono::steady_clock;

static std::vector<int> read_ids(const std::string& path) {
    std::ifstream f(path);
    std::vector<int> v;
    int x;
    while (f >> x) v.push_back(x);
    return v;
}

static std::vector<int> topk(const std::vector<float>& l, int k) {
    std::vector<int> idx(l.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    idx.resize(k);
    return idx;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.q4nx DIR NAME [--ctx N] [--new N] [--chunk N]\n", argv[0]);
        return 2;
    }
    const std::string base = std::string(argv[2]) + "/" + argv[3];
    int ctx = 0, n_new = -1, chunk = 0;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--ctx")) ctx = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--new")) n_new = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--chunk")) chunk = atoi(argv[i + 1]);
    }
    const std::vector<int> ids = read_ids(base + ".ids");
    const std::vector<int> ref_new = read_ids(base + ".new_ids");
    std::vector<float> ref(VOCAB);
    {
        std::ifstream f(base + ".first_logits", std::ios::binary);
        f.read(reinterpret_cast<char*>(ref.data()), VOCAB * sizeof(float));
        if (!f) {
            fprintf(stderr, "cannot read %s.first_logits\n", base.c_str());
            return 2;
        }
    }
    if (n_new < 0) n_new = (int)ref_new.size();
    if (!ctx) ctx = (int)ids.size() + n_new + 16;

    model m;
    auto t0 = clk::now();
    m.load(argv[1]);
    m.set_max_length(ctx);
    auto t1 = clk::now();
    printf("load + alloc: %.2f s, prompt %zu tokens, ctx %d\n", std::chrono::duration<double>(t1 - t0).count(), ids.size(), ctx);

    std::vector<float> logits(VOCAB);
    t0 = clk::now();
    if (chunk) {
        for (size_t i = 0; i < ids.size(); i += chunk) {
            const int n = std::min<int>(chunk, (int)(ids.size() - i));
            m.run(ids.data() + i, n, i + n == ids.size() ? logits.data() : nullptr);
        }
    } else {
        m.run(ids.data(), (int)ids.size(), logits.data());
    }
    t1 = clk::now();
    const double tp = std::chrono::duration<double>(t1 - t0).count();
    printf("prefill: %.2f s, %.1f tok/s\n", tp, ids.size() / tp);

    // first-step logits
    double max_abs = 0, sum_abs = 0;
    for (int i = 0; i < VOCAB; i++) {
        const double d = std::fabs((double)logits[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
    }
    const auto a = topk(logits, 20), b = topk(ref, 20);
    int overlap = 0;
    for (int x : a) overlap += std::count(b.begin(), b.end(), x);
    // KL(ref || got) over the softmax
    auto lse = [](const std::vector<float>& l) {
        const float mx = *std::max_element(l.begin(), l.end());
        double s = 0;
        for (float v : l) s += std::exp((double)v - mx);
        return mx + std::log(s);
    };
    const double la = lse(logits), lb = lse(ref);
    double kl = 0;
    for (int i = 0; i < VOCAB; i++) {
        const double pb = std::exp(ref[i] - lb);
        kl += pb * ((ref[i] - lb) - (logits[i] - la));
    }
    printf("first logits: max |d| %.4f, mean |d| %.5f, KL %.5f, top20 overlap %d/20, argmax %d (ref %d)\n", max_abs,
           sum_abs / VOCAB, kl, overlap, a[0], b[0]);

    // greedy decode
    std::vector<int> got;
    int next = a[0];
    t0 = clk::now();
    for (int i = 0; i < n_new; i++) {
        got.push_back(next);
        if (i + 1 == n_new) break;
        m.run(&next, 1, logits.data());
        next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    t1 = clk::now();
    const double td = std::chrono::duration<double>(t1 - t0).count();
    printf("decode: %d tokens, %.1f ms/tok\n", (int)got.size() - 1, 1000 * td / std::max(1, (int)got.size() - 1));
    int same = 0;
    while (same < (int)got.size() && same < (int)ref_new.size() && got[same] == ref_new[same]) same++;
    printf("greedy: %d/%zu tokens match the reference%s\n", same, ref_new.size(),
           same == (int)std::min(got.size(), ref_new.size()) ? "" : " (diverged)");
    printf("ids:");
    for (int x : got) printf(" %d", x);
    printf("\n");
    return 0;
}
