// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// S5.2 test rig for the host DFlash drafter: reads the cases written by tools/ref/draft_case.py, runs inject()+draft(),
// writes the pre-norm hidden rows (bf16 [7][2560]) for the python checker, and measures timings.
//   usage: test_dflash_draft <pkg_dir> <case_dir> [timing_reps=20]
#include "../dflash_draft.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <pkg_dir> <case_dir> [reps]\n", argv[0]); return 2; }
    const std::string pkg = argv[1], cdir = argv[2];
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    dflash_draft_host d;
    auto t0 = clk::now();
    if (!d.load(pkg)) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::printf("load: %.0f ms\n", ms_since(t0));
    const int D = d.cfg().D;

    std::vector<int> ctxs;
    if (DIR* dir = opendir(cdir.c_str())) {
        while (dirent* e = readdir(dir)) {
            std::string n = e->d_name;
            if (n.rfind("case_", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".bin") ctxs.push_back(std::atoi(n.c_str() + 5));
        }
        closedir(dir);
    }
    std::sort(ctxs.begin(), ctxs.end());
    for (int ctx_id : ctxs) {
        std::ifstream f(cdir + "/case_" + std::to_string(ctx_id) + ".bin", std::ios::binary);
        int hdr[3];
        f.read((char*)hdr, sizeof(hdr));
        const int ctx = hdr[0], nch = hdr[2];
        std::vector<int> ch(nch);
        f.read((char*)ch.data(), nch * sizeof(int));
        std::vector<bf16> taps((size_t)ctx * 5 * D), emb((size_t)8 * D);
        f.read((char*)taps.data(), taps.size() * sizeof(bf16));
        f.read((char*)emb.data(), emb.size() * sizeof(bf16));

        d.reset();
        t0 = clk::now();
        int p = 0;
        for (int c : ch) { d.inject(taps.data() + (size_t)p * 5 * D, c, p); p += c; }
        const double inj_ms = ms_since(t0);
        std::vector<bf16> hid((size_t)7 * D);
        t0 = clk::now();
        d.draft(ctx, emb.data(), hid.data());
        const double first_ms = ms_since(t0);
        std::ofstream o(cdir + "/case_" + std::to_string(ctx) + ".out.bf16", std::ios::binary);
        o.write((const char*)hid.data(), hid.size() * sizeof(bf16));
        o.close();

        // timing: draft repeated at the same state; inject of 4 rows (a typical accepted block) repeated at the end of the ctx
        double best = 1e30, sum = 0;
        for (int i = 0; i < reps; i++) {
            t0 = clk::now();
            d.draft(ctx, emb.data(), hid.data());
            const double t = ms_since(t0);
            best = std::min(best, t); sum += t;
        }
        std::vector<bf16> tp((size_t)4 * 5 * D);
        for (size_t i = 0; i < tp.size(); i++) tp[i] = taps.empty() ? emb[i % emb.size()] : taps[i % taps.size()];
        double inj4 = 0;
        int pos = d.ctx_end();
        for (int i = 0; i < reps; i++) {
            t0 = clk::now();
            d.inject(tp.data(), 4, pos);
            inj4 += ms_since(t0);
            pos += 4;
        }
        std::printf("ctx=%5d  bulk inject %.1f ms (%.3f ms/row)  first draft %.2f ms  draft avg %.2f best %.2f ms  inject(4 rows) %.3f ms (%.3f ms/row)\n",
                    ctx, inj_ms, ctx ? inj_ms / ctx : 0.0, first_ms, sum / reps, best, inj4 / reps, inj4 / reps / 4);
        std::fflush(stdout);
    }
    return 0;
}
