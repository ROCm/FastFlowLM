// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// DIAG: per-phase core-timer timestamps of the dequant_mm kernel (build with kernel -DDIAG_TS, host -DDIAG_BUS_EN) and the
// calibration of where a y chunk's elements land in the y matrix (kernel -DDIAG_MAP).
// usage: ts_test <xclbin> map|ts <Din> <Dout> <L> [mapfile] [csv]
#include <bits/stdc++.h>
#include <stdfloat>
#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "lm_config.hpp"
#include "seq_gen.hpp"
#include "tensor_2d.hpp"
#include "cpu_q4_k.hpp"
#include "util.hpp"
#include "weights_desc.hpp"
using clk = std::chrono::steady_clock;
int main(int argc, char** argv){
    std::string xcl = argv[1], mode = argv[2]; int Din = atoi(argv[3]), Dout = atoi(argv[4]), L = atoi(argv[5]);
    std::string mapfile = argc > 6 ? argv[6] : "map.bin", csv = argc > 7 ? argv[7] : "ts.csv";
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm = mgr.register_xclbin(xcl);
    std::mt19937 rng(0);
    size_t wbytes = q4_k_size((size_t)Din * Dout);
    buffer<bf16> w((size_t)Dout * Din), wd((size_t)Dout * Din);
    std::normal_distribution<float> dist(0.f, 0.02f);
    for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
    buffer<u8> wp(wbytes);
    q4_k_quantize_pack(wp.data(), wd.data(), w.data(), Dout, Din);
    npu_app app = mm->create_app();
    gen_dequant_mm_rtm(app.seq(), L, Din, Dout, 0, 0, 0);
    buffer<u8> qw = app.create_bo_buffer<u8>(wbytes);
    buffer<bf16> x = app.create_bo_buffer<bf16>((size_t)L * Din);
    buffer<bf16> y = app.create_bo_buffer<bf16>((size_t)L * Dout);
    reorder_cpy(qw.data(), wp, Din, 2); qw.sync_to_device();
    std::normal_distribution<float> xd(0.f, 1.f);
    for (size_t i = 0; i < x.size(); i++) x[i] = bf16(xd(rng));
    x.sync_to_device(); y.memset(bf16(0.f)); y.sync_to_device();
    int warm = mode == "map" ? 1 : 4;
    for (int i = 0; i < warm; i++) app(y, x, qw);
    std::vector<double> t;
    for (int i = 0; i < 5; i++){ auto a = clk::now(); app(y, x, qw); auto b = clk::now(); t.push_back(std::chrono::duration<double, std::micro>(b - a).count()); }
    std::sort(t.begin(), t.end());
    printf("# dispatch median %.1f us (min %.1f)\n", t[2], t[0]);
    y.sync_from_device();
    const uint16_t* yr = (const uint16_t*)y.data();
    const int M = 128, N = 64, NB = L / 512, NR = Dout / 512;
    auto at = [&](int blk, int rnd, int r, int c, int rr, int cc){ return yr[(size_t)(blk * 512 + r * M + rr) * Dout + rnd * 512 + c * N + cc]; };
    if (mode == "map"){
        // chunk c, element e -> (row, col) inside the core's 128 x 64 tile
        std::vector<int16_t> inv(32 * 256 * 2, -1); int dup = 0, cnt = 0;
        for (int rr = 0; rr < M; rr++) for (int cc = 0; cc < N; cc++){
            unsigned v = at(0, 0, 0, 0, rr, cc);
            if (v >= 8192) continue;
            int ch = v / 256, e = v % 256;
            if (inv[(ch * 256 + e) * 2] >= 0) dup++;
            inv[(ch * 256 + e) * 2] = rr; inv[(ch * 256 + e) * 2 + 1] = cc; cnt++;
        }
        printf("# mapped %d positions, dup %d\n", cnt, dup);
        FILE* f = fopen(mapfile.c_str(), "wb"); fwrite(inv.data(), 2, inv.size(), f); fclose(f);
        // check another core / round gives the same chunk structure
        int bad = 0; for (int r = 0; r < 4; r++) for (int c = 0; c < 8; c++) for (int rr = 0; rr < M; rr++) for (int cc = 0; cc < N; cc++){
            unsigned v = at(NB - 1, NR - 1, r, c, rr, cc); int ch = v / 256, e = v % 256;
            if (v >= 8192 || inv[(ch * 256 + e) * 2] != rr || inv[(ch * 256 + e) * 2 + 1] != cc) bad++;
        }
        printf("# map consistency mismatches (last round, all cores): %d\n", bad);
        return 0;
    }
    std::vector<int16_t> inv(32 * 256 * 2); { FILE* f = fopen(mapfile.c_str(), "rb"); if (!f || fread(inv.data(), 2, inv.size(), f) != inv.size()) { printf("no map\n"); return 1; } fclose(f); }
    FILE* o = fopen(csv.c_str(), "w");
    for (int blk = 0; blk < NB; blk++) for (int rnd = 0; rnd < NR; rnd++) for (int r = 0; r < 4; r++) for (int c = 0; c < 8; c++){
        auto el = [&](int e){ int ie = (31 * 256 + e) * 2; return (unsigned)at(blk, rnd, r, c, inv[ie], inv[ie + 1]); };
        unsigned base = el(0) | (el(1) << 16), n = el(2);
        if (n < 2 || n > 224) { fprintf(o, "%d,%d,%d,%d,BAD,%u\n", blk, rnd, r, c, n); continue; }
        fprintf(o, "%d,%d,%d,%d,%u,%u", blk, rnd, r, c, base, n);
        for (unsigned q = 1; q < n; q++) fprintf(o, ",%u", el(2 + q));
        fprintf(o, "\n");
    }
    fclose(o);
    printf("# wrote %s\n", csv.c_str());
}
