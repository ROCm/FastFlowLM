// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Runtime-m kernel test: bit-exactness of every row count against the 512-row kernel, plus ordering/soak.
//   rtm_test ref <xclbin-512row> <out-dir>     run the shipped fixed-m kernel at 2048 rows, dump y per shape
//   rtm_test rtm <xclbin-rtm>    <out-dir> [reps] [shape-filter]
//                               gen_dequant_mm_rtm at many row counts, random order, compare to the dump
// Inputs are seeded by shape, so both modes see the same x / w.
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
static double us(clk::time_point a, clk::time_point b){ return std::chrono::duration<double, std::micro>(b - a).count(); }

int main(int argc, char** argv){
    if (argc < 4) { fprintf(stderr, "usage: rtm_test ref|rtm <xclbin> <dir> [reps] [filter]\n"); return 2; }
    const std::string mode = argv[1], xcl = argv[2], dir = argv[3];
    const int reps = argc > 4 ? atoi(argv[4]) : 3;
    const std::string filt = argc > 5 ? argv[5] : "";
    const int LMAX = 2048;
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm = mgr.register_xclbin(xcl);
    struct Shape { const char* name; int Din, Dout; int act; };
    std::vector<Shape> shapes = {
        {"qkv", 1024, 6144, 0}, {"z_silu", 1024, 2048, 1}, {"gelu", 1024, 2048, 2},
        {"attn_o", 2048, 1024, 0}, {"attn_q", 1024, 2048, 0},
        {"attn_k", 1024, 512, 0}, {"upgate", 1024, 7168, 0}, {"down", 3584, 1024, 0},
    };
    std::vector<int> rowsets = {1, 8, 16, 24, 63, 64, 65, 72, 128, 129, 200, 264, 408, 512, 513, 520, 640, 1032, 1536, 1600, 2048};
    if (const char* rs = getenv("RTM_ROWS")) { rowsets.clear(); std::stringstream ss(rs); std::string t; while (std::getline(ss, t, ',')) rowsets.push_back(atoi(t.c_str())); }
    const bool no_shuffle = getenv("RTM_NOSHUFFLE") != nullptr;
    int total_bad = 0;
    for (auto& s : shapes) {
        if (!filt.empty() && std::string(s.name).find(filt) == std::string::npos) continue;
        std::mt19937 rng(1234 + s.Din + s.Dout + s.act);
        size_t wbytes = q4_k_size((size_t)s.Din * s.Dout);
        buffer<bf16> w((size_t)s.Dout * s.Din), wd((size_t)s.Dout * s.Din);
        std::normal_distribution<float> dist(0.f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        buffer<u8> wp(wbytes);
        q4_k_quantize_pack(wp.data(), wd.data(), w.data(), s.Dout, s.Din);
        std::vector<bf16> xh((size_t)LMAX * s.Din);
        std::normal_distribution<float> xd(0.f, 1.f);
        for (auto& v : xh) v = bf16(xd(rng));
        const std::string fn = dir + "/" + s.name + ".bin";
        npu_app app0 = mm->create_app();
        buffer<u8> qw = app0.create_bo_buffer<u8>(wbytes);
        reorder_cpy(qw.data(), wp, s.Din, 2); qw.sync_to_device();
        if (mode == "ref") {
            gen_dequant_mm_512(app0.seq(), LMAX, s.Din, s.Dout, 0, 0, s.act);
            buffer<bf16> x = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Din);
            buffer<bf16> y = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Dout);
            memcpy(x.data(), xh.data(), xh.size() * sizeof(bf16)); x.sync_to_device();
            y.memset(bf16(0.f)); y.sync_to_device();
            app0(y, x, qw); app0(y, x, qw);
            y.sync_from_device();
            FILE* f = fopen(fn.c_str(), "wb"); fwrite(y.data(), sizeof(bf16), (size_t)LMAX * s.Dout, f); fclose(f);
            printf("ref %s done\n", s.name); fflush(stdout);
            continue;
        }
        std::vector<bf16> ref((size_t)LMAX * s.Dout);
        { FILE* f = fopen(fn.c_str(), "rb"); if (!f || fread(ref.data(), sizeof(bf16), ref.size(), f) != ref.size()) { fprintf(stderr, "missing ref %s\n", fn.c_str()); return 2; } fclose(f); }
        buffer<bf16> x = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Din);
        buffer<bf16> y = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Dout);
        memcpy(x.data(), xh.data(), xh.size() * sizeof(bf16)); x.sync_to_device();
        std::vector<std::unique_ptr<npu_app>> apps;
        for (int rows : rowsets) {
            apps.push_back(std::make_unique<npu_app>(mm->create_app()));
            gen_dequant_mm_rtm(apps.back()->seq(), rows, s.Din, s.Dout, 0, 0, s.act);
        }
        std::vector<int> order; for (size_t i = 0; i < rowsets.size(); i++) order.push_back((int)i);
        int bad = 0; double tsum[64] = {0}; int tcnt[64] = {0};
        const bf16 sentinel = bf16(12345.f);
        for (int r = 0; r < reps; r++) {
            if (r > 0 && !no_shuffle) std::shuffle(order.begin(), order.end(), rng);
            for (int oi : order) {
                const int rows = rowsets[oi];
                const rtm_geometry g = rtm_pick_geometry(rows);
                for (size_t i = 0; i < y.size(); i++) y[i] = sentinel;
                y.sync_to_device();
                if (getenv("RTM_VERBOSE")) { fprintf(stderr, "  %s rows=%d (m=%u blocks=%u) rep %d ...", s.name, rows, g.m, g.blocks, r); fflush(stderr); }
                auto a = clk::now();
                (*apps[oi])(y, x, qw);
                if (getenv("RTM_VERBOSE")) { fprintf(stderr, " done\n"); }
                auto b = clk::now();
                tsum[oi] += us(a, b); tcnt[oi]++;
                y.sync_from_device();
                size_t nb = 0, nover = 0;
                for (int row = 0; row < (int)g.covered && row < LMAX; row++) {
                    if (row < rows) { if (memcmp(&y[(size_t)row * s.Dout], &ref[(size_t)row * s.Dout], s.Dout * sizeof(bf16))) nb++; }
                }
                for (size_t row = g.covered; row < (size_t)LMAX; row++) {
                    const bf16* p = &y[row * s.Dout];
                    for (int n = 0; n < s.Dout; n++) if (!(p[n] == sentinel)) { nover++; break; }
                }
                if (getenv("RTM_VERBOSE")) fprintf(stderr, "    -> %zu bad rows, %zu overrun rows\n", nb, nover);
                if (nb && getenv("RTM_DUMP")) {
                    for (int row : {0, 1, 15, 16, 17, rows - 1}) { if (row >= rows) continue; fprintf(stderr, "    row %d got:", row); for (int n = 0; n < 8; n++) fprintf(stderr, " %g", (float)y[(size_t)row * s.Dout + n]); fprintf(stderr, "\n    row %d ref:", row); for (int n = 0; n < 8; n++) fprintf(stderr, " %g", (float)ref[(size_t)row * s.Dout + n]); fprintf(stderr, "\n"); }
                    size_t nz = 0, ns = 0; for (int row = 0; row < rows; row++) for (int n = 0; n < s.Dout; n++) { float v = (float)y[(size_t)row*s.Dout+n]; nz += v == 0.f; ns += v == 12345.f; } fprintf(stderr, "    zeros=%zu sentinels=%zu of %d\n", nz, ns, rows * s.Dout);
                }
                if (nb || nover) { printf("BAD %s rows=%d rep=%d: %zu real rows differ, %zu rows past covered=%u overwritten\n", s.name, rows, r, nb, nover, g.covered); bad++; }
            }
        }
        printf("%s: %s (%d checks)", s.name, bad ? "FAIL" : "ok", reps * (int)rowsets.size());
        for (size_t i = 0; i < rowsets.size(); i++) if (rowsets[i] == 24 || rowsets[i] == 264 || rowsets[i] == 520 || rowsets[i] == 1032 || rowsets[i] == 2048) printf("  t(%d)=%.0fus", rowsets[i], tsum[i] / tcnt[i]);
        printf("\n"); fflush(stdout);
        total_bad += bad;
    }
    printf(total_bad ? "RESULT FAIL\n" : "RESULT PASS\n");
    return total_bad != 0;
}
