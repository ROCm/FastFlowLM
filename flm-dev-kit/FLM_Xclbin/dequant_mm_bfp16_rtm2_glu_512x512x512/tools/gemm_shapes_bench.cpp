// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Per-shape timing + accuracy of a dequant_mm xclbin for the MiniCPM-V-4.7-1B prefill projection shapes
// (vs the fp32 product of the dequantised weights; L = 512 only, 128 sampled rows).
// usage: gemm_shapes_bench <xclbin> [iters] [shape-filter-substring] [noacc]
// env:   DUMP_PREFIX=<path>  write every output to <path>_<Din>x<Dout>_L<L>.bin (bitwise comparison of two xclbins:
//                            same filter and iteration count give the same inputs)
//        SOAK=<n>            n extra dispatches per shape/L, outputs compared with the first (races, hangs)
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
    std::string xcl = argv[1]; int N = argc > 2 ? atoi(argv[2]) : 60; std::string filt = argc > 3 ? argv[3] : ""; bool noacc = argc > 4 && std::string(argv[4]) == "noacc";
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm = mgr.register_xclbin(xcl);
    struct Shape { const char* name; int Din, Dout; int act; };
    std::vector<Shape> shapes = {
        {"gdn_qkv 1024->6144", 1024, 6144, 0}, {"gdn_z(silu) 1024->2048", 1024, 2048, 1}, {"gelu 1024->2048", 1024, 2048, 2},
        {"attn_o 2048->1024", 2048, 1024, 0}, {"attn_q 1024->2048", 1024, 2048, 0},
        {"attn_k 1024->512", 1024, 512, 0}, {"upgate 1024->7168", 1024, 7168, 0}, {"down 3584->1024", 3584, 1024, 0},
    };
    std::vector<int> Ls = {512, 1024, 2048};
    printf("shape,L,GFLOP,med_us,min_us,TFLOPS,relrms,maxabs_over_rms,bad_frac\n");
    std::mt19937 rng(0);
    for (auto& s : shapes) {
        if (!filt.empty() && std::string(s.name).find(filt) == std::string::npos) continue;
        size_t wbytes = q4_k_size((size_t)s.Din * s.Dout);
        buffer<bf16> w((size_t)s.Dout * s.Din), wd((size_t)s.Dout * s.Din);
        std::normal_distribution<float> dist(0.f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        buffer<u8> wp(wbytes);
        q4_k_quantize_pack(wp.data(), wd.data(), w.data(), s.Dout, s.Din);
        for (int L : Ls) {
            npu_app app = mm->create_app();
            gen_dequant_mm_512(app.seq(), L, s.Din, s.Dout, 0, 0, s.act);
            buffer<u8> qw = app.create_bo_buffer<u8>(wbytes);
            buffer<bf16> x = app.create_bo_buffer<bf16>((size_t)L * s.Din);
            buffer<bf16> y = app.create_bo_buffer<bf16>((size_t)L * s.Dout);
            reorder_cpy(qw.data(), wp, s.Din, 2); qw.sync_to_device();
            std::normal_distribution<float> xd(0.f, 1.f);
            for (size_t i = 0; i < x.size(); i++) x[i] = bf16(xd(rng));
            x.sync_to_device(); y.memset(bf16(0.f)); y.sync_to_device();
            for (int i = 0; i < 5; i++) app(y, x, qw);
            if (const char* so = getenv("SOAK")) {
                int reps = atoi(so); y.sync_from_device();
                std::vector<bf16> first(y.data(), y.data() + (size_t)L * s.Dout); size_t bad = 0;
                for (int r = 0; r < reps; r++) { app(y, x, qw); if (r % 5 == 0) { y.sync_from_device(); if (memcmp(first.data(), y.data(), first.size() * sizeof(bf16))) bad++; } }
                printf("# soak %s L=%d reps=%d mismatching_checks=%zu\n", s.name, L, reps, bad);
            }
            std::vector<double> t;
            for (int i = 0; i < N; i++){ auto a = clk::now(); app(y, x, qw); auto b = clk::now(); t.push_back(us(a, b)); }
            std::sort(t.begin(), t.end());
            double med = t[t.size()/2], mn = t.front();
            if (const char* dp = getenv("DUMP_PREFIX")) {
                y.sync_from_device();
                std::string fn = std::string(dp) + "_" + std::to_string(s.Din) + "x" + std::to_string(s.Dout) + "_L" + std::to_string(L) + ".bin";
                FILE* f = fopen(fn.c_str(), "wb"); fwrite(y.data(), sizeof(bf16), (size_t)L * s.Dout, f); fclose(f);
            }
            double gf = 2.0 * L * s.Din * s.Dout / 1e9;
            // accuracy on L==512 only, rows subsampled stride 4 (activation applied to ref if silu)
            double relrms = -1, mx = -1, bad = -1;
            if (L == 512 && !noacc) {
                y.sync_from_device();
                int rows = 128; double se = 0, sr = 0, maxe = 0; size_t nbad = 0, tot = 0;
                std::vector<double> ref(s.Dout);
                for (int r = 0; r < rows; r++) {
                    int l = r * 4;
                    #pragma omp parallel for
                    for (int n = 0; n < s.Dout; n++) {
                        double acc = 0;
                        for (int k = 0; k < s.Din; k++) acc += (double)(float)x[(size_t)l*s.Din+k] * (double)(float)wd[(size_t)n*s.Din+k];
                        if (s.act == 1) acc = acc / (1.0 + std::exp(-acc)); else if (s.act == 2) acc = 0.5 * acc * (1.0 + std::tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)));
                        ref[n] = acc;
                    }
                    for (int n = 0; n < s.Dout; n++) {
                        double yv = (float)y[(size_t)l*s.Dout+n]; double e = yv - ref[n];
                        se += e*e; sr += ref[n]*ref[n]; maxe = std::max(maxe, std::fabs(e)); tot++;
                        if (std::fabs(e) > 0.05*std::fabs(ref[n]) + 0.05) nbad++;
                    }
                }
                relrms = std::sqrt(se/sr); mx = maxe / std::sqrt(sr/tot); bad = (double)nbad/tot;
            }
            printf("%s,%d,%.2f,%.1f,%.1f,%.2f,%.5f,%.3f,%.5f\n", s.name, L, gf, med, mn, gf/med*1e3, relrms, mx, bad);
            fflush(stdout);
        }
    }
}
