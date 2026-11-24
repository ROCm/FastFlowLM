// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Interleaved A/B of two dequant_mm xclbins (same host sequence): blocks of `iters` dispatches alternate between the two
// (context switch only at block boundaries), `rounds` rounds, median of the per-round medians.
//   ab_bench <xclbinA> <xclbinB> <iters> <rounds> [shape-filter] [L list "512,1024,2048"]
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
    if (argc < 5) { fprintf(stderr, "usage\n"); return 2; }
    const int N = atoi(argv[3]), R = atoi(argv[4]); const std::string filt = argc > 5 ? argv[5] : "";
    std::vector<int> Ls = {512, 1024, 2048};
    if (argc > 6) { Ls.clear(); std::stringstream ss(argv[6]); std::string t; while (std::getline(ss, t, ',')) Ls.push_back(atoi(t.c_str())); }
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm[2] = { mgr.register_xclbin(argv[1]), mgr.register_xclbin(argv[2]) };
    struct Shape { const char* name; int Din, Dout; int act; };
    std::vector<Shape> shapes = {
        {"qkv", 1024, 6144, 0}, {"z_silu", 1024, 2048, 1}, {"attn_o", 2048, 1024, 0}, {"attn_q", 1024, 2048, 0},
        {"attn_k", 1024, 512, 0}, {"upgate", 1024, 7168, 0}, {"down", 3584, 1024, 0},
    };
    std::mt19937 rng(0);
    printf("shape,L,A_med_us,B_med_us,ratio_A_over_B,A_min,B_min\n");
    for (auto& s : shapes) {
        if (!filt.empty() && std::string(s.name).find(filt) == std::string::npos) continue;
        size_t wbytes = q4_k_size((size_t)s.Din * s.Dout);
        buffer<bf16> w((size_t)s.Dout * s.Din), wd((size_t)s.Dout * s.Din);
        std::normal_distribution<float> dist(0.f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        buffer<u8> wp(wbytes);
        q4_k_quantize_pack(wp.data(), wd.data(), w.data(), s.Dout, s.Din);
        for (int L : Ls) {
            std::normal_distribution<float> xd(0.f, 1.f);
            std::vector<bf16> xh((size_t)L * s.Din); for (auto& v : xh) v = bf16(xd(rng));
            struct Side { npu_app app; buffer<u8> qw; buffer<bf16> x, y; };
            std::vector<std::unique_ptr<Side>> side;
            for (int k = 0; k < 2; k++) {
                npu_app app = mm[k]->create_app();
                gen_dequant_mm_512(app.seq(), L, s.Din, s.Dout, 0, 0, s.act);
                auto sd = std::make_unique<Side>(Side{std::move(app), buffer<u8>(), buffer<bf16>(), buffer<bf16>()});
                sd->qw = sd->app.create_bo_buffer<u8>(wbytes); sd->x = sd->app.create_bo_buffer<bf16>((size_t)L * s.Din); sd->y = sd->app.create_bo_buffer<bf16>((size_t)L * s.Dout);
                reorder_cpy(sd->qw.data(), wp, s.Din, 2); sd->qw.sync_to_device();
                memcpy(sd->x.data(), xh.data(), xh.size() * sizeof(bf16)); sd->x.sync_to_device();
                sd->y.memset(bf16(0.f)); sd->y.sync_to_device();
                side.push_back(std::move(sd));
            }
            std::vector<double> med[2], mn[2];
            for (int r = 0; r < R; r++) for (int k0 = 0; k0 < 2; k0++) {
                int k = (r % 2) ? 1 - k0 : k0;   // alternate which side goes first
                auto& sd = *side[k];
                for (int i = 0; i < 3; i++) sd.app(sd.y, sd.x, sd.qw);
                std::vector<double> t;
                for (int i = 0; i < N; i++) { auto a = clk::now(); sd.app(sd.y, sd.x, sd.qw); auto b = clk::now(); t.push_back(us(a, b)); }
                std::sort(t.begin(), t.end()); med[k].push_back(t[t.size() / 2]); mn[k].push_back(t.front());
            }
            auto m = [](std::vector<double> v){ std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            double a = m(med[0]), b = m(med[1]);
            printf("%s,%d,%.1f,%.1f,%.3f,%.1f,%.1f\n", s.name, L, a, b, a / b, *std::min_element(mn[0].begin(), mn[0].end()), *std::min_element(mn[1].begin(), mn[1].end()));
            fflush(stdout);
        }
    }
}
