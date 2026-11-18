// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Kernel-only timing of the runtime-m sequence vs the fixed 512-row sequence, per projection shape and row count.
//   rtm_bench <xclbin> <iters> rtm|fixed [shape-filter]      RTM_ROWS=a,b,c overrides the row counts
// fixed: gen_dequant_mm_512 over ceil(rows / 512) x 512 rows (what the fixed kernel does for `rows` real rows)
// rtm:   gen_dequant_mm_rtm over `rows` (covers ceil(rows / 64) x 64 rows)
// Output (CSV): shape,rows,covered,med_us,min_us,sum of the 132-dispatch mix (see tools in the report) is left to the caller.
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
    if (argc < 4) { fprintf(stderr, "usage: rtm_bench <xclbin> <iters> rtm|fixed [filter]\n"); return 2; }
    const std::string xcl = argv[1]; const int N = atoi(argv[2]); const bool rtm = std::string(argv[3]) == "rtm";
    const std::string filt = argc > 4 ? argv[4] : "";
    std::vector<int> rowsets = {8, 24, 64, 72, 128, 200, 264, 400, 512, 520, 700, 1032, 1536, 2056};
    if (const char* rs = getenv("RTM_ROWS")) { rowsets.clear(); std::stringstream ss(rs); std::string t; while (std::getline(ss, t, ',')) rowsets.push_back(atoi(t.c_str())); }
    xrt::device dev(0);
    npu_xclbin_manager mgr(npu_device::device_npu2, &dev, false);
    npu_app_manager* mm = mgr.register_xclbin(xcl);
    struct Shape { const char* name; int Din, Dout; int act; };
    std::vector<Shape> shapes = {
        {"qkv", 1024, 6144, 0}, {"z_silu", 1024, 2048, 1}, {"attn_o", 2048, 1024, 0}, {"attn_q", 1024, 2048, 0},
        {"attn_k", 1024, 512, 0}, {"upgate", 1024, 7168, 0}, {"down", 3584, 1024, 0},
    };
    const int LMAX = 2560;
    printf("shape,rows,covered,med_us,min_us\n");
    std::mt19937 rng(7);
    for (auto& s : shapes) {
        if (!filt.empty() && std::string(s.name).find(filt) == std::string::npos) continue;
        size_t wbytes = q4_k_size((size_t)s.Din * s.Dout);
        buffer<bf16> w((size_t)s.Dout * s.Din), wd((size_t)s.Dout * s.Din);
        std::normal_distribution<float> dist(0.f, 0.02f);
        for (size_t i = 0; i < w.size(); i++) w[i] = bf16(dist(rng));
        buffer<u8> wp(wbytes);
        q4_k_quantize_pack(wp.data(), wd.data(), w.data(), s.Dout, s.Din);
        npu_app app0 = mm->create_app();
        buffer<u8> qw = app0.create_bo_buffer<u8>(wbytes);
        reorder_cpy(qw.data(), wp, s.Din, 2); qw.sync_to_device();
        buffer<bf16> x = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Din);
        buffer<bf16> y = app0.create_bo_buffer<bf16>((size_t)LMAX * s.Dout);
        std::normal_distribution<float> xd(0.f, 1.f);
        for (size_t i = 0; i < x.size(); i++) x[i] = bf16(xd(rng));
        x.sync_to_device(); y.memset(bf16(0.f)); y.sync_to_device();
        for (int rows : rowsets) {
            npu_app app = mm->create_app();
            uint32_t covered;
            if (rtm) { gen_dequant_mm_rtm(app.seq(), rows, s.Din, s.Dout, 0, 0, s.act); covered = rtm_pick_geometry(rows).covered; }
            else { covered = (rows + 511) / 512 * 512; gen_dequant_mm_512(app.seq(), covered, s.Din, s.Dout, 0, 0, s.act); }
            for (int i = 0; i < 8; i++) app(y, x, qw);
            std::vector<double> t;
            for (int i = 0; i < N; i++){ auto a = clk::now(); app(y, x, qw); auto b = clk::now(); t.push_back(us(a, b)); }
            std::sort(t.begin(), t.end());
            printf("%s,%d,%u,%.1f,%.1f\n", s.name, rows, covered, t[t.size()/2], t.front());
            fflush(stdout);
        }
    }
    return 0;
}
