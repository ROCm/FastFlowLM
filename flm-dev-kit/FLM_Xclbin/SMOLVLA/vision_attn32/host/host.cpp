// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Standalone test of the 32-core vision attention (all 8 cols x 4 rows): random Q/K/V, fp32 CPU reference.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "tensor_2d.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("L,l", po::value<int>()->default_value(1024), "tokens per image (multiple of 512)");
    desc.add_options()("Images,n", po::value<int>()->default_value(3), "images per call");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(50), "Iteration");
    desc.add_options()("Xclbin,x", po::value<std::string>()->default_value("build/0/xclbins/attn.xclbin"), "xclbin");
    desc.add_options()("Cus,c", po::value<int>()->default_value(4), "4 = 32 cores, 2 = 16-core engine layout (use with the vision xclbin)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int L = vm["L"].as<int>();
    const int NIMG = vm["Images"].as<int>();
    const int NT = vm["Iteration"].as<int>();
    const int DH = 64, H = 12, D = DH * H;
    const size_t VOL = (size_t)NIMG * L * D;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> qf(VOL), kf(VOL), vf(VOL);
    for (auto* v : {&qf, &kf, &vf}) for (auto& x : *v) x = nd(rng);

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin(vm["Xclbin"].as<std::string>());
    npu_app app = mgr->create_app();
    buffer<bf16> o = app.create_bo_buffer<bf16>(VOL);
    buffer<bf16> q = app.create_bo_buffer<bf16>(VOL);
    buffer<bf16> k = app.create_bo_buffer<bf16>(VOL);
    buffer<bf16> v = app.create_bo_buffer<bf16>(VOL);
    for (size_t i = 0; i < VOL; i++) { q.data()[i] = (bf16)qf[i]; k.data()[i] = (bf16)kf[i]; v.data()[i] = (bf16)vf[i]; }
    q.sync_to_device(); k.sync_to_device(); v.sync_to_device();
    o.memset((bf16)(-1.0f));
    o.sync_to_device();

    gen_attn32_seq(app.seq(), L, DH, H, NIMG, vm["Cus"].as<int>());

    double tot = 0, best = 1e30;
    for (int i = 0; i < NT; i++) {
        auto t0 = time_utils::now();
        app(o, q, k, v);
        auto t1 = time_utils::now();
        double us = time_utils::duration_us(t0, t1).first;
        tot += us; best = std::min(best, us);
    }
    const double avg = tot / NT;
    // QK^T + PV: 2 * 2 * L * L * D per image
    const double flop = 4.0 * (double)L * L * D * NIMG;
    std::cout << "[attn32] L=" << L << " images=" << NIMG << " avg " << avg << " us  best " << best << " us  "
              << flop / (avg * 1e-6) / 1e12 << " TOPS (avg)\n";
    o.sync_from_device();

    // fp32 reference (bf16-rounded inputs), all images, all heads
    std::vector<float> ref((size_t)L * DH);
    double worst_cos = 1.0, sum_cos = 0; int cnt = 0;
    double max_abs = 0;
    for (int img = 0; img < NIMG; img++) {
        for (int h = 0; h < H; h++) {
            if (img != 0 && h % 4 != img % 4) continue; // sample heads on the later images
            const size_t base = (size_t)img * L * D + (size_t)h * DH;
            std::vector<float> Kb((size_t)L * DH), Vb((size_t)L * DH), Qb((size_t)L * DH);
            for (int r = 0; r < L; r++) for (int c = 0; c < DH; c++) {
                Kb[(size_t)r * DH + c] = (float)k.data()[base + (size_t)r * D + c];
                Vb[(size_t)r * DH + c] = (float)v.data()[base + (size_t)r * D + c];
                Qb[(size_t)r * DH + c] = (float)q.data()[base + (size_t)r * D + c];
            }
#pragma omp parallel for
            for (int r = 0; r < L; r++) {
                std::vector<float> s(L);
                float m = -1e30f;
                for (int j = 0; j < L; j++) {
                    float a = 0;
                    for (int c = 0; c < DH; c++) a += Qb[(size_t)r * DH + c] * Kb[(size_t)j * DH + c];
                    s[j] = a * 0.125f; m = std::max(m, s[j]);
                }
                float den = 0;
                for (int j = 0; j < L; j++) { s[j] = std::exp(s[j] - m); den += s[j]; }
                for (int c = 0; c < DH; c++) {
                    float a = 0;
                    for (int j = 0; j < L; j++) a += s[j] * Vb[(size_t)j * DH + c];
                    ref[(size_t)r * DH + c] = a / den;
                }
            }
            double dot = 0, na = 0, nb = 0;
            for (int r = 0; r < L; r++) for (int c = 0; c < DH; c++) {
                double a = ref[(size_t)r * DH + c], b = (float)o.data()[base + (size_t)r * D + c];
                dot += a * b; na += a * a; nb += b * b;
                max_abs = std::max(max_abs, std::fabs(a - b));
            }
            double cs = dot / std::sqrt(na * nb + 1e-30);
            worst_cos = std::min(worst_cos, cs); sum_cos += cs; cnt++;
        }
    }
    std::cout << "[attn32] cos vs fp32 ref: mean " << sum_cos / cnt << " worst " << worst_cos << " max|d| " << max_abs
              << " (" << cnt << " head slices)\n";
    std::cout << (worst_cos > 0.999 ? "PASSED" : "FAILED") << "\n";
    return 0;
}
