// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Inkling2 prefill attention on random inputs against a CPU reference: sink logits, relative-position
// bias R . W[q - k] for q - k < 1024, causal, optional sliding window.
#include <bits/stdc++.h>
#include <boost/program_options.hpp>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;
using namespace inkling2_attn;

static float rnd(std::mt19937& g, float a) { return std::uniform_real_distribution<float>(-a, a)(g); }
static float bfr(float x) { return (float)(bf16)x; }

int main(int argc, char** argv) {
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1024), "query rows");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "first query row (cached prefix length)");
    desc.add_options()("Window,w", po::value<int>()->default_value(0), "sliding window (0: global)");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(20), "Iteration");
    desc.add_options()("NoBias", po::value<bool>()->default_value(false), "zero the rel features (debug)");
    desc.add_options()("NoSink", po::value<bool>()->default_value(false), "sinks at -30 (debug)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int NT = vm["Iteration"].as<int>();
    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();
    const int window = vm["Window"].as<int>();
    const bool no_bias = vm["NoBias"].as<bool>();
    const bool no_sink = vm["NoSink"].as<bool>();
    const int L_end = L_begin + L_wanted;
    const int Lb = L_begin / ROUND * ROUND, Le = (L_end + ROUND - 1) / ROUND * ROUND;
    const int rounds = (Le - Lb) / ROUND;
    const int MAX_L = Le, NKV = 8, DK = NKV * DH;

    std::mt19937 g(vm["Seed"].as<unsigned int>());
    std::vector<float> Q((size_t)L_end * HEADS * DH), R((size_t)L_end * HEADS * NR), S((size_t)L_end * HEADS);
    std::vector<float> K((size_t)L_end * DK), V((size_t)L_end * DK), W((size_t)REL_EXT * NR);
    for (auto& x : Q) x = bfr(rnd(g, 0.6f));
    for (auto& x : K) x = bfr(rnd(g, 1.0f));
    for (auto& x : V) x = bfr(rnd(g, 1.0f));
    if (getenv("ATT_KZERO")) for (auto& x : K) x = 0.0f;
    if (getenv("ATT_QZERO")) for (auto& x : Q) x = 0.0f;
    if (getenv("ATT_VONE")) for (auto& x : V) x = 1.0f;
    for (auto& x : R) x = no_bias ? 0.0f : bfr(rnd(g, 1.0f));
    for (auto& x : W) x = bfr(rnd(g, 0.5f));
    for (auto& x : S) x = no_sink ? -30.0f : bfr(rnd(g, 2.0f));

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin("build/xclbins/attn.xclbin");
    npu_app app = mgr->create_app();

    buffer<bf16> o = app.create_bo_buffer<bf16>((size_t)(Le - Lb) * HEADS * DH);
    buffer<bf16> qp = app.create_bo_buffer<bf16>(q_pack_size(rounds));
    const int KV_CACHE_SIZE = MAX_L * NKV * KV_ROW * 2;
    buffer<bf16> kv = app.create_bo_buffer<bf16>(KV_CACHE_SIZE);
    buffer<bf16> tab = app.create_bo_buffer<bf16>(TAB_SIZE);

    memset(qp.data(), 0, qp.size() * sizeof(bf16));
    for (int t = L_begin; t < L_end; t++)
        for (int h = 0; h < HEADS; h++)
            pack_q(qp.data(), rounds, h, t - Lb, &Q[((size_t)t * HEADS + h) * DH], &R[((size_t)t * HEADS + h) * NR],
                   S[(size_t)t * HEADS + h]);
    // bfp16 blocks: round to nearest even, saturating; a block whose largest exponent is below 6 is zero
    auto pack2 = [](const float* p, uint8_t* b0, uint8_t* b1) {
        for (int b = 0; b < 2; b++) {
            uint8_t* blk = b ? b1 : b0;
            int e = 0;
            for (int i = 0; i < 8; i++) {
                uint32_t u;
                memcpy(&u, &p[b * 8 + i], 4);
                e = std::max(e, (int)((u >> 23) & 0xFF));
            }
            if (e < 6) { memset(blk, 0, 9); continue; }
            blk[0] = (uint8_t)e;
            for (int i = 0; i < 8; i++)
                blk[1 + i] = (uint8_t)(int8_t)std::clamp(std::nearbyint(std::ldexp(p[b * 8 + i], 133 - e)), -128.0f, 127.0f);
        }
    };
    memset(kv.data(), 0, kv.size() * sizeof(bf16));
    for (int t = 0; t < MAX_L; t += 16)
        for (int kh = 0; kh < NKV; kh++) {
            // rows past L_end are zero
            std::vector<float> kk(16 * DH, 0.0f), vv(16 * DH, 0.0f);
            for (int r = 0; r < 16 && t + r < L_end; r++)
                for (int c = 0; c < DH; c++) {
                    kk[r * DH + c] = K[(size_t)(t + r) * DK + kh * DH + c];
                    vv[r * DH + c] = V[(size_t)(t + r) * DK + kh * DH + c];
                }
            uint8_t* kt = (uint8_t*)kv.data() + ((size_t)kh * MAX_L + t) * KV_ROW * 2;
            pack_kv_tile(kt, kt + (size_t)KV_CACHE_SIZE, kk.data(), vv.data(), DH, pack2);
        }
    pack_table(tab.data(), W.data());
    qp.sync_to_device();
    kv.sync_to_device();
    tab.sync_to_device();

    gen_seq(app.seq(), Lb, Le, MAX_L, window, KV_CACHE_SIZE);
    double us = 0;
    for (int i = 0; i < NT; i++) {
        auto t0 = std::chrono::steady_clock::now();
        app(o, qp, kv, tab);
        us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    us /= NT;
    o.sync_from_device();

    // reference
    const float scale = 1.0f / std::sqrt((float)DH);
    double err2 = 0, ref2 = 0, max_err = 0, flops = 0;
    int worst_t = -1, worst_h = -1;
    std::vector<float> sc(L_end);
    for (int t = L_begin; t < L_end; t++)
        for (int h = 0; h < HEADS; h++) {
            const float* q = &Q[((size_t)t * HEADS + h) * DH];
            const float* r = &R[((size_t)t * HEADS + h) * NR];
            const int kh = h / 3;
            const int k0 = window > 0 ? std::max(0, t - window + 1) : 0;
            float m = S[(size_t)t * HEADS + h];
            for (int k = k0; k <= t; k++) {
                const float* kk = &K[(size_t)k * DK + kh * DH];
                float s = 0;
                for (int c = 0; c < DH; c++) s += q[c] * kk[c];
                s *= scale;
                const int d = t - k;
                if (d < REL_EXT)
                    for (int f = 0; f < NR; f++) s += r[f] * W[d * NR + f];
                sc[k] = s;
                m = std::max(m, s);
            }
            float l = std::exp(S[(size_t)t * HEADS + h] - m);
            float y[DH] = {};
            for (int k = k0; k <= t; k++) {
                const float p = std::exp(sc[k] - m);
                l += p;
                const float* vv = &V[(size_t)k * DK + kh * DH];
                for (int c = 0; c < DH; c++) y[c] += p * vv[c];
            }
            flops += 4.0 * DH * (t - k0 + 1);
            const bf16* on = &o[((size_t)(t - Lb) * HEADS + h) * DH];
            double e = 0;
            for (int c = 0; c < DH; c++) {
                const double ref = y[c] / l, dv = (float)on[c] - ref;
                err2 += dv * dv;
                ref2 += ref * ref;
                e = std::max(e, std::abs(dv));
            }
            if (e > max_err) { max_err = e; worst_t = t; worst_h = h; }
            if (getenv("ATT_DUMP") && h == atoi(getenv("ATT_DUMP")) && (getenv("ATT_DUMPALL") || t < 40 || t % 97 == 0)) {
                std::cout << "t " << t << " err " << e << " ref";
                for (int c = 0; c < 4; c++) std::cout << " " << y[c] / l;
                std::cout << " npu";
                for (int c = 0; c < 4; c++) std::cout << " " << (float)on[c];
                std::cout << "\n";
            }
        }
    const double rel = std::sqrt(err2 / ref2);
    std::cout << "L_begin " << L_begin << " L_wanted " << L_wanted << " window " << window << ": NPU " << us
              << " us, " << flops / us * 1e-6 << " TOPS (useful)\n";
    std::cout << "rel L2 err " << rel << ", max abs err " << max_err << " at t " << worst_t << " h " << worst_h << "\n";
    const bool pass = rel < 0.06;  // bfp16 V and P: 1/128 steps of the block max
    std::cout << (pass ? "PASSED" : "FAILED!") << std::endl;
    return pass ? 0 : 1;
}
