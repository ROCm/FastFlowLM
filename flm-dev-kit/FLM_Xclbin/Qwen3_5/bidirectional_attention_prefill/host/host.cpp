// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Standalone test of the bidirectional prefill attention (d3-lite's full-attention layers):
// random q / k / v at Qwen3.5-0.8B's shapes, every query against keys [0, L_valid), checked against
// an fp64 CPU reference. Rows past L_valid hold a large K (an unmasked padding key would take the
// softmax) and a constant V. Keep padding V small: the array's bf16 MMUL is emulated with bfp16,
// whose blocks share an exponent along the reduction dimension, so in P * V a huge padding V row
// costs the real V rows of its block their precision even at weight 0 (V = 1000: |err| ~8).
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("L_valid,l", po::value<std::vector<int>>()->multitoken()->default_value({1, 100, 128, 129, 1000, 2047, 4096}, "1 100 128 129 1000 2047 4096"), "prompt lengths");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(3), "timed runs per length");
    desc.add_options()("xclbin,x", po::value<std::string>()->default_value("build/xclbins/attn.xclbin"), "xclbin");
    desc.add_options()("pad_k", po::value<float>()->default_value(30.0f), "value of padding K rows");
    desc.add_options()("pad_v", po::value<float>()->default_value(1.5f), "value of padding V rows");
    desc.add_options()("all_rows", po::value<bool>()->default_value(false), "check every row, print the worst");
    desc.add_options()("tolerance,t", po::value<double>()->default_value(0.15), "max |npu - ref| allowed (bf16 scores and exp, bfp16 MMUL: <= 0.1 on these inputs)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int DH = 256, Heads = 8, NKV = 2, DQ = DH * Heads, DK = DH * NKV;
    const double tol = vm["tolerance"].as<double>();
    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin(vm["xclbin"].as<std::string>());
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    bool pass = true;

    for (int L_valid : vm["L_valid"].as<std::vector<int>>()) {
        const int L_padded = (L_valid + 511) / 512 * 512;   // the engine's row bucket
        npu_app app = mgr->create_app();
        buffer<bf16> o = app.create_bo_buffer<bf16>((size_t)L_padded * DQ);
        buffer<bf16> q = app.create_bo_buffer<bf16>((size_t)L_padded * DQ);
        buffer<bf16> kv = app.create_bo_buffer<bf16>((size_t)L_padded * DK * 2);
        for (size_t i = 0; i < q.size(); i++) q[i] = bf16(U(rng) * 2.0f);
        // real keys/values, then padding past L_valid
        for (size_t r = 0; r < (size_t)L_padded; r++)
            for (int c = 0; c < DK; c++) {
                const bool pad = r >= (size_t)L_valid;
                kv[r * DK + c] = bf16(pad ? vm["pad_k"].as<float>() : U(rng) * 2.0f);
                kv[(size_t)L_padded * DK + r * DK + c] = bf16(pad ? vm["pad_v"].as<float>() : U(rng) * 2.0f);
            }
        q.sync_to_device();
        kv.sync_to_device();
        gen_seq_bidirectional(app.seq(), L_valid, 16, DH, Heads, NKV, L_padded * DK * 2);

        std::vector<double> ms;
        std::vector<bf16> first;
        bool deterministic = true;
        for (int it = 0; it < vm["Iteration"].as<int>() + 1; it++) {
            auto t0 = std::chrono::steady_clock::now();
            app(o, q, kv);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            o.sync_from_device();
            if (it == 0) first.assign(o.data(), o.data() + (size_t)L_valid * DQ);
            else deterministic &= std::memcmp(first.data(), o.data(), (size_t)L_valid * DQ * sizeof(bf16)) == 0;
        }
        ms.erase(ms.begin());
        std::sort(ms.begin(), ms.end());

        // reference: sampled rows (first, chunk edges, last), every head
        std::set<int> rows = {0, L_valid / 2, L_valid - 1};
        for (int r : {15, 16, 127, 128, 129}) if (r < L_valid) rows.insert(r);
        if (vm["all_rows"].as<bool>()) for (int r = 0; r < L_valid; r++) rows.insert(r);
        int worst_r = -1, worst_h = -1;
        double err = 0;
        for (int r : rows) {
            for (int h = 0; h < Heads; h++) {
                const int hk = h / (Heads / NKV);
                std::vector<double> s(L_valid);
                double m = -1e300;
                for (int j = 0; j < L_valid; j++) {
                    double acc = 0;
                    for (int d = 0; d < DH; d++)
                        acc += (double)(float)q[(size_t)r * DQ + h * DH + d] * (double)(float)kv[(size_t)j * DK + hk * DH + d];
                    s[j] = acc / std::sqrt((double)DH);
                    m = std::max(m, s[j]);
                }
                double z = 0;
                for (auto& x : s) { x = std::exp(x - m); z += x; }
                for (int d = 0; d < DH; d++) {
                    double y = 0;
                    for (int j = 0; j < L_valid; j++) y += s[j] * (double)(float)kv[(size_t)L_padded * DK + (size_t)j * DK + hk * DH + d];
                    const double e = std::fabs(y / z - (double)(float)first[(size_t)r * DQ + h * DH + d]);
                    if (e > err) { err = e; worst_r = r; worst_h = h; }
                }
            }
        }
        const bool ok = err <= tol && deterministic;
        pass &= ok;
        std::printf("L_valid %6d  npu %9.3f ms  max|err| %.4f (row %d head %d)  deterministic %s  %s\n",
                    L_valid, ms[ms.size() / 2], err, worst_r, worst_h, deterministic ? "yes" : "NO", ok ? "ok" : "FAIL");
    }
    header_print("info", pass ? "PASSED" : "FAILED!");
    return pass ? 0 : 1;
}
