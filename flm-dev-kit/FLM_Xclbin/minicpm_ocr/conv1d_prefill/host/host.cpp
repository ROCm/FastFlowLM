// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Standalone test for conv.xclbin: causal conv1d (kernel 4) + SiLU over all D channels,
// then an L2 norm over each 128-wide head of the first D_NORM channels (q and k).
//
// usage: ./run.exe [L] [D] [D_NORM]        defaults 1024 6144 4096 (MiniCPM-OCR)
// env:   CONV_XCLBIN  xclbin to load (default build/xclbins/conv8.xclbin)
//        CONV_ITERS   timed runs to average (default 5)
//        CONV_L_ADDR  address of the L rtp buffer (default 33152, conv8.xclbin)
//        CONV_SAVE    file to write the raw output to
//        CONV_DUMP=c  print got / want / conv sum for 16 channels from c, token 0
//        CONV_REF     compare against an intermediate instead, for probe kernels that
//                     output one: conv (raw conv sum), exp (e^-|a|), recip
//                     (1/(1+e^-|a|)), sigmoid, exp_t
//
// The reference is the engine's host path (minicpm_ocr_prefill_cpu.hpp conv1d_silu +
// l2norm_heads): fp32 conv and SiLU, rounded to bf16, then the norm on the bf16 values.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "seq_gen.hpp"

static std::string env_str(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::string(v) : fallback;
}

// conv + SiLU, then the per-head L2 norm, exactly in the engine host's order
static void conv_reference(std::vector<float>& y, const std::vector<bf16>& x, const std::vector<bf16>& w,
                           int L, int D, int D_norm, bool conv_only) {
    constexpr int KS = 4;
    constexpr int DH = 128;
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < L; l++) {
        std::vector<float> row(D);
        for (int c = 0; c < D; c++) {
            float acc = 0.0f;
            for (int t = 0; t < KS; t++) {
                acc += float(x[(size_t)(l + t) * D + c]) * float(w[(size_t)t * D + c]);
            }
            row[c] = conv_only ? float(bf16(acc)) : float(bf16(acc / (1.0f + std::exp(-acc))));
            if (env_str("CONV_REF", "") == "exp") row[c] = float(bf16(std::exp(-std::fabs(acc))));
            if (env_str("CONV_REF", "") == "recip") row[c] = float(bf16(1.0 / (1.0 + std::exp(-std::fabs((double)acc)))));
            if (env_str("CONV_REF", "") == "sigmoid") row[c] = float(bf16(1.0 / (1.0 + std::exp(-(double)acc))));
            if (env_str("CONV_REF", "") == "exp_t") row[c] = float(bf16(std::exp2(float(bf16(-std::fabs(acc) * 1.4426950408889634f)))));
        }
        if (conv_only) {
            for (int c = 0; c < D; c++) y[(size_t)l * D + c] = row[c];
            continue;
        }
        for (int h = 0; h < D_norm / DH; h++) {
            float ss = 0.0f;
            for (int i = 0; i < DH; i++) ss += row[h * DH + i] * row[h * DH + i];
            const float inv = 1.0f / std::sqrt(ss + 1e-6f);
            for (int i = 0; i < DH; i++) row[h * DH + i] = float(bf16(row[h * DH + i] * inv));
        }
        for (int c = 0; c < D; c++) y[(size_t)l * D + c] = row[c];
    }
}

int main(int argc, char** argv) {
    const int L      = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int D      = argc > 2 ? std::atoi(argv[2]) : 6144;
    const int D_norm = argc > 3 ? std::atoi(argv[3]) : 4096;
    const int iters  = std::max(1, std::stoi(env_str("CONV_ITERS", "5")));
    const int l_addr = std::stoi(env_str("CONV_L_ADDR", "33152"));
    const std::string xclbin_name = env_str("CONV_XCLBIN", "build/xclbins/conv8.xclbin");
    constexpr int PAD = 3;               // conv-state rows in front of x
    const int L_in = L + PAD;

    std::cout << "L=" << L << " D=" << D << " D_norm=" << D_norm << " xclbin=" << xclbin_name << std::endl;

    // inputs: roughly the scale of the in_proj outputs and conv weights
    std::mt19937 rng(0);
    std::normal_distribution<float> nx(0.0f, 1.0f);
    std::uniform_real_distribution<float> uw(-0.5f, 0.5f);
    std::vector<bf16> x((size_t)L_in * D), w((size_t)4 * D);
    for (auto& v : x) v = bf16(nx(rng));
    for (auto& v : w) v = bf16(uw(rng));

    std::vector<float> ref((size_t)L * D);
    // CONV_REF=conv: compare against the raw conv sum (for probe kernels that skip SiLU and the norm)
    conv_reference(ref, x, w, L, D, D_norm, env_str("CONV_REF", "") != "");

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mgr = xclbin_mgr.register_xclbin(xclbin_name);
    npu_app app = mgr->create_app();

    buffer<bf16> o_buf = app.create_bo_buffer<bf16>((size_t)L * D);
    buffer<bf16> x_buf = app.create_bo_buffer<bf16>((size_t)L_in * D);
    buffer<bf16> w_buf = app.create_bo_buffer<bf16>((size_t)4 * D);
    std::memcpy(x_buf.data(), x.data(), x.size() * sizeof(bf16));
    std::memcpy(w_buf.data(), w.data(), w.size() * sizeof(bf16));
    for (size_t i = 0; i < (size_t)L * D; i++) o_buf.data()[i] = bf16(-1.0f);
    x_buf.sync_to_device();
    w_buf.sync_to_device();
    o_buf.sync_to_device();

    gen_seq(app.seq(), L_in, D, L, 0, l_addr);

    app(o_buf, x_buf, w_buf);   // warm-up: the first run also loads the xclbin
    double total_us = 0.0;
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        app(o_buf, x_buf, w_buf);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    o_buf.sync_from_device();

    // CONV_DUMP=c: got / want for 16 channels from c, token 0, plus the conv sum there
    if (env_str("CONV_DUMP", "") != "") {
        const int c0 = std::stoi(env_str("CONV_DUMP", "0"));
        for (int c = c0; c < c0 + 16 && c < D; c++) {
            float a = 0.0f;
            for (int t = 0; t < 4; t++) a += float(x[(size_t)t * D + c]) * float(w[(size_t)t * D + c]);
            std::cout << "c=" << c << " a=" << a << " got=" << float(o_buf.data()[c]) << " want=" << ref[c] << std::endl;
        }
    }

    // error, overall and per channel group (q, k / normed part vs v)
    double dot = 0, nn = 0, rr = 0, diff2 = 0, max_abs = 0;
    size_t max_at = 0, untouched = 0;
    for (size_t i = 0; i < (size_t)L * D; i++) {
        const double got = float(o_buf.data()[i]);
        const double want = ref[i];
        dot += got * want; nn += got * got; rr += want * want;
        diff2 += (got - want) * (got - want);
        if (std::fabs(got - want) > max_abs) { max_abs = std::fabs(got - want); max_at = i; }
        if (o_buf.data()[i] == bf16(-1.0f) && want != -1.0) untouched++;
    }
    // the same, split into the normed channels (q, k) and the rest (v)
    for (int part = 0; part < 2; part++) {
        const int c_lo = part == 0 ? 0 : D_norm, c_hi = part == 0 ? D_norm : D;
        double pd = 0, pn = 0, pr = 0, pe = 0, worst_rel = 0;
        for (size_t l = 0; l < (size_t)L; l++) {
            for (int c = c_lo; c < c_hi; c++) {
                const double got = float(o_buf.data()[l * D + c]);
                const double want = ref[l * D + c];
                pd += got * want; pn += got * got; pr += want * want; pe += (got - want) * (got - want);
                if (std::fabs(want) > 0.05) worst_rel = std::max(worst_rel, std::fabs(got - want) / std::fabs(want));
            }
        }
        size_t exact = 0, above = 0, below = 0;
        for (size_t l = 0; l < (size_t)L; l++) {
            for (int c = c_lo; c < c_hi; c++) {
                const float got = float(o_buf.data()[l * D + c]);
                const float want = ref[l * D + c];
                if (got == want) exact++; else if (got > want) above++; else below++;
            }
        }
        const double n_part = (double)L * (c_hi - c_lo);
        std::cout << (part == 0 ? "  q/k (normed): " : "  v (silu only): ")
                  << "exact " << exact / n_part << " above " << above / n_part << " below " << below / n_part << "  ";
        std::cout << (part == 0 ? "" : "")
                  << "cos=" << pd / (std::sqrt(pn) * std::sqrt(pr) + 1e-30)
                  << " relL2=" << std::sqrt(pe / (pr + 1e-30))
                  << " worst rel err (|want|>0.05)=" << worst_rel << std::endl;
    }
    const double cos = dot / (std::sqrt(nn) * std::sqrt(rr) + 1e-30);
    const double rel_l2 = std::sqrt(diff2 / (rr + 1e-30));
    std::cout << "max abs err " << max_abs << " at l=" << max_at / D << " c=" << max_at % D
              << " (got " << float(o_buf.data()[max_at]) << " want " << ref[max_at] << ")" << std::endl;
    std::cout << "SUMMARY L=" << L << " D=" << D << " npu_us=" << total_us / iters
              << " cos=" << cos << " relL2=" << rel_l2 << " untouched=" << untouched << std::endl;

    // CONV_SAVE=<file>: raw output bytes, to compare two builds bit for bit
    const std::string save_path = env_str("CONV_SAVE", "");
    if (!save_path.empty()) {
        std::ofstream f(save_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(o_buf.data()), (size_t)L * D * sizeof(bf16));
    }
    return 0;
}
