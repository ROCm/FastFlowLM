// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file host.cpp
/// \brief Bring-up for the MiniCPM-V vision-tower matmul, design P (packed bfp16 operands).
///
/// Every shape runs on seeded random bf16 data, pre-rounded onto the bfp16 grid
/// the way the engine does it, through BOTH design C (bf16 operands, converted
/// on the core) and design P (the same values packed by the host). With
/// pre-rounded inputs C's conversion is exact, so P must reproduce C's y
/// bit for bit; that is the gate. Cosine against an fp32 CPU reference, a
/// rotated-weight control, redispatch identity and pad inertness are checked
/// on P as for C.
///
/// --Preround 0 feeds raw bf16 instead: C then rounds on the core and P's
/// packer truncates, so the mismatch count measures that difference (diagnostic
/// only, not a gate).
///
/// Shapes (vision tower: hidden 1152, intermediate 4304 padded to 4608, merger
/// 4608, vit_merger fc1 17280):
///
///   attn.qkv          L   x 1152 x 3456
///   attn.proj         L   x 1152 x 1152
///   mlp.linear_fc1    L   x 1152 x 4608   (GELU)
///   mlp.linear_fc2    L   x 4608 x 1152
///   merger.fc1        L/4 x 4608 x 4608   (GELU)
///   vit_merger.fc1    L/4 x 4608 x 17280
#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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
#include "../../vision_mm_c/host/seq_gen.hpp"
#include "cpu_ref_vision_mm.hpp"
#include "pack_bfp16.hpp"

namespace po = boost::program_options;
using vision_mm_ref::activation;

namespace {

struct mm_shape {
    std::string name;
    int M, K, N;        // logical
    activation act;
};

constexpr int kM = 64, kK = 384, kN = 96;   // per-core tile, both designs
constexpr int kTileM = kM * 8, kTileN = kN * 4;

int round_up(int v, int q) { return ((v + q - 1) / q) * q; }

double rel_l2(const bf16* y, const bf16* ref, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)(float)y[i] - (double)(float)ref[i];
        num += d * d;
        den += (double)(float)ref[i] * (double)(float)ref[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? INFINITY : 0.0);
}

double cosine(const bf16* y, const bf16* ref, size_t n) {
    double dot = 0.0, ny = 0.0, nr = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double a = (float)y[i], b = (float)ref[i];
        dot += a * b; ny += a * a; nr += b * b;
    }
    return (ny > 0.0 && nr > 0.0) ? dot / (std::sqrt(ny) * std::sqrt(nr)) : 0.0;
}

void fill_random(bf16* p, size_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < n; i++) p[i] = (bf16)dist(rng);
}

/// rows x K, groups of 8 along K (K % 8 == 0)
void preround_rows(bf16* p, size_t rows, size_t K) {
    uint16_t* u = reinterpret_cast<uint16_t*>(p);
    for (size_t i = 0; i < rows * K; i += 8) pack_bfp16::preround8(u + i);
}

uint32_t read_rtp(const std::string& file) {
    std::ifstream f(file);
    if (!f) throw std::runtime_error("missing " + file);
    uint32_t a = 0; f >> a; return a;
}

}  // namespace

int main(int argc, char **argv) {
    srand(0);
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()("Rows,L", po::value<int>()->default_value(1024), "Vision token count L");
    desc.add_options()("Iterations,i", po::value<int>()->default_value(20), "Timing iterations");
    desc.add_options()("CosBar", po::value<double>()->default_value(0.999), "Cosine pass bar");
    desc.add_options()("CtrlBar", po::value<double>()->default_value(0.5), "Control must score BELOW this");
    desc.add_options()("Preround", po::value<int>()->default_value(1), "Pre-round inputs onto the bfp16 grid (0 = raw, diagnostic)");
    desc.add_options()("Ref", po::value<int>()->default_value(1), "Score against the fp32 CPU reference (slow at large L)");
    desc.add_options()("Only", po::value<std::string>()->default_value(""), "Run only this shape");
    desc.add_options()("CXclbin", po::value<std::string>()->default_value("../vision_mm_c/build/VISION_MM_C"),
                       "Design C build dir (reference)");

    arg_utils::parse_options(argc, argv, desc, vm);

    const int L = vm["Rows"].as<int>();
    const int Iterations = vm["Iterations"].as<int>();
    const double cos_bar = vm["CosBar"].as<double>();
    const double ctrl_bar = vm["CtrlBar"].as<double>();
    const bool preround = vm["Preround"].as<int>() != 0;
    const bool use_ref = vm["Ref"].as<int>() != 0;
    const std::string only = vm["Only"].as<std::string>();
    const std::string c_dir = vm["CXclbin"].as<std::string>();

    const int H = 1152, I = 4304, MERGE = 4608;
    const int Lm = L / 4;

    std::vector<mm_shape> shapes = {
        {"attn.qkv",       L,  H,     3 * H, activation::none},
        {"attn.proj",      L,  H,     H,     activation::none},
        {"mlp.fc1",        L,  H,     I,     activation::gelu},
        {"mlp.fc2",        L,  I,     H,     activation::none},
        {"merger.fc1",     Lm, MERGE, MERGE, activation::gelu},
        {"vit_merger.fc1", Lm, MERGE, 17280, activation::gelu},
    };

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);

    const std::string p_dir = "build/VISION_MM_P";
    for (const std::string& f : {c_dir + "/xclbins/vision_mm_c.xclbin", p_dir + "/xclbins/vision_mm_p.xclbin"})
        if (!std::filesystem::exists(f)) { header_print("error", "missing " << f); return 1; }
    npu_app_manager* mgr_c = xclbin_mgr.register_xclbin(c_dir + "/xclbins/vision_mm_c.xclbin");
    npu_app_manager* mgr_p = xclbin_mgr.register_xclbin(p_dir + "/xclbins/vision_mm_p.xclbin");
    // A stale RTP address does not fault, it hangs in wait(): read both from their builds.
    const uint32_t rtp_c = read_rtp(c_dir + "/rtp_addr.txt");
    const uint32_t rtp_p = read_rtp(p_dir + "/rtp_addr.txt");
    header_print("info", "rtp_address C = " << rtp_c << ", P = " << rtp_p << ", preround = " << preround);

    bool all_pass = true;

    for (const auto& s : shapes) {
        if (!only.empty() && s.name != only) continue;
        const int M_pad = round_up(s.M, kTileM), K_pad = round_up(s.K, kK), N_pad = round_up(s.N, kTileN);

        MSG_BONDLINE(72);
        header_print("info", s.name << "  M=" << s.M << " K=" << s.K << " N=" << s.N
                     << "  ->  padded " << M_pad << " x " << K_pad << " x " << N_pad);

        // ---- source data, padded row-major bf16, zero in the pad ----
        const float wscale = 1.0f / std::sqrt((float)s.K);
        std::vector<bf16> x_src((size_t)s.M * s.K), w_src((size_t)s.N * s.K), b_src((size_t)s.N);
        fill_random(x_src.data(), x_src.size(), 0x1234u, 1.0f);
        fill_random(w_src.data(), w_src.size(), 0x5678u, wscale);
        fill_random(b_src.data(), b_src.size(), 0x9abcu, 0.1f);
        if (preround) {
            preround_rows(x_src.data(), s.M, s.K);
            preround_rows(w_src.data(), s.N, s.K);
        }
        std::vector<bf16> x_pad((size_t)M_pad * K_pad, (bf16)0.0f), w_pad((size_t)N_pad * K_pad, (bf16)0.0f);
        for (int i = 0; i < s.M; i++)
            memcpy(x_pad.data() + (size_t)i * K_pad, x_src.data() + (size_t)i * s.K, s.K * sizeof(bf16));
        for (int j = 0; j < s.N; j++)
            memcpy(w_pad.data() + (size_t)j * K_pad, w_src.data() + (size_t)j * s.K, s.K * sizeof(bf16));

        const size_t Y_VOL = (size_t)M_pad * N_pad;

        // ---- design C: bf16 operands, bias then weights ----
        npu_app app_c = mgr_c->create_app();
        buffer<bf16> yc_bo = app_c.create_bo_buffer<bf16>(Y_VOL);
        buffer<bf16> xc_bo = app_c.create_bo_buffer<bf16>(x_pad.size());
        buffer<bf16> wc_bo = app_c.create_bo_buffer<bf16>((size_t)N_pad + w_pad.size());
        memcpy(xc_bo.data(), x_pad.data(), x_pad.size() * sizeof(bf16));
        wc_bo.memset((bf16)0.0f);
        memcpy(wc_bo.data(), b_src.data(), s.N * sizeof(bf16));
        memcpy(wc_bo.data() + N_pad, w_pad.data(), w_pad.size() * sizeof(bf16));
        yc_bo.memset((bf16)(-1.0f));
        xc_bo.sync_to_device(); wc_bo.sync_to_device(); yc_bo.sync_to_device();
        gen_mm_seq(app_c.seq(), M_pad, K_pad, N_pad, kM, kK, kN, 0, (int32_t)s.act, rtp_c);

        // ---- design P: packed operands ----
        const size_t xp_bytes = (size_t)M_pad * K_pad / 8 * 9;
        const size_t wp_bytes = (size_t)N_pad * 2 + (size_t)N_pad * K_pad / 8 * 9;
        npu_app app_p = mgr_p->create_app();
        buffer<bf16> yp_bo = app_p.create_bo_buffer<bf16>(Y_VOL);
        buffer<bf16> xp_bo = app_p.create_bo_buffer<bf16>(xp_bytes / 2);
        buffer<bf16> wp_bo = app_p.create_bo_buffer<bf16>(wp_bytes / 2);
        uint8_t* xp = reinterpret_cast<uint8_t*>(xp_bo.data());
        uint8_t* wp = reinterpret_cast<uint8_t*>(wp_bo.data());
        const time_utils::time_point tp0 = time_utils::now();
        pack_bfp16::pack_rows(xp, reinterpret_cast<const uint16_t*>(x_pad.data()), M_pad, M_pad, K_pad, kM);
        const double pack_x_us = time_utils::duration_us(tp0, time_utils::now()).first;
        memset(wp, 0, (size_t)N_pad * 2);
        memcpy(wp, b_src.data(), s.N * sizeof(bf16));
        pack_bfp16::pack_rows(wp + (size_t)N_pad * 2, reinterpret_cast<const uint16_t*>(w_pad.data()),
                              N_pad, N_pad, K_pad, kN);
        yp_bo.memset((bf16)(-1.0f));
        xp_bo.sync_to_device(); wp_bo.sync_to_device(); yp_bo.sync_to_device();
        gen_mm_p_seq(app_p.seq(), M_pad, K_pad, N_pad, kM, kK, kN, (int32_t)s.act, rtp_p);

        // ---- run C ----
        ert_cmd_state st = app_c(yc_bo, xc_bo, wc_bo);
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("error", s.name << ": C state " << (int)st); all_pass = false; continue;
        }
        yc_bo.sync_from_device();

        // ---- run P twice ----
        time_utils::time_point t0 = time_utils::now();
        st = app_p(yp_bo, xp_bo, wp_bo);
        const double us1 = time_utils::duration_us(t0, time_utils::now()).first;
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("error", s.name << ": P dispatch 1 state " << (int)st); all_pass = false; continue;
        }
        yp_bo.sync_from_device();
        std::vector<bf16> y1(Y_VOL);
        memcpy(y1.data(), yp_bo.data(), Y_VOL * sizeof(bf16));
        yp_bo.memset((bf16)(-1.0f));
        yp_bo.sync_to_device();
        st = app_p(yp_bo, xp_bo, wp_bo);
        if (st != ERT_CMD_STATE_COMPLETED) {
            header_print("error", s.name << ": P dispatch 2 state " << (int)st); all_pass = false; continue;
        }
        yp_bo.sync_from_device();
        const bool redispatch_ok = memcmp(y1.data(), yp_bo.data(), Y_VOL * sizeof(bf16)) == 0;

        // ---- P vs C, element for element over the logical block ----
        size_t n_diff = 0, first_diff = SIZE_MAX;
        const uint16_t* yc = reinterpret_cast<const uint16_t*>(yc_bo.data());
        const uint16_t* yp = reinterpret_cast<const uint16_t*>(y1.data());
        for (int i = 0; i < s.M; i++)
            for (int j = 0; j < s.N; j++) {
                const size_t o = (size_t)i * N_pad + j;
                if (yc[o] != yp[o]) { if (!n_diff) first_diff = o; n_diff++; }
            }

        // ---- accuracy against the fp32 reference ----
        double cos = 1.0, l2 = 0.0, cos_ctrl = 0.0;
        if (use_ref) {
            std::vector<bf16> ref((size_t)s.M * s.N), ctrl((size_t)s.M * s.N), got((size_t)s.M * s.N);
            vision_mm_ref::mm_ref(x_src.data(), w_src.data(), b_src.data(), ref.data(), s.M, s.K, s.N, s.act, 0);
            vision_mm_ref::mm_ref(x_src.data(), w_src.data(), b_src.data(), ctrl.data(), s.M, s.K, s.N, s.act, s.N / 2);
            for (int i = 0; i < s.M; i++)
                memcpy(got.data() + (size_t)i * s.N, y1.data() + (size_t)i * N_pad, s.N * sizeof(bf16));
            cos = cosine(got.data(), ref.data(), got.size());
            l2 = rel_l2(got.data(), ref.data(), got.size());
            cos_ctrl = cosine(got.data(), ctrl.data(), got.size());
        }

        bool pad_zero = true;
        for (int i = 0; i < s.M && pad_zero; i++)
            for (int j = s.N; j < N_pad; j++)
                if ((float)y1[(size_t)i * N_pad + j] != 0.0f) { pad_zero = false; break; }
        for (int i = s.M; i < M_pad && pad_zero; i++)
            for (int j = 0; j < s.N; j++)
                if ((float)y1[(size_t)i * N_pad + j] != 0.0f) { pad_zero = false; break; }

        const bool bit_ok = n_diff == 0;
        const bool ok = (cos >= cos_bar) && (cos_ctrl < ctrl_bar) && redispatch_ok && (bit_ok || !preround);

        header_print("info", "  P vs C        : " << (bit_ok ? "bit-identical" : "DIFFERS") << "  ("
                     << n_diff << " of " << (size_t)s.M * s.N << " differ"
                     << (bit_ok ? "" : ", first at row " + std::to_string(first_diff / N_pad) +
                                       " col " + std::to_string(first_diff % N_pad)) << ")"
                     << (preround ? (bit_ok ? "  PASS" : "  FAIL") : "  (diagnostic)"));
        if (use_ref) {
        header_print("info", "  cosine        : " << std::setprecision(6) << cos
                     << (cos >= cos_bar ? "  PASS" : "  FAIL"));
        header_print("info", "  relative L2   : " << l2);
        header_print("info", "  control cos   : " << cos_ctrl << (cos_ctrl < ctrl_bar ? "  PASS" : "  FAIL"));
        }
        header_print("info", "  redispatch    : " << (redispatch_ok ? "bit-identical  PASS" : "DIFFERS  FAIL"));
        header_print("info", "  pad inert     : " << (pad_zero ? "yes" : "NO - pad is nonzero"));
        header_print("info", "  first P us    : " << us1 << "   (host x pack " << pack_x_us << " us, scalar)");
        if (!ok) all_pass = false;

        if (Iterations > 0 && ok) {
            // In blocks, one warm-up each: alternating the two xclbins every call
            // times the context switch, not the matmul.
            double acc_c = 0.0, acc_p = 0.0;
            app_c(yc_bo, xc_bo, wc_bo);
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point a = time_utils::now();
                app_c(yc_bo, xc_bo, wc_bo);
                acc_c += time_utils::duration_us(a, time_utils::now()).first;
            }
            app_p(yp_bo, xp_bo, wp_bo);
            for (int i = 0; i < Iterations; i++) {
                time_utils::time_point a = time_utils::now();
                app_p(yp_bo, xp_bo, wp_bo);
                acc_p += time_utils::duration_us(a, time_utils::now()).first;
            }
            header_print("info", "  time C / P us : " << std::setprecision(5) << acc_c / Iterations << " / "
                         << acc_p / Iterations << "   (" << std::setprecision(3)
                         << 100.0 * (acc_p - acc_c) / acc_c << "%)");
            time_utils::time_with_unit t = {acc_p / Iterations, "us"};
            utils::print_npu_profile(t, 2.0 * (double)M_pad * (double)K_pad * (double)N_pad);
        }
    }

    MSG_BONDLINE(72);
    if (all_pass) { header_print("info", "PASSED"); return 0; }
    header_print("info", "FAILED!");
    return 1;
}
