// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file inkling2_vision_run.cpp
/// \brief the image tower against the HF goldens (goldens/vision_st/*.safetensors, see scripts/vision_export_st.py)
///
/// Usage: inkling2_vision_run vision_weight.q4nx golden.safetensors [--npu DIR] [--no-preround] [--fused-gelu]
///                            [--profile] [--repeat N]
///   --npu DIR      run the matrix products and the attention on the NPU (DIR holds vision_mm.xclbin and vision_attn.xclbin)
///   --no-preround  skip the bfp16 pre-round of weights and activations
///   --fused-gelu   fc1 and proj_in use the kernel's tanh gelu instead of the host's erf
///   --repeat N     encode N times (the first one pays the sequence setup); the last one is scored
///   Runs the tower on the golden's RGB image (patchify checked against the golden's pixel values first) and prints
///   the cosine and relative L2 error of image_embeds and of every tap the golden holds, with the time per run.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../inkling2_q4nx_file.hpp"
#include "../inkling2_vision.hpp"
#include "../inkling2_vision_npu.hpp"
#include "npu_utils/npu_utils.hpp"

using namespace inkling2_vision;
using clk = std::chrono::steady_clock;

static void score(const char* tag, const float* a, const float* ref, size_t n) {
    double dot = 0, na = 0, nr = 0, d2 = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * ref[i];
        na += (double)a[i] * a[i];
        nr += (double)ref[i] * ref[i];
        d2 += ((double)a[i] - ref[i]) * ((double)a[i] - ref[i]);
        mx = std::max(mx, (double)std::fabs(a[i] - ref[i]));
    }
    std::printf("  %-11s cos %.6f  relL2 %.3e  max|d| %.3e  rms(ref) %.3e\n", tag, dot / std::sqrt(na * nr + 1e-30),
                std::sqrt(d2 / (nr + 1e-30)), mx, std::sqrt(nr / (double)n));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s vision_weight.q4nx golden.safetensors [--npu DIR] [--no-preround] [--fused-gelu] [--profile] [--repeat N]\n", argv[0]);
        return 2;
    }
    std::string npu_dir;
    npu_options opt;
    int repeat = 1;
    bool rows_dbg = false;
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--npu") && i + 1 < argc) npu_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--no-preround")) opt.preround = false;
        else if (!std::strcmp(argv[i], "--fused-gelu")) opt.fused_gelu = true;
        else if (!std::strcmp(argv[i], "--profile")) opt.profile = true;
        else if (!std::strcmp(argv[i], "--rows")) rows_dbg = true;
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    try {
        inkling2_host::q4nx_file gold;
        gold.open(argv[2]);
        const auto& rgb = gold.find("rgb");
        const int H = (int)rgb.shape[0], W = (int)rgb.shape[1];
        const grid g = grid_for(H, W);
        const int* gg = reinterpret_cast<const int*>(gold.find("grid").data);
        if (gg[0] != g.rows || gg[1] != g.cols) throw std::runtime_error("grid differs from the golden's");
        std::printf("%s: %dx%d, grid %dx%d, %d tokens\n", argv[2], W, H, g.rows, g.cols, g.tokens());

        std::vector<float> px((size_t)g.tokens() * CELL * CELL * 3);
        patchify(rgb.data, H, W, px.data());
        const float* ref_px = reinterpret_cast<const float*>(gold.find("pixel_values").data);
        double dmax = 0;
        for (size_t i = 0; i < px.size(); i++) dmax = std::max(dmax, (double)std::fabs(px[i] - ref_px[i]));
        std::printf("  patchify max|d| %.3e\n", dmax);

        tower t;
        auto t0 = clk::now();
        t.load(argv[1]);
        std::printf("  load %.0f ms\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        std::unique_ptr<flm_rt::device> device;
        std::unique_ptr<npu_xclbin_manager> npu;
        if (!npu_dir.empty()) {
            device = std::make_unique<flm_rt::device>(0);
            npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, device.get(), false);
            t0 = clk::now();
            t.set_accel(make_npu_accel(npu.get(), npu_dir + "/vision_mm.xclbin", npu_dir + "/vision_attn.xclbin", opt));
            std::printf("  npu bind %.0f ms (preround %s, %s gelu)\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count(),
                        opt.preround ? "on" : "off", opt.fused_gelu ? "fused tanh" : "host erf");
        }
        std::map<std::string, std::vector<float>> got;
        t.trace = [&](const char* tag, const float* x, int rows, int cols) {
            got[tag].assign(x, x + (size_t)rows * cols);
        };
        std::vector<float> out((size_t)g.tokens() * OUT_DIM);
        std::vector<u16> bf(out.size());
        for (int it = 0; it < repeat; it++) {
            t0 = clk::now();
            t.encode(px.data(), g, bf.data(), out.data());
            std::printf("  encode %.0f ms\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        }

        for (const char* tag : {"embed", "ln_pre", "layer0", "layer11", "layer22", "merger_fc1"}) {
            const auto& r = gold.find(tag);
            const auto it = got.find(tag);
            if (it == got.end() && npu) continue;   // the device path has no merger_fc1 tap
            if (it == got.end()) throw std::runtime_error(std::string("no trace of ") + tag);
            if (it->second.size() * 4 != r.bytes) throw std::runtime_error(std::string("shape of ") + tag);
            score(tag, it->second.data(), reinterpret_cast<const float*>(r.data), it->second.size());
        }
        const auto& r = gold.find("image_embeds");
        score("embeds", out.data(), reinterpret_cast<const float*>(r.data), out.size());
        if (rows_dbg) {
            const float* rf = reinterpret_cast<const float*>(r.data);
            for (int tk = 0; tk < std::min(g.tokens(), 8); tk++) score("  row", out.data() + (size_t)tk * OUT_DIM, rf + (size_t)tk * OUT_DIM, OUT_DIM);
            for (int c0 = 0; c0 < OUT_DIM; c0 += 512) {
                double dot = 0, na = 0, nr = 0;
                for (int tk = 0; tk < g.tokens(); tk++)
                    for (int c = c0; c < std::min(c0 + 512, OUT_DIM); c++) {
                        const double x = out[(size_t)tk * OUT_DIM + c], y = rf[(size_t)tk * OUT_DIM + c];
                        dot += x * y, na += x * x, nr += y * y;
                    }
                std::printf("  cols %4d..: cos %.4f\n", c0, dot / std::sqrt(na * nr + 1e-30));
            }
        }
        std::vector<float> back(out.size());
        for (size_t i = 0; i < back.size(); i++) {
            const uint32_t u = (uint32_t)bf[i] << 16;
            memcpy(&back[i], &u, 4);
        }
        score("embeds bf16", back.data(), reinterpret_cast<const float*>(r.data), out.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
