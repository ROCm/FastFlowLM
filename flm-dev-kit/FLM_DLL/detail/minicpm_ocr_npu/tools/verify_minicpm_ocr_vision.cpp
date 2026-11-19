// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file verify_minicpm_ocr_vision.cpp
/// \brief MiniCPM-V-4.7's vision tower (`image/`) against the captured HF
///        reference, stage by stage, on the host or on the array -- with the
///        controls a cosine alone cannot supply. Modelled on Qwen3.8's N14/N16.
///
/// Build (one line; from FLM_DLL/detail/minicpm_ocr_npu):
///
///   g++-13 -O3 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mfma -fopenmp -std=c++20 -ffast-math
///       -I. -I../../include -I../q4nx -I$XRT_ROOT/include
///       tools/verify_minicpm_ocr_vision.cpp image/vision_tower.cpp image/vision_ops.cpp
///       image/vision_glue.cpp image/mrope.cpp image/npu/vision_mm_sequence.cpp
///       image/npu/vision_attn_sequence.cpp
///       -L../../../../src/lib/xrt -L$XRT_ROOT/lib -lxrt_coreutil -laiebu -luuid -lpthread
///       -o /tmp/verify_minicpm_ocr_vision
///
/// -O3 -ffast-math, not -O2: the host arm is built here, and an A/B against the
/// device has to build it the way the engine ships it.
///
/// Run:
///   /tmp/verify_minicpm_ocr_vision [--backend cpu|npu-attn|npu] [--model DIR]
///        [--ref FILE] [--xclbins DIR] [--bench]
///   FLM_MINICPM_OCR_VISION_FP32=1 ...     # the precision control (host arm)
///
/// The reference is an HF transformers capture (not shipped): one 960x720
/// image, FIVE views (a 28x36 source and a 2x2 grid of 28x36 slices), fp32
/// arithmetic over the checkpoint's bf16 weight values.
///
/// ---------------------------------------------------------------------------
/// The two host runs, and what each one measures
/// ---------------------------------------------------------------------------
///   FLM_MINICPM_OCR_VISION_FP32=1   the reference's own arithmetic. Every stage
///                                 must be ~exact: a gap here is a logic bug.
///   default (bf16)                activations rounded at every op boundary,
///                                 as the device does. The gap from the fp32
///                                 run is the precision floor, not a bug.
/// The device run adds the bfp16 mmul floor at every op on top of that.
///
/// ---------------------------------------------------------------------------
/// Controls that must collapse (host, run once)
/// ---------------------------------------------------------------------------
///   pos table, transposed walk     a w x h lookup instead of h x w. The source
///                                  view is 28 x 36, so this is a real
///                                  permutation, not a symmetry.
///   block 0 attention across ALL   the per-view mask is the one thing a
///     views as one segment         single-view test cannot see.
///   vit_merger attention over the  the window is the one thing that separates
///     full view, not 2x2 windows   it from a block's attention.
///   projector gelu_tanh for erf    gated in fp32 against merger_fc1 only, where
///                                  the activation is the only difference.
///   image_embeds one row shifted   rows right, start wrong.
///   canvas m-rope vs a 1-D walk    positions are integers; the check is exact.
#include "image/mrope.hpp"
#include "image/vision_common.hpp"
#include "image/vision_glue.hpp"
#include "image/vision_model_def.hpp"
#include "image/vision_ops.hpp"
#include "image/vision_profile.hpp"
#include "image/vision_tower.hpp"
#include "image/vision_weights.hpp"
#include "image/npu/vision_op_factory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using mco::view_grid;
using mco::vision::activation;
namespace ref = mco::ref;

int n_pass = 0, n_fail = 0;

void report(const char* tag, bool ok, const std::string& detail) {
    std::printf("  [%s] %-44s %s\n", ok ? "PASS" : "FAIL", tag, detail.c_str());
    if (ok) n_pass++; else n_fail++;
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap; va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct stage {
    std::vector<float> v;
    int rows = 0, cols = 0;
};

/// Score `got` against a reference tensor, on the reference's leading `cols`
/// columns (vm.fc1 is 17280 wide on our side and 17216 in HF's).
ref::err_stats score(const stage& got, const ref::safetensors& R, const std::string& name,
                     int* rows_out = nullptr) {
    const ref::tensor_info& t = R.info(name);
    const int rr = (int)t.shape[0], rc = (int)t.shape[1];
    if (rows_out) *rows_out = rr;
    if (got.rows != rr || got.cols < rc)
        throw std::runtime_error(name + ": got [" + std::to_string(got.rows) + "][" +
                                 std::to_string(got.cols) + "] vs ref [" + std::to_string(rr) +
                                 "][" + std::to_string(rc) + "]");
    const std::vector<float> r = R.f32(name);
    std::vector<float> g((size_t)rr * rc);
    for (int i = 0; i < rr; i++)
        std::memcpy(g.data() + (size_t)i * rc, got.v.data() + (size_t)i * got.cols,
                    sizeof(float) * rc);
    return ref::error_matrix(g.data(), r.data(), g.size());
}

double worst_row_cos(const float* a, const float* b, int rows, int cols) {
    double worst = 2.0;
    for (int r = 0; r < rows; r++) {
        const ref::err_stats s = ref::error_matrix(a + (size_t)r * cols, b + (size_t)r * cols, (size_t)cols);
        worst = std::min(worst, s.cosine);
    }
    return worst;
}

LM_Config load_config(const std::string& model_dir) {
    LM_Config c;
    c.model_path = model_dir;
    std::ifstream f(model_dir + "/config.json");
    if (!f) throw std::runtime_error("no " + model_dir + "/config.json");
    c._json_config = nlohmann::json::parse(f);
    // The packaged config gains vision_model_weight in this section; let the
    // driver run against a package made before that.
    if (c._json_config.value("vision_model_weight", std::string()).empty())
        c._json_config["vision_model_weight"] = "vision_weight.q4nx";
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    std::string backend = "cpu";
    std::string model_dir = "/path/to/models/MiniCPM-V-4.7-1B-NPU2";
    std::string ref_path;
    std::string xclbin_dir = "/path/to/FastFlowLM/src/xclbins/Qwen3.8-27B-NPU2";
    bool bench = false, bench_only = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--backend") backend = next();
        else if (a == "--model") model_dir = next();
        else if (a == "--ref") ref_path = next();
        else if (a == "--xclbins") xclbin_dir = next();
        else if (a == "--bench") bench = true;
        else if (a == "--bench-only") bench = bench_only = true;   // skip the checks: run 0 is truly cold
        else throw std::runtime_error("unknown argument " + a);
    }
    if (ref_path.empty()) ref_path = model_dir + "/vision_ref.safetensors";
    const bool bf16 = mco::vision::cpu_ops_emulate_bf16();
    const bool host = backend == "cpu";

    std::printf("== MiniCPM-V-4.7 vision tower against its HF reference ==\n");
    std::printf("   model    %s\n   ref      %s\n   backend  %s\n   mode     %s\n\n",
                model_dir.c_str(), ref_path.c_str(), backend.c_str(),
                bf16 ? "bf16 activations (shipping)" : "fp32 activations (FLM_MINICPM_OCR_VISION_FP32=1)");

    LM_Config config = load_config(model_dir);
    minicpm_ocr_vision_desc desc;
    desc.build(config);
    // ---- views and pixels (the checks only; --bench-only needs no reference) ----
    std::unique_ptr<ref::safetensors> R_owner;
    std::vector<view_grid> views;
    int P = 0;
    std::vector<float> pixels;
    if (!bench_only) {
        R_owner = std::make_unique<ref::safetensors>(ref_path);
        const std::vector<int32_t> ts = R_owner->i32("target_sizes");
        for (size_t i = 0; i + 1 < ts.size(); i += 2) {
            views.push_back(view_grid{ts[i], ts[i + 1]});
            P += ts[i] * ts[i + 1];
        }
        pixels = R_owner->f32("pixel_values");
        std::printf("   %zu views, %d patches:", views.size(), P);
        for (const view_grid& v : views) std::printf(" %dx%d", v.h, v.w);
        std::printf("\n\n");
    }

    // ---- the device ------------------------------------------------------------
    std::unique_ptr<xrt::device> dev;
    std::unique_ptr<npu_xclbin_manager> mgr;
    std::unique_ptr<mco::vision::op_factory> factory;
    if (host) {
        factory = std::make_unique<mco::vision::cpu_op_factory>();
    } else {
        dev = std::make_unique<xrt::device>(0);
        mgr = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, dev.get(), false);
        // vision_overlay.xclbin in the folder: attention + design-A matmuls on it, unless
        // FLM_MINICPM_OCR_VISION_OVERLAY=0 (the A/B arm)
        const std::string overlay = xclbin_dir + "/vision_overlay.xclbin";
        const char* ov = std::getenv("FLM_MINICPM_OCR_VISION_OVERLAY");
        const bool use_overlay = backend == "npu" && std::filesystem::exists(overlay) && !(ov && ov[0] == '0');
        if (backend != "npu" && backend != "npu-attn")
            throw std::runtime_error("--backend must be cpu, npu-attn or npu");
        if (use_overlay) {
            auto f = std::make_unique<mco_npu::vision_op_factory>(mgr->register_xclbin(overlay));
            f->enable_overlay(mgr->register_xclbin(xclbin_dir + "/vision_mm_b.xclbin"));
            factory = std::move(f);
        } else {
            auto f = std::make_unique<mco_npu::vision_op_factory>(
                mgr->register_xclbin(xclbin_dir + "/vision_attn.xclbin"));
            if (backend == "npu")
                f->enable_mm(mgr->register_xclbin(xclbin_dir + "/vision_mm_a.xclbin"),
                             mgr->register_xclbin(xclbin_dir + "/vision_mm_b.xclbin"));
            factory = std::move(f);
        }
    }

    auto t_load = std::chrono::steady_clock::now();
    mco::vision_tower tower(desc, std::move(factory));
    std::printf("   tower loaded in %.0f ms: %s\n", ms_since(t_load), tower.backend());
    std::printf("%s\n", desc.str().c_str());

    if (!bench_only) {   // the checks; --bench-only skips to the timing with a cold tower
    ref::safetensors& R = *R_owner;

    // ---- 1. canvas m-rope, exact ------------------------------------------------
    std::printf("-- canvas m-rope --\n");
    if (R.has("e2e.input_ids")) {
        const std::vector<int32_t> ids = R.i32("e2e.input_ids");
        const std::vector<int32_t> pos = R.i32("e2e.position_ids");   // [3][L]
        const int L = (int)ids.size();
        std::vector<int> out((size_t)L * 3);
        const int next = mco::canvas_mrope(ids.data(), L, mco::canvas_token_ids{}, views, 0, out.data());
        int bad = 0, max_ref = 0, flat = 0;
        for (int p = 0; p < L; p++)
            for (int c = 0; c < 3; c++) {
                const int r = pos[(size_t)c * L + p];
                max_ref = std::max(max_ref, r);
                if (out[(size_t)p * 3 + c] != r) bad++;
                if (p != r) flat++;   // what a plain 1-D walk would miss
            }
        report("canvas positions match HF word for word", bad == 0,
               fmt("%d of %d differ (L=%d); a 1-D walk differs in %d", bad, 3 * L, L, flat));
        report("next position is max + 1", next == max_ref + 1,
               fmt("next %d, HF max %d", next, max_ref));
        report("control: a 1-D walk is wrong", flat > 0, fmt("%d positions", flat));
    } else {
        std::printf("  (no e2e.* tensors in the reference; skipped)\n");
    }

    // ---- 2. the position lookup, alone ---------------------------------------------
    std::printf("\n-- position table lookup --\n");
    {
        mco::vision::weights W(desc);
        const int D = (int)desc.D;
        std::vector<float> got((size_t)P * D), trans((size_t)views[0].h * views[0].w * D);
        size_t off = 0;
        for (const view_grid& v : views) {
            mco::vision::pos_embed_lookup(W.pos_embed(), (int)desc.num_patches_per_side, D, v.h, v.w,
                                          got.data() + off * D);
            off += (size_t)v.h * v.w;
        }
        const std::vector<float> r = R.f32("pos_embed");
        double maxd = 0;
        for (size_t i = 0; i < got.size(); i++) maxd = std::max(maxd, (double)std::fabs(got[i] - r[i]));
        report("pos_embed is bit-exact (embed - patch_embed)", maxd < 1e-5,
               fmt("max|d| %.3e (HF recovers it by a subtraction, so ~1e-6 is its own error)", maxd));
        // control: the transposed walk on the first, non-square view
        std::vector<float> tmp((size_t)views[0].h * views[0].w * D);
        mco::vision::pos_embed_lookup(W.pos_embed(), (int)desc.num_patches_per_side, D, views[0].w,
                                      views[0].h, tmp.data());
        const ref::err_stats s = ref::error_matrix(tmp.data(), r.data(), tmp.size());
        report("control: transposed walk collapses", s.rel_l2 > 0.3,
               fmt("relL2 %.4f cos %.4f", s.rel_l2, s.cosine));
    }

    // ---- 3. the tower, every tap ------------------------------------------------------
    std::printf("\n-- the tower (%s) --\n", tower.backend());
    std::map<std::string, stage> taps;
    tower.set_tap([&](const char* name, const float* p, int rows, int cols) {
        stage& s = taps[name];
        s.v.assign(p, p + (size_t)rows * cols);
        s.rows = rows; s.cols = cols;
    });
    std::vector<float> emb((size_t)(P / (int)desc.token_divisor()) * desc.out_hidden_size);
    mco::vision::prof::reset();
    auto t0 = std::chrono::steady_clock::now();
    tower.encode(pixels.data(), views, emb.data());
    const double ms_first = ms_since(t0);
    mco::vision::prof::report(ms_first);
    tower.set_tap(nullptr);

    // Bars. fp32 host is the reference's own arithmetic. bf16 host carries the
    // activation-rounding floor down 27 blocks. The device carries the bfp16
    // mmul floor at every op on top: measured op by op in section 4b at ~2.7x
    // the bf16 host error for plain matmuls and ~5-6x for patch_embed and the
    // projector, which compounds to image_embeds cos 0.9968 / relL2 0.080 here
    // (Qwen3.8's tower, one 16x16 view: 0.9992 / 0.040). The device bar sits
    // just under that measured floor so a regression trips it; the task-level
    // arbiter is host_ref/run_vision_e2e's answer, not this number.
    const double bar = !host ? 0.990 : (bf16 ? 0.9990 : 0.99999);
    const double bar_out = !host ? 0.995 : (bf16 ? 0.9995 : 0.99999);
    std::vector<std::string> names = {"patch_embed", "embed"};
    for (uint32_t b = 0; b < desc.depth; b++) {
        char n[32];
        std::snprintf(n, sizeof n, "block_%02u.", b);
        for (const char* s : {"norm1", "attn_proj", "out"})
            if (R.has(std::string(n) + s)) names.push_back(std::string(n) + s);
        if ((int)b == desc.insert_layer_id)
            for (const char* s : {"vm.norm1", "vm.attn_proj", "vm.attn_residual", "vm.merged",
                                  "vm.mean", "vm.pre_norm", "vm.fc1", "vm.out"})
                names.push_back(s);
    }
    for (const char* s : {"post_ln", "merger.merged", "merger.pre_norm", "merger_fc1", "image_embeds"})
        names.push_back(s);

    double worst_cos = 2.0;
    std::string worst_name;
    for (const std::string& n : names) {
        if (!taps.count(n)) { report(n.c_str(), false, "never emitted"); continue; }
        int rows = 0;
        const ref::err_stats s = score(taps[n], R, n, &rows);
        const bool last = n == "image_embeds";
        const double b = last ? bar_out : bar;
        report(n.c_str(), s.cosine >= b,
               fmt("cos %.7f relL2 %.5f  [%d x %d]", s.cosine, s.rel_l2, rows, taps[n].cols));
        if (s.cosine < worst_cos) { worst_cos = s.cosine; worst_name = n; }
    }
    {
        // the padded columns of vm.fc1 must be exactly zero: gelu(0 + 0) = 0
        const stage& f = taps["vm.fc1"];
        const int rc = (int)R.info("vm.fc1").shape[1];
        double mx = 0;
        for (int r = 0; r < f.rows; r++)
            for (int c = rc; c < f.cols; c++) mx = std::max(mx, (double)std::fabs(f.v[(size_t)r * f.cols + c]));
        report("vm.fc1 pad columns are zero", mx == 0.0, fmt("%d pad cols, max|v| %.3e", f.cols - rc, mx));
    }
    {
        const std::vector<float> r = R.f32("image_embeds");
        const int rows = (int)R.info("image_embeds").shape[0], cols = (int)R.info("image_embeds").shape[1];
        report("image_embeds worst row", true,
               fmt("worst row cos %.6f over %d rows", worst_row_cos(emb.data(), r.data(), rows, cols), rows));
        // control: shifted by one row
        const ref::err_stats s = ref::error_matrix(emb.data() + cols, r.data(), (size_t)(rows - 1) * cols);
        report("control: image_embeds shifted one row collapses", s.cosine < bar_out - 0.05,
               fmt("cos %.4f", s.cosine));
    }
    std::printf("   worst stage: %s at cos %.7f; first encode %.0f ms\n", worst_name.c_str(), worst_cos, ms_first);

    // ---- 4. host controls on the reference's own inputs -----------------------------------
    if (host) {
        std::printf("\n-- controls (host, from the reference's own inputs) --\n");
        mco::vision::weights W(desc);
        const int D = (int)desc.D, H = (int)desc.num_heads, DH = (int)desc.DH;
        mco::vision::cpu_mm_op qkv, proj;
        mco::vision::cpu_attn_op attn;

        // (a) block 0's attention per view vs across all views
        {
            const auto& bw = W.block(0);
            qkv.bind(bw.qkv.w, bw.qkv.b, D, 3 * D, activation::none);
            proj.bind(bw.proj.w, bw.proj.b, D, D, activation::none);
            const std::vector<float> n1 = R.f32("block_00.norm1");
            std::vector<float> f((size_t)P * 3 * D), q((size_t)P * D), k((size_t)P * D), v((size_t)P * D);
            qkv.run(n1.data(), P, f.data());
            for (int t = 0; t < P; t++) {
                std::memcpy(&q[(size_t)t * D], &f[(size_t)t * 3 * D], sizeof(float) * D);
                std::memcpy(&k[(size_t)t * D], &f[(size_t)t * 3 * D + D], sizeof(float) * D);
                std::memcpy(&v[(size_t)t * D], &f[(size_t)t * 3 * D + 2 * D], sizeof(float) * D);
            }
            attn.configure(P, H, DH, desc.attn_scale);
            std::vector<float> o((size_t)P * D), y((size_t)P * D);
            size_t off = 0;
            for (const view_grid& g : views) {
                const int Lv = g.h * g.w;
                attn.run(&q[off * D], &k[off * D], &v[off * D], Lv, &o[off * D]);
                off += Lv;
            }
            proj.run(o.data(), P, y.data());
            const std::vector<float> r = R.f32("block_00.attn_proj");
            const ref::err_stats good = ref::error_matrix(y.data(), r.data(), y.size());
            attn.run(q.data(), k.data(), v.data(), P, o.data());
            proj.run(o.data(), P, y.data());
            const ref::err_stats bad = ref::error_matrix(y.data(), r.data(), y.size());
            report("block 0 attention, per view", good.cosine > bar,
                   fmt("cos %.7f relL2 %.5f", good.cosine, good.rel_l2));
            report("control: one segment across views collapses", bad.rel_l2 > 10 * good.rel_l2,
                   fmt("cos %.5f relL2 %.5f (%.0fx)", bad.cosine, bad.rel_l2, bad.rel_l2 / good.rel_l2));
        }
        // (b) vit_merger: 2x2 windows vs full-view attention
        {
            const auto& vw = W.vit_merger();
            qkv.bind(vw.qkv.w, vw.qkv.b, D, 3 * D, activation::none);
            proj.bind(vw.proj.w, vw.proj.b, D, D, activation::none);
            const std::vector<float> n1 = R.f32("vm.norm1");
            std::vector<float> f((size_t)P * 3 * D), q((size_t)P * D), k((size_t)P * D), v((size_t)P * D);
            qkv.run(n1.data(), P, f.data());
            for (int t = 0; t < P; t++) {
                std::memcpy(&q[(size_t)t * D], &f[(size_t)t * 3 * D], sizeof(float) * D);
                std::memcpy(&k[(size_t)t * D], &f[(size_t)t * 3 * D + D], sizeof(float) * D);
                std::memcpy(&v[(size_t)t * D], &f[(size_t)t * 3 * D + 2 * D], sizeof(float) * D);
            }
            std::vector<float> o((size_t)P * D), y((size_t)P * D);
            size_t off = 0;
            for (const view_grid& g : views) {
                mco::vision::window_attention(&q[off * D], &k[off * D], &v[off * D], D, g.h, g.w, H, DH,
                                              desc.attn_scale, &o[off * D]);
                off += (size_t)g.h * g.w;
            }
            proj.run(o.data(), P, y.data());
            const std::vector<float> r = R.f32("vm.attn_proj");
            const ref::err_stats good = ref::error_matrix(y.data(), r.data(), y.size());
            off = 0;
            for (const view_grid& g : views) {
                attn.run(&q[off * D], &k[off * D], &v[off * D], g.h * g.w, &o[off * D]);
                off += (size_t)g.h * g.w;
            }
            proj.run(o.data(), P, y.data());
            const ref::err_stats bad = ref::error_matrix(y.data(), r.data(), y.size());
            report("vit_merger attention, 2x2 windows", good.cosine > bar,
                   fmt("cos %.7f relL2 %.5f", good.cosine, good.rel_l2));
            report("control: full-view attention collapses", bad.rel_l2 > 10 * good.rel_l2,
                   fmt("cos %.5f relL2 %.5f", bad.cosine, bad.rel_l2));
        }
        // (c) projector erf vs tanh, against merger_fc1 -- gated in fp32 only
        {
            const int MI = (int)desc.merger_in();
            const std::vector<float> x = R.f32("merger.pre_norm");
            const int rows = (int)R.info("merger.pre_norm").shape[0];
            const std::vector<float> r = R.f32("merger_fc1");
            mco::vision::cpu_mm_op a, b;
            a.bind(W.merger_fc1().w, W.merger_fc1().b, MI, MI, activation::gelu_erf);
            b.bind(W.merger_fc1().w, W.merger_fc1().b, MI, MI, activation::gelu);
            std::vector<float> ya((size_t)rows * MI), yb((size_t)rows * MI);
            a.run(x.data(), rows, ya.data());
            b.run(x.data(), rows, yb.data());
            const ref::err_stats se = ref::error_matrix(ya.data(), r.data(), ya.size());
            const ref::err_stats st = ref::error_matrix(yb.data(), r.data(), yb.size());
            report(bf16 ? "projector gelu: erf vs tanh (printed only)" : "projector gelu is erf, not tanh",
                   !bf16 ? se.rel_l2 * 3 < st.rel_l2 : true,
                   fmt("erf relL2 %.6f, tanh relL2 %.6f", se.rel_l2, st.rel_l2));
        }
    }

    // ---- 4b. device matmuls one at a time, on the reference's own inputs ---------------------
    // The stage table compounds every op's error into the next op's input. This
    // isolates each one: same input bytes into the device op and into the bf16
    // host op, both scored against the fp32 reference output. The ratio is the
    // device's own floor for that shape (Qwen3.8's N16 check 3, here per op).
    if (backend == "npu") {
        std::printf("\n-- device matmuls in isolation (input = reference tap) --\n");
        mco::vision::weights W(desc);
        mco_npu::vision_op_factory nf(mgr->register_xclbin(xclbin_dir + "/vision_attn.xclbin"));
        nf.enable_mm(mgr->register_xclbin(xclbin_dir + "/vision_mm_a.xclbin"),
                     mgr->register_xclbin(xclbin_dir + "/vision_mm_b.xclbin"));
        const int D = (int)desc.D;
        auto one = [&](const char* tag, const mco::vision::linear& w, int K, int N, activation act,
                       const std::vector<float>& x, int M, const std::vector<float>& want) {
            auto d = nf.make_mm(tag);
            mco::vision::cpu_mm_op h;
            d->bind(w.w, w.b, K, N, act);
            h.bind(w.w, w.b, K, N, act);
            std::vector<float> yd((size_t)M * N), yh((size_t)M * N);
            d->run(x.data(), M, yd.data());
            h.run(x.data(), M, yh.data());
            const ref::err_stats sd = ref::error_matrix(yd.data(), want.data(), yd.size());
            const ref::err_stats sh = ref::error_matrix(yh.data(), want.data(), yh.size());
            const ref::err_stats dh = ref::error_matrix(yd.data(), yh.data(), yd.size());
            std::printf("   %-16s K %5d N %5d M %5d  device relL2 %.5f  host-bf16 %.5f  (x%.1f)  dev-vs-host %.5f\n",
                        tag, K, N, M, sd.rel_l2, sh.rel_l2, sd.rel_l2 / std::max(sh.rel_l2, 1e-12), dh.rel_l2);
            return yd;
        };
        // The fp32 target, independent of the cached bf16-emulation mode.
        auto fp32_linear = [&](const mco::vision::linear& w, int K, int N, activation act,
                               const std::vector<float>& x, int M) {
            std::vector<float> y((size_t)M * N);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float acc = ref::bf16_to_f32(w.b[j]);
                    const uint16_t* wr = w.w + (size_t)j * K;
                    const float* xr = &x[(size_t)i * K];
                    for (int k = 0; k < K; k++) acc += xr[k] * ref::bf16_to_f32(wr[k]);
                    y[(size_t)i * N + j] = mco::vision::apply_activation(acc, act);
                }
            return y;
        };
        auto ln = [&](const std::vector<float>& x, int rows, int cols, const std::vector<float>& g,
                      const std::vector<float>& b) {
            std::vector<float> y(x.size());
            mco::vision::layer_norm(x.data(), rows, cols, g.data(), b.data(), desc.ln_eps, y.data());
            return y;
        };
        auto pad_cols = [](const std::vector<float>& x, int rows, int c0, int c1) {
            std::vector<float> y((size_t)rows * c1, 0.0f);
            for (int r = 0; r < rows; r++)
                std::memcpy(&y[(size_t)r * c1], &x[(size_t)r * c0], sizeof(float) * c0);
            return y;
        };
        one("patch_embed", mco::vision::linear{W.patch_w(), W.patch_b()}, (int)desc.patch_dim(), D,
            activation::none, pixels, P, R.f32("patch_embed"));
        // block 0: qkv from norm1; proj is checked through attn_proj only on the
        // host (its input is the attention output, not tapped). fc1/fc2 from the
        // reconstructed attn residual.
        {
            const auto& bw = W.block(0);
            const std::vector<float> n1 = R.f32("block_00.norm1");
            one("block0.qkv", bw.qkv, D, 3 * D, activation::none, n1, P,
                fp32_linear(bw.qkv, D, 3 * D, activation::none, n1, P));
            std::vector<float> res = R.f32("embed");
            const std::vector<float> ap = R.f32("block_00.attn_proj");
            for (size_t i = 0; i < res.size(); i++) res[i] += ap[i];
            const std::vector<float> n2 = ln(res, P, D, bw.norm2_w, bw.norm2_b);
            // fp32 reference for fc1/fc2: block_00.out - residual = fc2(gelu(fc1(n2)))
            const std::vector<float> out = R.f32("block_00.out");
            std::vector<float> mlp(out.size());
            for (size_t i = 0; i < mlp.size(); i++) mlp[i] = out[i] - res[i];
            const std::vector<float> h1 = fp32_linear(bw.fc1, D, (int)desc.D_ff, activation::gelu, n2, P);
            one("block0.fc1", bw.fc1, D, (int)desc.D_ff, activation::gelu, n2, P, h1);
            one("block0.fc2", bw.fc2, (int)desc.D_ff, D, activation::none, h1, P, mlp);
        }
        {
            const auto& vw = W.vit_merger();
            const int Lq = (int)R.info("vm.pre_norm").shape[0];
            const int VF = (int)desc.vm_ff, VFr = (int)R.info("vm.fc1").shape[1];
            one("vit_merger.fc1", vw.fc1, (int)desc.vm_in(), VF, activation::gelu, R.f32("vm.pre_norm"),
                Lq, pad_cols(R.f32("vm.fc1"), Lq, VFr, VF));
            std::vector<float> want = R.f32("vm.out");
            const std::vector<float> mean = R.f32("vm.mean");
            for (size_t i = 0; i < want.size(); i++) want[i] -= mean[i];
            one("vit_merger.fc2", vw.fc2, VF, D, activation::none, pad_cols(R.f32("vm.fc1"), Lq, VFr, VF),
                Lq, want);
        }
        {
            const int rows = (int)R.info("merger.pre_norm").shape[0];
            const int MI = (int)desc.merger_in();
            one("merger.fc1", W.merger_fc1(), MI, MI, activation::gelu_erf, R.f32("merger.pre_norm"), rows,
                R.f32("merger_fc1"));
            one("merger.fc2", W.merger_fc2(), MI, (int)desc.out_hidden_size, activation::none,
                R.f32("merger_fc1"), rows, R.f32("image_embeds"));
        }
    }

    // ---- 5. timing at the operating points -----------------------------------------------
    }   // !bench_only
    if (bench) {
        std::printf("\n-- encode timing, synthetic pixels at the processor's real view sets --\n");
        struct pt { const char* name; std::vector<view_grid> v; };
        std::vector<pt> pts = {
            {"icon  (1 view 32x32)", {{32, 32}}},
            {"720p  (1280x720)", {{24, 44}, {28, 36}, {28, 36}, {28, 36}, {28, 36}, {28, 36}, {28, 36}}},
            {"1080p (1920x1080)", {{24, 44}, {32, 32}, {32, 32}, {32, 32}, {32, 32},
                                   {32, 32}, {32, 32}, {32, 32}, {32, 32}}},
        };
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        for (const pt& p : pts) {
            int L = 0;
            for (const view_grid& v : p.v) L += v.h * v.w;
            std::vector<float> px((size_t)L * desc.patch_dim());
            for (float& x : px) x = U(rng);
            std::vector<float> o((size_t)(L / (int)desc.token_divisor()) * desc.out_hidden_size);
            double best = 1e30, sum = 0;
            const int reps = 4;
            for (int r = 0; r < reps; r++) {
                mco::vision::prof::reset();
                auto ts0 = std::chrono::steady_clock::now();
                tower.encode(px.data(), p.v, o.data());
                const double ms = ms_since(ts0);
                if (r == 0 || r == reps - 1) mco::vision::prof::report(ms);
                if (r > 0) { best = std::min(best, ms); sum += ms; }
                // VISION_SAVE=<prefix>: the last run's output, to compare two builds bit for bit
                if (const char* sp = std::getenv("VISION_SAVE"); sp && r == reps - 1) {
                    const std::string path = std::string(sp) + "_" + std::to_string(L) + ".bin";
                    FILE* f = std::fopen(path.c_str(), "wb");
                    if (f) { std::fwrite(o.data(), sizeof(float), o.size(), f); std::fclose(f); }
                }
                std::printf("   %s  %zu views %d patches -> %d rows: run %d %.0f ms\n", p.name,
                            p.v.size(), L, L / (int)desc.token_divisor(), r, ms);
            }
            std::printf("   %s  warm best %.0f ms, warm mean %.0f ms\n", p.name, best, sum / (reps - 1));
        }
    }

    std::printf("\n== %d passed, %d failed ==\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
