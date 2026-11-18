// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Vision tower alone (M4 measurements): time encode() for many segments (temporal groups / images) and check that a segment's
// merger rows do not depend on the other segments of the call.
//
//   d3_vision_bench <pkg_dir> <xclbin_root> <spec> [--reps R] [--check K] [--seed N]
//
// spec: comma separated "NxGHxGW" (N segments of a GH x GW patch grid, e.g. 16x28x28 = a 32-frame window with 784-patch groups) or
// "GHxGW" (one segment). Pixels are random (fixed seed per segment). Prints one CSV line per call:
//   spec,segments,patches,tokens,rep,ms
// --dump PREFIX: write PREFIX.pix (int32 n; per segment int32 grid_h, grid_w; bf16 patches [n_patches][1536], the d3_engine_run
// --pix-dir format), PREFIX.joint (the joint call's merger rows, bf16 [tokens][out hidden]) and PREFIX.alone<k> (--check segments alone).
// --check K: after the joint call, encode each of the first K segments alone and compare their rows bit for bit with the joint result
// (prints check,<segment>,identical|DIFFERENT,max_abs_diff). Segments are independent in the tower: this must be identical.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/d3/d3_npu.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "d3_vision.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s pkg_dir xclbin_root spec [--reps R] [--check K] [--seed N]\n", argv[0]); return 2; }
    std::string pkg = std::filesystem::absolute(argv[1]).lexically_normal().string(), xroot = argv[2], spec = argv[3];
    while (pkg.size() > 1 && pkg.back() == '/') pkg.pop_back();
    int reps = 2, check = 0, seed = 1;
    std::string dump;
    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--check" && i + 1 < argc) check = std::atoi(argv[++i]);
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<std::pair<int, int>> grids;
    {
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int n = 1, gh = 0, gw = 0;
            if (std::sscanf(item.c_str(), "%dx%dx%d", &n, &gh, &gw) != 3 && std::sscanf(item.c_str(), "%dx%d", &gh, &gw) != 2) { std::fprintf(stderr, "bad spec item %s\n", item.c_str()); return 2; }
            for (int k = 0; k < n; k++) grids.push_back({gh, gw});
        }
    }
    flm_rt::device device(0);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    LM_Config cfg;
    cfg.model_path = pkg;
    cfg.model_name = std::filesystem::path(pkg).filename().string();
    cfg.exec_path = xroot;
    {
        std::ifstream cf(std::filesystem::path(pkg) / "config.json");
        cfg._json_config = nlohmann::json::parse(cf);
        cfg._json_config["is_vlm"] = false;
        cfg._json_config["vision_model_weight"] = (std::filesystem::path(pkg) / "vision_weight.q4nx").string();
        cfg._json_config["is_audio"] = false;
        cfg.flm_version = cfg._json_config.value("flm_version", std::string("0.0.0"));
    }
    auto t = std::chrono::steady_clock::now();
    D3VisionEncoder enc(cfg, &npu);
    SafeTensors w(cfg.get<std::string>("vision_model_weight", ""));
    enc.init_weights(w);
    std::fprintf(stderr, "load_ms,%.0f\n", ms_since(t));
    const size_t per_patch = 3 * 2 * 16 * 16;
    std::vector<std::vector<bf16>> seg_pix(grids.size());
    for (size_t i = 0; i < grids.size(); i++) {
        std::mt19937 rng(seed * 100003 + (unsigned)i);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        seg_pix[i].resize((size_t)grids[i].first * grids[i].second * per_patch);
        for (auto& v : seg_pix[i]) v = bf16(nd(rng));
    }
    d3_image_payload p;
    size_t patches = 0, tokens = 0;
    for (size_t i = 0; i < grids.size(); i++) {
        p.images.push_back({grids[i].first, grids[i].second});
        p.pixels.insert(p.pixels.end(), seg_pix[i].begin(), seg_pix[i].end());
        patches += (size_t)grids[i].first * grids[i].second;
        tokens += (size_t)grids[i].first * grids[i].second / 4;
    }
    if (!dump.empty()) {
        std::ofstream pf(dump + ".pix", std::ios::binary);
        const int32_t n = (int32_t)grids.size();
        pf.write(reinterpret_cast<const char*>(&n), 4);
        for (auto& g : grids) { const int32_t gg[2] = {g.first, g.second}; pf.write(reinterpret_cast<const char*>(gg), 8); }
        pf.write(reinterpret_cast<const char*>(p.pixels.data()), (std::streamsize)(p.pixels.size() * sizeof(bf16)));
    }
    auto write_rows = [&](const std::string& path, const std::vector<bf16>& v, size_t tok) {
        std::ofstream f(path, std::ios::binary);
        for (size_t r = 0; r < tok; r++) f.write(reinterpret_cast<const char*>(v.data() + r * enc.Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE), (std::streamsize)(enc._QWEN3_5_VISION_OUT_HIDDEN_SIZE * sizeof(bf16)));
    };
    std::printf("spec,segments,patches,tokens,rep,ms\n");
    std::vector<bf16> joint;
    const size_t stride = enc.Padded_QWEN3_5_VISION_OUT_HIDDEN_SIZE;
    for (int rep = 0; rep < reps; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        auto f = enc.encode(p);
        const double ms = ms_since(t0);
        std::printf("%s,%zu,%zu,%zu,%d,%.1f\n", spec.c_str(), grids.size(), patches, tokens, rep, ms);
        std::fflush(stdout);
        if (rep == 0) { joint = f.back(); if (!dump.empty()) write_rows(dump + ".joint", joint, tokens); }
        else {
            size_t diff_rows = 0;
            const size_t rows = f.back().size() / stride;
            for (size_t r = 0; r < tokens && r < rows; r++)
                for (size_t c = 0; c < enc._QWEN3_5_VISION_OUT_HIDDEN_SIZE; c++)
                    if (f.back()[r * stride + c] != joint[r * stride + c]) { diff_rows++; break; }
            std::printf("repeat,%d,rows_differing_from_rep0,%zu\n", rep, diff_rows);
        }
    }
    size_t row = 0;
    for (int k = 0; k < check && (size_t)k < grids.size(); k++) {
        d3_image_payload one;
        one.images.push_back({grids[k].first, grids[k].second});
        one.pixels = seg_pix[k];
        auto t0 = std::chrono::steady_clock::now();
        auto f = enc.encode(one);
        const size_t n = (size_t)grids[k].first * grids[k].second / 4;
        // rows of segment k in the joint result
        size_t at = 0;
        for (int j = 0; j < k; j++) at += (size_t)grids[j].first * grids[j].second / 4;
        double md = 0;
        bool same = true;
        size_t bad_rows = 0, first_bad = 0;
        for (size_t r = 0; r < n; r++) {
            bool row_bad = false;
            for (size_t c = 0; c < enc._QWEN3_5_VISION_OUT_HIDDEN_SIZE; c++) {
                const float a = (float)joint[(at + r) * stride + c], b = (float)f.back()[r * stride + c];
                if (a != b) { same = false; row_bad = true; md = std::max(md, (double)std::fabs(a - b)); }
            }
            if (row_bad) { if (!bad_rows) first_bad = r; bad_rows++; }
        }
        if (!dump.empty()) write_rows(dump + ".alone" + std::to_string(k), f.back(), n);
        std::printf("check,%d,%s,%.4g,rows_differing,%zu/%zu,first,%zu,alone_ms,%.1f\n", k, same ? "identical" : "DIFFERENT", md, bad_rows, n, first_bad, ms_since(t0));
        (void)row;
    }
    return 0;
}
