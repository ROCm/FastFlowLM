// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Drive libd3_npu.so over prompts given as token ids: the engine-level harness for accuracy and
// latency measurements and for .github/workflows/d3-check.yml.
//
//   d3_engine_run <pkg_dir> <xclbin_root> <prompts.txt> <out.bin> [--pack N] [--causal] [--group]
//                 [--reps R] [--max-length N] [--state-file F] [--state-once] [--image GHxGW] [--pix-dir D] [--rig-dump D]
//
// prompts.txt: one prompt per line, "<group> <id> <id> ...". With --group, consecutive lines that
// share <group> go through one encode_batch() call (a request's questions); otherwise every line is
// its own encode(). --pack N sets the rows per packed pass (0: one prompt per pass); without it the
// engine's default (what FastFlowLM serves with) applies. out.bin: bf16 [n_prompts, D], input order.
// stdout: CSV timing per call. --state-file F: one integer per prompt (same order), the prefix_bidir state
// length S of that prompt (0 = no split, noncausal_full_attention numerics); passed to the engine as
// d3_image_payload::state_len. Without it the call has no payload at all, as before.
// --pix-dir D: D/<group>.pix supplies the images of a group (a request): int32 n_images; per image int32 grid_h,
// int32 grid_w; then every image's patches as uint16 bf16 [grid_h*grid_w][3*2*16*16] (d3rig/export_pixels.py).
// --rig-dump D (RIG=1 build; needs one prompt per call, i.e. no --group): sets D3_RIG_DUMP=D/call<k> before call k.
// A group without a file is text only. Needs vision_mm / vision_attn xclbins and vision_weight.q4nx.
// --state-once (needs --state-file; one S per group, the prompts of a group share their first S tokens): the group goes through
// d3_npu::encode_state_once(state = those S tokens, questions = the rest of every prompt) instead of encode_batch(); the output
// rows are the same quantity, so out.bin of both modes compare directly (bit-identical with host attention). It also checks that
// readout_from_end = {0, 2} returns the plain result as its first row.
// --image GHxGW: every prompt holds one image (a run of GH*GW/4 <|image_pad|> tokens, 248056, in its state, in the prompts file) of a
// GHxGW patch grid with synthetic random pixels (fixed seed): the vision tower runs once per call; with --state-once the image belongs
// to the state (d3_state_request::media). Needs vision_weight.q4nx in the package and vision_*.xclbin next to mm.xclbin.
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/d3/d3_npu.hpp"   // FLM_DLL/include
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s pkg_dir xclbin_root prompts.txt out.bin [--pack N] [--causal] [--group] [--reps R] [--max-length N] [--state-file F]\n", argv[0]);
        return 2;
    }
    std::string pkg = argv[1], xroot = argv[2], in_path = argv[3], out_path = argv[4];
    {
        // the engine picks xclbins/<package dir name>/mm.xclbin: "pkg/" or "." must still name the package
        // (lexically, not canonical(): a symlinked package keeps its own name)
        std::filesystem::path p = std::filesystem::absolute(pkg).lexically_normal();
        if (p.filename().empty()) p = p.parent_path();
        if (!std::filesystem::is_directory(p)) {
            std::fprintf(stderr, "%s: no package directory %s\n", argv[0], pkg.c_str());
            return 2;
        }
        pkg = p.string();
    }
    int pack = -1, reps = 1, max_length = 32768;   // pack -1: the engine's default
    bool causal = false, group = false, state_once = false;
    std::string state_file, image_arg, pix_dir, rig_dump;
    for (int i = 5; i < argc; i++) {
        std::string a = argv[i];
        // an option that takes a value must have one, and it must be a whole integer
        auto value = [&](int& dst) {
            size_t used = 0;
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            const std::string v = argv[++i];
            try {
                dst = std::stoi(v, &used);
            } catch (const std::exception&) {
                used = 0;
            }
            if (used == 0 || used != v.size()) throw std::invalid_argument(a + " takes an integer, got '" + v + "'");
        };
        try {
            if (a == "--pack") value(pack);
            else if (a == "--causal") causal = true;
            else if (a == "--group") group = true;
            else if (a == "--state-once") state_once = true;
            else if (a == "--image") {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                image_arg = argv[++i];
            }
            else if (a == "--reps") value(reps);
            else if (a == "--max-length") value(max_length);
            else if (a == "--rig-dump") {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                rig_dump = argv[++i];
            }
            else if (a == "--pix-dir") {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                pix_dir = argv[++i];
            }
            else if (a == "--state-file") {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                state_file = argv[++i];
            }
            else throw std::invalid_argument("unknown option " + a);
        } catch (const std::invalid_argument& e) {
            std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
            return 2;
        }
    }
    if (state_once && (state_file.empty() || !group)) {
        std::fprintf(stderr, "%s: --state-once needs --state-file and --group\n", argv[0]);
        return 2;
    }
    // before the device and the weights: --reps 0 would skip every encode and still write a
    // full-size, uncomputed output file
    if (reps < 1 || max_length < 1 || pack < -1) {
        std::fprintf(stderr, "%s: --reps and --max-length must be >= 1, --pack >= 0\n", argv[0]);
        return 2;
    }
    std::vector<std::string> gid;
    std::vector<std::vector<int>> prompts;
    {
        // every check here runs before the device and the weights are touched
        std::ifstream f(in_path);
        if (!f) {
            std::fprintf(stderr, "%s: cannot open %s\n", argv[0], in_path.c_str());
            return 2;
        }
        std::string line;
        size_t line_no = 0;
        while (std::getline(f, line)) {
            line_no++;
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string g, tok;
            ss >> g;
            std::vector<int> ids;
            while (ss >> tok) {   // each token must be a whole non-negative integer that fits an int
                size_t used = 0;
                long v = -1;
                try {
                    v = std::stol(tok, &used);
                } catch (const std::exception&) {
                    used = 0;
                }
                if (used != tok.size() || v < 0 || v > INT32_MAX) {
                    std::fprintf(stderr, "%s: %s line %zu: '%s' is not a token id\n", argv[0], in_path.c_str(), line_no, tok.c_str());
                    return 2;
                }
                ids.push_back((int)v);
            }
            gid.push_back(g);
            prompts.push_back(ids);
        }
        if (f.bad()) {
            std::fprintf(stderr, "%s: error reading %s\n", argv[0], in_path.c_str());
            return 2;
        }
        if (prompts.empty()) {
            std::fprintf(stderr, "%s: %s has no prompts\n", argv[0], in_path.c_str());
            return 2;
        }
    }

    std::vector<int> states;
    if (!state_file.empty()) {
        std::ifstream sf(state_file);
        int v;
        while (sf >> v) states.push_back(v);
        if (states.size() != prompts.size()) {
            std::fprintf(stderr, "%s: %s holds %zu state lengths for %zu prompts\n", argv[0], state_file.c_str(), states.size(), prompts.size());
            return 2;
        }
    }

    flm_rt::device device(0);   // xrt::device, or hrx::device in an FLM_USE_HRX=1 build
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    // LM_Config::from_pretrained lives in FastFlowLM, not in FLM_DLL: fill the fields it would set
    LM_Config cfg;
    cfg.model_path = pkg;
    cfg.model_name = std::filesystem::path(pkg).filename().string();
    cfg.exec_path = xroot;
    {
        std::ifstream cf(std::filesystem::path(pkg) / "config.json");
        cfg._json_config = nlohmann::json::parse(cf);
        cfg._json_config["is_vlm"] = false;
        if (cfg._json_config.contains("vision_model_weight")) {   // FastFlowLM makes it absolute
            std::filesystem::path w = cfg._json_config["vision_model_weight"].get<std::string>();
            if (w.is_relative()) cfg._json_config["vision_model_weight"] = (std::filesystem::path(pkg) / w).string();
        }
        cfg._json_config["is_audio"] = false;
        cfg.flm_version = cfg._json_config.value("flm_version", std::string("0.0.0"));
    }
    d3_image_payload image_payload;   // --image: one synthetic image shared by every prompt
    if (!image_arg.empty()) {
        int gh = 0, gw = 0;
        if (std::sscanf(image_arg.c_str(), "%dx%d", &gh, &gw) != 2 || gh < 2 || gw < 2 || gh % 2 || gw % 2) {
            std::fprintf(stderr, "%s: --image takes GHxGW with even sizes >= 2\n", argv[0]);
            return 2;
        }
        image_payload.images.push_back({gh, gw});
        std::mt19937 rng(2024);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        image_payload.pixels.resize((size_t)gh * gw * 3 * 2 * 16 * 16);
        for (auto& v : image_payload.pixels) v = bf16(nd(rng));
        cfg._json_config["vision_model_weight"] = (std::filesystem::path(pkg) / "vision_weight.q4nx").string();
    }
    auto t = std::chrono::steady_clock::now();
    Q4NX q4nx(pkg);
    d3_npu engine(cfg, &npu, max_length);
    engine.load_weights(q4nx);
    if (pack >= 0) engine.set_pack_rows(pack);   // else the default FastFlowLM serves with
    engine.set_causal(causal);
    std::fprintf(stderr, "load_ms,%.0f\n", ms_since(t));

    const int D = engine.hidden_size();
    std::vector<bf16> out((size_t)prompts.size() * D);
    std::printf("call,first,n_prompts,rows,rep,ms\n");
    size_t call = 0;
    for (int rep = 0; rep < reps; rep++) {
        size_t i = 0;
        call = 0;
        while (i < prompts.size()) {
            size_t j = i + 1;
            if (group) while (j < prompts.size() && gid[j] == gid[i]) j++;
            std::vector<std::vector<int>> batch(prompts.begin() + i, prompts.begin() + j);
            int rows = 0;
            for (auto& p : batch) rows += (int)p.size();
            if (!rig_dump.empty()) setenv("D3_RIG_DUMP", (rig_dump + "/call" + std::to_string(call)).c_str(), 1);
            auto t0 = std::chrono::steady_clock::now();
            d3_image_payload payload;   // images (--image / --pix-dir) and optionally the prompts' state lengths
            if (!image_arg.empty()) { payload.images = image_payload.images; payload.pixels = image_payload.pixels; }
            if (!states.empty()) payload.state_len.assign(states.begin() + i, states.begin() + j);
            if (!pix_dir.empty()) {
                std::ifstream pf(std::filesystem::path(pix_dir) / (gid[i] + ".pix"), std::ios::binary);
                if (pf) {
                    int32_t n = 0;
                    pf.read(reinterpret_cast<char*>(&n), 4);
                    size_t patches = 0;
                    for (int32_t k = 0; k < n; k++) {
                        int32_t g[2];
                        pf.read(reinterpret_cast<char*>(g), 8);
                        payload.images.push_back({g[0], g[1]});
                        patches += (size_t)g[0] * g[1];
                    }
                    payload.pixels.resize(patches * 3 * 2 * 16 * 16);
                    pf.read(reinterpret_cast<char*>(payload.pixels.data()), payload.pixels.size() * sizeof(bf16));
                    if (!pf) { std::fprintf(stderr, "%s: short pixel file for %s\n", argv[0], gid[i].c_str()); return 1; }
                }
            }
            std::vector<buffer<bf16>> h;
            double ms = 0;
            if (state_once) {
                const int S = states[i];
                d3_state_request req;
                if (!image_arg.empty()) req.media = &image_payload;
                else if (!payload.images.empty()) req.media = &payload;
                if (S < 1) { std::fprintf(stderr, "%s: --state-once needs S >= 1 (line group %s)\n", argv[0], gid[i].c_str()); return 2; }
                for (auto& p : batch) {
                    if ((int)p.size() <= S || states[i + (&p - &batch[0])] != S || !std::equal(p.begin(), p.begin() + S, batch[0].begin())) {
                        std::fprintf(stderr, "%s: group %s: prompts must share their first S=%d tokens and be longer than S\n", argv[0], gid[i].c_str(), S);
                        return 2;
                    }
                }
                req.state_ids.assign(batch[0].begin(), batch[0].begin() + S);
                for (auto& p : batch) {
                    d3_question_block qb;
                    qb.ids.assign(p.begin() + S, p.end());
                    req.questions.push_back(std::move(qb));
                }
                h = engine.encode_state_once(req);
                ms = ms_since(t0);
                if (rep == 0 && call == 0) {   // multi-row readout (untimed): the first row is the plain result
                    d3_state_request one;
                    one.state_ids = req.state_ids;
                    one.media = req.media;
                    one.questions.push_back(req.questions[0]);
                    one.questions[0].readout_from_end = {0, 2};
                    auto r = engine.encode_state_once(one);
                    const bool ok = r.size() == 1 && r[0].size() == 2 * (size_t)D && memcmp(r[0].data(), h[0].data(), D * sizeof(bf16)) == 0 &&
                                    memcmp(r[0].data() + D, h[0].data(), D * sizeof(bf16)) != 0;
                    std::fprintf(stderr, "readout_rows_check,%s\n", ok ? "ok" : "FAIL");
                }
            } else {
                h = engine.encode_batch(batch, (states.empty() && payload.images.empty()) ? nullptr : &payload);
                ms = ms_since(t0);
            }
            std::printf("%zu,%zu,%zu,%d,%d,%.3f\n", call, i, j - i, rows, rep, ms);
            for (size_t k = 0; k < h.size(); k++) memcpy(out.data() + (i + k) * D, h[k].data(), D * sizeof(bf16));
            i = j;
            call++;
        }
    }
    std::ofstream fo(out_path, std::ios::binary);
    fo.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(bf16));
    fo.close();   // flushes; a full disk or an unwritable path shows up here, not as exit 0
    if (fo.fail()) {
        std::fprintf(stderr, "%s: could not write %s\n", argv[0], out_path.c_str());
        return 1;
    }
    return 0;
}
