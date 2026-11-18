// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Drive d3_npu::encode_state_once with whole requests read from files (M3 / M4 validation): states with images or video
// groups, questions with readout rows and images inside their blocks.
//
//   d3_req_run <pkg_dir> <xclbin_root> <list.txt> <out_dir> [--max-length N] [--reps R] [--pack N] [--row-check K] [--rewrite-video]
//
// --row-check K: the first K questions of every request are also run through the per-row prefix_bidir path (encode_batch with
//   state_len; the state's and the question's images as the prompt's payload, video pad ids rewritten to the image pad id) and their
//   last row is compared bit for bit with encode_state_once's (prints rowcheck,<file>,<question>,identical|DIFFERENT,max_abs_diff);
//   with D3_ATTN_ARRAY_MIN=0 (host attention on both sides) the two must be identical.
// --keep-going: an exception of encode_state_once is printed as "error,<file>,<what>" and the next request runs (negative tests).
// --dup-qimages: every question image table entry is stored twice (entry 2k and 2k+1 are equal) and the blocks use the copy 2k+1
//   in odd-numbered questions: the engine must merge them (one vision tower run per distinct image) and give identical rows.
// --rewrite-video: run encode_state_once with video pad ids rewritten to the image pad id (video_token_id = -1) instead of
//   the video_token_id field (the FLM integration's first option); the result must equal the video_token_id run bit for bit.
//
// list.txt: one request file per line (written by d3rig-style exporters, see tools/make_req_file.py). Per file the harness writes
// <out_dir>/<file stem>.out (bf16 [sum of readout rows][D], question order, rows in readout order) and prints one CSV line:
//   file,S,n_questions,rows_total,qimages,ms
// Environment (engine): D3_VISION_TIMING=1 prints one line per vision tower call to stderr, D3_VISION_DUMP=<file> appends every
// call's merger rows, D3_POS_DUMP=<file> appends every segment's M-RoPE positions, D3_ATTN_ARRAY_MIN=0 host attention.
//
// Request file (little endian):
//   "D3RQ" u32 version(1); i32 image_token_id, video_token_id (-1: none); i32 S, S x i32 state ids;
//   i32 n_state_images; n x (i32 grid_h, i32 grid_w); pixels u16 (bf16) [sum grid_h*grid_w][1536];
//   i32 n_question_images; the same; i32 n_questions; per question: i32 n_ids, ids; i32 n_readout, offsets; i32 n_image_ids, ids
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/d3/d3_npu.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

namespace {
struct Reader {
    std::ifstream f;
    std::string path;
    explicit Reader(const std::string& p) : f(p, std::ios::binary), path(p) { if (!f) throw std::runtime_error("cannot open " + p); }
    int32_t i32() {
        int32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);
        if (!f) throw std::runtime_error("short request file " + path);
        return v;
    }
    std::vector<int> ints(int n) {
        if (n < 0 || n > (1 << 26)) throw std::runtime_error("bad count in " + path);
        std::vector<int> v((size_t)n);
        if (n) f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)n * 4);
        if (!f) throw std::runtime_error("short request file " + path);
        return v;
    }
    void payload(d3_image_payload& p) {
        const int n = i32();
        size_t patches = 0;
        for (int k = 0; k < n; k++) {
            const int gh = i32(), gw = i32();
            p.images.push_back({gh, gw});
            patches += (size_t)gh * gw;
        }
        p.pixels.resize(patches * 3 * 2 * 16 * 16);
        if (!p.pixels.empty()) f.read(reinterpret_cast<char*>(p.pixels.data()), (std::streamsize)(p.pixels.size() * sizeof(bf16)));
        if (!f) throw std::runtime_error("short pixels in " + path);
    }
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s pkg_dir xclbin_root list.txt out_dir [--max-length N] [--reps R] [--pack N]\n", argv[0]);
        return 2;
    }
    std::string pkg = argv[1], xroot = argv[2], list = argv[3], out_dir = argv[4];
    int max_length = 32768, reps = 1, pack = -1, row_check = 0;
    bool rewrite_video = false, dup_qimages = false, keep_going = false;
    for (int i = 5; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--max-length" && i + 1 < argc) max_length = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--pack" && i + 1 < argc) pack = std::atoi(argv[++i]);
        else if (a == "--row-check" && i + 1 < argc) row_check = std::atoi(argv[++i]);
        else if (a == "--rewrite-video") rewrite_video = true;
        else if (a == "--dup-qimages") dup_qimages = true;
        else if (a == "--keep-going") keep_going = true;
        else { std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], a.c_str()); return 2; }
    }
    {
        std::filesystem::path p = std::filesystem::absolute(pkg).lexically_normal();
        if (p.filename().empty()) p = p.parent_path();
        if (!std::filesystem::is_directory(p)) { std::fprintf(stderr, "%s: no package directory %s\n", argv[0], pkg.c_str()); return 2; }
        pkg = p.string();
    }
    std::vector<std::string> files;
    {
        std::ifstream lf(list);
        std::string line;
        while (std::getline(lf, line)) if (!line.empty()) files.push_back(line);
        if (files.empty()) { std::fprintf(stderr, "%s: %s lists no request\n", argv[0], list.c_str()); return 2; }
    }
    std::filesystem::create_directories(out_dir);

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
    Q4NX q4nx(pkg);
    d3_npu engine(cfg, &npu, max_length);
    engine.load_weights(q4nx);
    if (pack >= 0) engine.set_pack_rows(pack);
    std::fprintf(stderr, "load_ms,%.0f\n", ms_since(t));
    std::fprintf(stderr, "abi,%d,minor,%d,caps,%u\n", d3_npu_abi_version(), d3_npu_abi_minor(), d3_npu_capabilities());
    const int D = engine.hidden_size();

    std::printf("file,S,n_questions,rows_total,state_images,question_images,rep,ms\n");
    for (const std::string& file : files) {
        d3_image_payload media;
        d3_state_request req;
        try {
            Reader r(file);
            char magic[4];
            r.f.read(magic, 4);
            if (!r.f || std::string(magic, 4) != "D3RQ") throw std::runtime_error("not a request file: " + file);
            if (r.i32() != 1) throw std::runtime_error("unknown request file version");
            media.image_token_id = r.i32();
            req.question_images.image_token_id = media.image_token_id;
            req.video_token_id = r.i32();
            req.state_ids = r.ints(r.i32());
            r.payload(media);
            r.payload(req.question_images);
            const int nq = r.i32();
            for (int q = 0; q < nq; q++) {
                d3_question_block b;
                b.ids = r.ints(r.i32());
                b.readout_from_end = r.ints(r.i32());
                b.image_ids = r.ints(r.i32());
                req.questions.push_back(std::move(b));
            }
            if (rewrite_video && req.video_token_id >= 0) {
                for (int& id : req.state_ids) if (id == req.video_token_id) id = media.image_token_id;
                for (auto& q : req.questions) for (int& id : q.ids) if (id == req.video_token_id) id = media.image_token_id;
                req.video_token_id = -1;
            }
            if (dup_qimages && !req.question_images.images.empty()) {
                d3_image_payload dup;
                dup.image_token_id = req.question_images.image_token_id;
                const size_t per_patch = (size_t)3 * 2 * 16 * 16;
                size_t at = 0;
                for (const auto& g : req.question_images.images) {
                    const size_t n = (size_t)g.grid_h * g.grid_w * per_patch;
                    for (int c = 0; c < 2; c++) {
                        dup.images.push_back(g);
                        dup.pixels.insert(dup.pixels.end(), req.question_images.pixels.begin() + at, req.question_images.pixels.begin() + at + n);
                    }
                    at += n;
                }
                req.question_images = std::move(dup);
                for (size_t qn = 0; qn < req.questions.size(); qn++)
                    for (int& id : req.questions[qn].image_ids) id = 2 * id + (int)(qn % 2);
            }
            if (!media.images.empty()) req.media = &media;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
            return 1;
        }
        size_t rows_total = 0;
        for (auto& q : req.questions) rows_total += q.readout_from_end.empty() ? 1 : q.readout_from_end.size();
        std::vector<bf16> out(rows_total * D);
        bool failed = false;
        for (int rep = 0; rep < reps; rep++) {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<buffer<bf16>> h;
            try {
                h = engine.encode_state_once(req);
            } catch (const std::exception& e) {
                if (keep_going) { std::printf("error,%s,%s\n", std::filesystem::path(file).filename().string().c_str(), e.what()); std::fflush(stdout); failed = true; break; }
                std::fprintf(stderr, "%s: %s: %s\n", argv[0], file.c_str(), e.what());
                return 1;
            }
            const double ms = ms_since(t0);
            size_t row = 0;
            for (size_t k = 0; k < h.size(); k++) {
                const size_t n = req.questions[k].readout_from_end.empty() ? 1 : req.questions[k].readout_from_end.size();
                memcpy(out.data() + row * D, h[k].data(), n * D * sizeof(bf16));
                row += n;
            }
            std::printf("%s,%zu,%zu,%zu,%zu,%zu,%d,%.1f\n", std::filesystem::path(file).filename().string().c_str(), req.state_ids.size(),
                        req.questions.size(), rows_total, media.images.size(), req.question_images.images.size(), rep, ms);
            std::fflush(stdout);
        }
        if (failed) continue;
        for (int k = 0; k < row_check && (size_t)k < req.questions.size(); k++) {
            const auto& q = req.questions[k];
            int last = -1;   // index of the readout row of offset 0
            if (q.readout_from_end.empty()) last = 0;
            for (size_t r = 0; r < q.readout_from_end.size(); r++) if (q.readout_from_end[r] == 0) last = (int)r;
            if (last < 0) continue;
            std::vector<int> ids = req.state_ids;
            ids.insert(ids.end(), q.ids.begin(), q.ids.end());
            for (int& id : ids) if (req.video_token_id >= 0 && id == req.video_token_id) id = media.image_token_id;
            d3_image_payload pay;
            pay.image_token_id = media.image_token_id;
            pay.images = media.images;
            pay.pixels = media.pixels;
            const size_t per_patch = (size_t)3 * 2 * 16 * 16;
            for (int id : q.image_ids) {
                size_t at = 0;
                for (int j = 0; j < id; j++) at += (size_t)req.question_images.images[j].grid_h * req.question_images.images[j].grid_w * per_patch;
                const auto& g = req.question_images.images[id];
                pay.images.push_back(g);
                pay.pixels.insert(pay.pixels.end(), req.question_images.pixels.begin() + at,
                                  req.question_images.pixels.begin() + at + (size_t)g.grid_h * g.grid_w * per_patch);
            }
            pay.state_len = {(int)req.state_ids.size()};
            std::vector<buffer<bf16>> rr;
            try {
                rr = engine.encode_batch({ids}, &pay);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "%s: row check %s question %d: %s\n", argv[0], file.c_str(), k, e.what());
                return 1;
            }
            size_t row0 = 0;
            for (int j = 0; j < k; j++) row0 += req.questions[j].readout_from_end.empty() ? 1 : req.questions[j].readout_from_end.size();
            const bf16* a = out.data() + (row0 + (size_t)last) * D;
            const bf16* b = rr[0].data();
            double md = 0;
            bool same = true;
            for (int c = 0; c < D; c++) if (a[c] != b[c]) { same = false; md = std::max(md, (double)std::fabs((float)a[c] - (float)b[c])); }
            std::printf("rowcheck,%s,%d,%s,%.4g\n", std::filesystem::path(file).filename().string().c_str(), k, same ? "identical" : "DIFFERENT", md);
            std::fflush(stdout);
        }
        const std::string stem = std::filesystem::path(file).stem().string();
        std::ofstream fo(std::filesystem::path(out_dir) / (stem + ".out"), std::ios::binary);
        fo.write(reinterpret_cast<const char*>(out.data()), (std::streamsize)(out.size() * sizeof(bf16)));
        fo.close();
        if (fo.fail()) { std::fprintf(stderr, "%s: could not write the output of %s\n", argv[0], file.c_str()); return 1; }
    }
    return 0;
}
