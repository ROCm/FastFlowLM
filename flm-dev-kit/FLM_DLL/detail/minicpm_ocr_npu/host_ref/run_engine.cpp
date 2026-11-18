// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// \file run_engine.cpp
/// \brief Drives libminicpm_ocr_npu.so directly on S2's fixed prompt, for scoring the engine
///        against models_simple/dumps/minicpm_ocr/ (see README.md in this directory).
///
///   run_engine <model_dir> <input_ids.npy> <out_dir> [--split K] [--gen N]
///
/// Writes to <out_dir>:
///   logits_last.npy   [1, V]  prefill of the whole prompt, last row
///   gen_ids.npy       [N]     greedy continuation (--gen N, default 16)
///   logits_split.npy  [L-K+1, V]  with --split K: prefill ids[0:K], then decode
///                     ids[K:] one token at a time; row j is the logits after token
///                     K-1+j, to be scored against logits.npy[K-1:]
/// With FLM_DUMP_DIR set and a DEBUG_LEVEL >= 1 engine, the full prefill also dumps every
/// manifest tensor it has.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "models/minicpm_ocr/minicpm_ocr_npu.hpp"

static std::vector<int> read_npy_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string header(hl, '\0');
    f.read(header.data(), hl);
    if (header.find("'<i4'") == std::string::npos) throw std::runtime_error(path + ": expected int32");
    std::vector<int> out;
    int v;
    while (f.read(reinterpret_cast<char*>(&v), 4)) out.push_back(v);
    return out;
}

static void write_npy(const std::string& path, const std::vector<float>& data, std::vector<size_t> shape,
                      const char* descr = "<f4") {
    std::string dict = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); i++)
        dict += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
    dict += "), }";
    size_t total = 10 + dict.size() + 1;
    dict.append((64 - total % 64) % 64, ' ');
    dict += '\n';
    std::ofstream f(path, std::ios::binary);
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
    f.write(magic, 8);
    const uint16_t hl = (uint16_t)dict.size();
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(dict.data(), dict.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * 4);
}

static int argmax(buffer<bf16>& logits) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); i++)
        if (static_cast<float>(logits[i]) > static_cast<float>(logits[best])) best = (int)i;
    return best;
}

static void append(std::vector<float>& rows, buffer<bf16>& logits) {
    for (size_t i = 0; i < logits.size(); i++) rows.push_back(static_cast<float>(logits[i]));
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <model_dir> <input_ids.npy> <out_dir> [--split K] [--gen N]\n";
        return 2;
    }
    const std::string model_dir = argv[1], ids_path = argv[2], out_dir = argv[3];
    int split = -1, gen = 16;
    for (int i = 4; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--split") split = std::atoi(argv[i + 1]);
        else if (a == "--gen") gen = std::atoi(argv[i + 1]);
    }

    LM_Config config;
    config.model_path = model_dir;
    config.model_name = "minicpm-v-4.7-1b";
    config.exec_path = ".";
    {
        std::ifstream cf(model_dir + "/config.json");
        config._json_config = nlohmann::json::parse(cf);
    }

    // The all-host build registers no xclbin and opens no hw context, but the weight
    // buffers are still device BOs (see minicpm_ocr_npu.cpp), so it needs the device node.
    flm_rt::device device(0u);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);

    auto t0 = std::chrono::steady_clock::now();
    minicpm_ocr_npu engine(config, &npu, 4096);
    {
        Q4NX q4nx(model_dir);
        engine.load_weights(q4nx);
    }
    std::cout << "load: " << ms_since(t0) << " ms" << std::endl;

    std::vector<int> ids = read_npy_i32(ids_path);
    const int L = (int)ids.size();
    const size_t V = engine.forward(ids[0]).size();   // warm-up token also sizes V
    engine.clear_context();

    // prefill a prefix, decode the rest of the prompt through forward(). Runs first so
    // the FLM_DUMP_DIR dumps the full-prompt prefill below leaves are the ones on disk.
    if (split > 0 && split <= L) {
        engine.clear_context();
        std::vector<int> prefix(ids.begin(), ids.begin() + split);
        std::vector<float> rows;
        buffer<bf16> logits = engine.prefill(prefix);
        append(rows, logits);
        for (int i = split; i < L; i++) {
            buffer<bf16> l = engine.forward(ids[i]);
            append(rows, l);
        }
        write_npy(out_dir + "/logits_split.npy", rows, {(size_t)(L - split + 1), V});
        std::cout << "split: prefill " << split << " + decode " << (L - split) << " tokens" << std::endl;

        // multi-turn: prefill [0, K), then [K, K+2) -- shorter than the conv kernel, so
        // its conv window straddles the carried state -- then [K+2, L). Rows score
        // against logits.npy[K+1] and logits.npy[L-1].
        if (split + 2 < L) {
            engine.clear_context();
            std::vector<float> chunk_rows;
            std::vector<int> c0(ids.begin(), ids.begin() + split);
            std::vector<int> c1(ids.begin() + split, ids.begin() + split + 2);
            std::vector<int> c2(ids.begin() + split + 2, ids.end());
            engine.prefill(c0);
            buffer<bf16> l1 = engine.prefill(c1);
            append(chunk_rows, l1);
            buffer<bf16> l2 = engine.prefill(c2);
            append(chunk_rows, l2);
            write_npy(out_dir + "/logits_chunks.npy", chunk_rows, {2, V});
            std::cout << "chunks: prefill " << split << " + 2 + " << (L - split - 2) << " tokens" << std::endl;
        }
    }

    // full prefill, last-row logits, greedy continuation
    {
        engine.clear_context();
        std::vector<int> prompt = ids;
        t0 = std::chrono::steady_clock::now();
        buffer<bf16> logits = engine.prefill(prompt);
        const double t_prefill = ms_since(t0);
        std::vector<float> row;
        append(row, logits);
        write_npy(out_dir + "/logits_last.npy", row, {1, V});

        std::vector<float> gen_ids;
        int tok = argmax(logits);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < gen; i++) {
            gen_ids.push_back((float)tok);
            buffer<bf16> l = engine.forward(tok);
            tok = argmax(l);
        }
        const double t_decode = ms_since(t0);
        write_npy(out_dir + "/gen_ids.npy", gen_ids, {gen_ids.size()});
        std::cout << "prefill L=" << L << ": " << t_prefill << " ms; first token " << (int)gen_ids[0]
                  << "; decode " << gen << " tokens: " << t_decode / (gen > 0 ? gen : 1) << " ms/token" << std::endl;
        std::cout << "greedy ids:";
        for (float g : gen_ids) std::cout << " " << (int)g;
        std::cout << std::endl;
    }

    return 0;
}
