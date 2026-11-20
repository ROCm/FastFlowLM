// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Engine-level harness for libqwen3_asr_npu.so: prefill on the CPU, decode on the NPU, against the frozen goldens.
//
//   qwen3_asr_engine_run <pkg_dir> <xclbin_root> <goldens_dir> <clip> [<clip> ...] [--max-length N] [--cpu-decode]
//
// pkg_dir: the weights package (config.json, model.q4nx, tokenizer.json); the engine reads
//   <xclbin_root>/xclbins/<basename of pkg_dir>/layer.xclbin.
// goldens_dir/<clip>/{meta.json, enc.npy}: prompt token ids and the reference encoder output (the audio embeddings).
// Per clip: engine.prefill(ids, audio payload) -> greedy decode with engine.forward() until EOS, compare the tokens with
// meta.json["gen_ids"] and the first three logit vectors' top-20 with meta.json["top20_first3"].
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "lm_config.hpp"
#include "models/qwen3_asr/qwen3_asr_npu.hpp"
#include "nlohmann/json.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "npy_io.hpp"

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s pkg_dir xclbin_root goldens_dir clip... [--max-length N]\n", argv[0]);
        return 2;
    }
    std::filesystem::path p = std::filesystem::absolute(argv[1]).lexically_normal();
    if (p.filename().empty()) p = p.parent_path();
    const std::string pkg = p.string(), xroot = argv[2], gdir = argv[3];
    std::vector<std::string> clips;
    int max_length = 8192;
    bool use_mel = false;
    std::string out_ids;
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--max-length" && i + 1 < argc) max_length = std::stoi(argv[++i]);
        else if (a == "--mel") use_mel = true;
        else if (a == "--out-ids" && i + 1 < argc) out_ids = argv[++i];
        else clips.push_back(a);
    }
    constexpr int EOS0 = 151645, EOS1 = 151643;

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
        cfg._json_config["is_audio"] = false;
        cfg.flm_version = "0.0.0";
    }
    auto t = std::chrono::steady_clock::now();
    Q4NX q4nx(pkg);
    qwen3_asr_npu engine(cfg, &npu, max_length);
    engine.load_weights(q4nx);
    std::fprintf(stderr, "load_ms,%.0f\n", ms_since(t));
    const int D = (int)cfg.get("hidden_size");

    int bad = 0;
    std::ofstream ids_out;
    if (!out_ids.empty()) ids_out.open(out_ids);
    for (const std::string& clip : clips) {
        std::ifstream mf(gdir + "/" + clip + "/meta.json");
        nlohmann::json meta = nlohmann::json::parse(mf);
        std::vector<int> ids = meta["input_ids"].get<std::vector<int>>();
        std::vector<int> gold = meta.contains("gen_ids") ? meta["gen_ids"].get<std::vector<int>>() : std::vector<int>();
        const bool have_ref = meta.contains("gen_ids") && meta.contains("top20_first3");

        qwen3_asr_payload_t payload;
        if (use_mel) {
            // mel.npy is [1][128][T] (feature extractor layout); the engine wants [T][128]
            qwen3_asr_cpu::npy_f32 mel = qwen3_asr_cpu::read_npy_f32(gdir + "/" + clip + "/mel.npy");
            const int bins = (int)mel.shape[1], T = (int)mel.shape[2];
            payload.mel_frames = T;
            payload.mel.resize((size_t)T * bins);
            for (int t = 0; t < T; t++) for (int b = 0; b < bins; b++) payload.mel[(size_t)t * bins + b] = mel.data[(size_t)b * T + t];
            payload.n_audio = qwen3_asr_audio_tokens(T);
        } else {
            qwen3_asr_cpu::npy_f32 enc = qwen3_asr_cpu::read_npy_f32(gdir + "/" + clip + "/enc.npy");
            payload.n_audio = (int)enc.shape[0];
            payload.audio_embeds.resize(enc.data.size());
            for (size_t i = 0; i < enc.data.size(); i++) payload.audio_embeds[i] = (bf16)enc.data[i];
        }

        engine.clear_context();
        auto t0 = std::chrono::steady_clock::now();
        buffer<bf16> logits = engine.prefill(ids, &payload);
        const double prefill_ms = ms_since(t0);

        std::vector<int> mine;
        double max_dlogit[3] = {0, 0, 0};
        int overlap[3] = {0, 0, 0};
        double decode_ms = 0;
        const int max_new = have_ref ? (int)gold.size() + 2 : 512;
        for (int step = 0; step < max_new; step++) {
            const int V = (int)cfg.get("vocab_size");
            std::vector<int> order(V);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 20, order.end(), [&](int a, int b) { return (float)logits[a] > (float)logits[b]; });
            if (have_ref && step < 3 && step < (int)meta["top20_first3"].size()) {
                auto gi = meta["top20_first3"][step]["ids"].get<std::vector<int>>();
                auto gv = meta["top20_first3"][step]["vals"].get<std::vector<double>>();
                for (int k = 0; k < 20; k++) {
                    if (std::find(order.begin(), order.begin() + 20, gi[k]) != order.begin() + 20) overlap[step]++;
                    max_dlogit[step] = std::max(max_dlogit[step], std::fabs((double)(float)logits[gi[k]] - gv[k]));
                }
            }
            // TRACE_STEPS: top 5 of the first 5 steps; TRACE_FROM=n: top 5 and the end-of-text logit of steps n .. n+4
            static const int trace_from = std::getenv("TRACE_FROM") ? std::atoi(std::getenv("TRACE_FROM")) : -1;
            if ((std::getenv("TRACE_STEPS") && step < 5) || (trace_from >= 0 && step >= trace_from && step < trace_from + 5)) {
                std::printf("  step %d top5:", step);
                for (int k = 0; k < 5; k++) std::printf("  %d(%.2f)", order[k], (float)logits[order[k]]);
                std::printf("  | eos %.2f\n", (float)logits[EOS0]);
            }
            const int tok = order[0];
            mine.push_back(tok);
            if (tok == EOS0 || tok == EOS1) break;
            auto s0 = std::chrono::steady_clock::now();
            logits = engine.forward(tok);
            decode_ms += ms_since(s0);
        }
        if (!std::getenv("NO_RELEASE")) engine.release_host_memory();   // as FLM does at the end of each request
        if (ids_out.is_open()) {
            ids_out << "{\"id\":\"" << clip << "\",\"gen\":[";
            for (size_t i = 0; i < mine.size(); i++) ids_out << (i ? "," : "") << mine[i];
            ids_out << "],\"prefill_ms\":" << prefill_ms << ",\"decode_ms_per_tok\":" << decode_ms / std::max<size_t>(1, mine.size() - 1) << "}\n";
            ids_out.flush();
        }
        const bool same = !have_ref || mine == gold;
        size_t fd = 0;
        while (fd < std::min(mine.size(), gold.size()) && mine[fd] == gold[fd]) fd++;
        char note[64] = "";
        if (!same) std::snprintf(note, sizeof note, ", first diff at %zu", fd);
        std::printf("%-18s prompt %4zu tok  prefill %7.1f ms  decode %6.1f ms/tok  tokens %s (%zu vs %zu%s)  top20 overlap %d/%d/%d  max|dlogit| %.3f/%.3f/%.3f\n",
                    clip.c_str(), ids.size(), prefill_ms, decode_ms / std::max<size_t>(1, mine.size() - 1), same ? "IDENTICAL" : "DIFFER",
                    mine.size(), gold.size(), note, overlap[0], overlap[1], overlap[2], max_dlogit[0], max_dlogit[1], max_dlogit[2]);
        bad += !same;
    }
    return bad ? 1 : 0;
}
