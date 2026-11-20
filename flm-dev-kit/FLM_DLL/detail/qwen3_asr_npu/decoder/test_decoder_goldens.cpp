// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Checks the CPU decoder against the frozen BF16 PyTorch goldens (goldens/<clip>/{meta.json, enc.npy}).
//   ./test_decoder_goldens <model.q4nx> <goldens dir> <clip> [<clip> ...]
// Per clip: builds the prompt embeddings (text rows from the embedding table, <|audio_pad|> rows replaced by the
// reference encoder output), prefills, then decodes greedily and compares tokens and the first three logit vectors.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>

#include "nlohmann/json.hpp"
#include "npy_io.hpp"
#include "qwen3_asr_cpu_decoder.hpp"

using namespace qwen3_asr_cpu;

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s model.q4nx goldens_dir clip...\n", argv[0]); return 2; }
    const std::string model = argv[1], gdir = argv[2];
    constexpr int AUDIO_PAD = 151676, EOS0 = 151645, EOS1 = 151643;

    decoder dec(model);
    const int D = dec.cfg().hidden;
    int bad = 0;
    for (int ci = 3; ci < argc; ++ci) {
        const std::string clip = argv[ci];
        std::ifstream mf(gdir + "/" + clip + "/meta.json");
        nlohmann::json meta = nlohmann::json::parse(mf);
        npy_f32 enc = read_npy_f32(gdir + "/" + clip + "/enc.npy");
        std::vector<int> ids = meta["input_ids"].get<std::vector<int>>();
        std::vector<int> gold = meta["gen_ids"].get<std::vector<int>>();

        const int n = (int)ids.size();
        std::vector<float> emb((size_t)n * D);
        size_t audio_row = 0;
        for (int i = 0; i < n; ++i) {
            if (ids[i] == AUDIO_PAD) {
                if (audio_row >= enc.shape[0]) { std::fprintf(stderr, "%s: more audio_pad than encoder rows\n", clip.c_str()); return 1; }
                std::copy(&enc.data[audio_row * D], &enc.data[audio_row * D] + D, &emb[(size_t)i * D]);
                ++audio_row;
            } else {
                dec.embed(ids[i], &emb[(size_t)i * D]);
            }
        }
        if (audio_row != enc.shape[0]) { std::fprintf(stderr, "%s: audio rows %zu != encoder rows %zu\n", clip.c_str(), audio_row, enc.shape[0]); return 1; }

        dec.reset();
        std::vector<float> logits;
        auto t0 = std::chrono::steady_clock::now();
        dec.forward(emb.data(), n, logits);
        double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::vector<int> mine;
        double max_logit_diff[3] = {0, 0, 0};
        int top20_overlap[3] = {0, 0, 0};
        double decode_ms = 0;
        for (int step = 0; step < (int)gold.size() + 2; ++step) {
            std::vector<int> order(logits.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 20, order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            if (step < 3 && step < (int)meta["top20_first3"].size()) {
                auto gi = meta["top20_first3"][step]["ids"].get<std::vector<int>>();
                auto gv = meta["top20_first3"][step]["vals"].get<std::vector<double>>();
                for (int k = 0; k < 20; ++k) {
                    if (std::find(order.begin(), order.begin() + 20, gi[k]) != order.begin() + 20) ++top20_overlap[step];
                    max_logit_diff[step] = std::max(max_logit_diff[step], std::fabs((double)logits[gi[k]] - gv[k]));
                }
            }
            const int tok = order[0];
            mine.push_back(tok);
            if (tok == EOS0 || tok == EOS1) break;
            std::vector<float> e(D);
            dec.embed(tok, e.data());
            auto s0 = std::chrono::steady_clock::now();
            dec.forward(e.data(), 1, logits);
            decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count();
        }
        const bool same = mine == gold;
        size_t first_diff = 0;
        while (first_diff < std::min(mine.size(), gold.size()) && mine[first_diff] == gold[first_diff]) ++first_diff;
        char diff_note[64] = "";
        if (!same) std::snprintf(diff_note, sizeof diff_note, ", first diff at %zu", first_diff);
        std::printf("%-18s prompt %4d tok  prefill %7.1f ms  decode %5.1f ms/tok  tokens %s (%zu vs %zu%s)  top20 overlap %d/%d/%d  max|dlogit| %.3f/%.3f/%.3f\n",
                    clip.c_str(), n, prefill_ms, decode_ms / std::max<size_t>(1, mine.size() - 1), same ? "IDENTICAL" : "DIFFER",
                    mine.size(), gold.size(), diff_note, top20_overlap[0], top20_overlap[1], top20_overlap[2],
                    max_logit_diff[0], max_logit_diff[1], max_logit_diff[2]);
        bad += !same;
    }
    return bad ? 1 : 0;
}
