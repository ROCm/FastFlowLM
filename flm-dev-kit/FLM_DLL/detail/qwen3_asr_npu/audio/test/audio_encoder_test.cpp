// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// Standalone accuracy / speed test of qwen3_asr_audio_encoder against the frozen R2T2 goldens.
///   ./audio_encoder_test <audio_weight.q4nx> <goldens dir> clip [clip ...]
/// per clip: goldens/<clip>/mel.npy (fp32 [1,128,T]) and enc.npy (fp32 [N,2048], the patched BF16 reference output).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../qwen3_asr_audio_encoder.hpp"

static std::vector<float> load_npy(const std::string& path, std::vector<long>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    unsigned char ver[2];
    f.read((char*)ver, 2);
    uint32_t hl = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char*)&h, 2); hl = h; } else { f.read((char*)&hl, 4); }
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    if (hdr.find("'<f4'") == std::string::npos || hdr.find("True") != std::string::npos) throw std::runtime_error("expect C-order <f4: " + hdr);
    auto p = hdr.find("'shape': (");
    shape.clear();
    size_t i = p + 10;
    while (hdr[i] != ')') {
        if (isdigit(hdr[i])) { long v = 0; while (isdigit(hdr[i])) v = v * 10 + (hdr[i++] - '0'); shape.push_back(v); } else i++;
    }
    size_t n = 1;
    for (auto d : shape) n *= d;
    std::vector<float> data(n);
    f.read((char*)data.data(), n * 4);
    return data;
}

static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s audio_weight.q4nx goldens_dir clip...\n", argv[0]); return 2; }
    auto t0 = std::chrono::steady_clock::now();
    qwen3_asr_audio_encoder enc(argv[1]);
    std::printf("load %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const int D = 2048, WT = enc.config().window_tokens();
    int fails = 0;
    std::printf("%-22s %6s %6s %9s %9s %9s %9s %10s %9s\n", "clip", "sec", "tok", "cos", "mincos_tk", "max|d|", "relL2", "worstwin", "ms/s");
    for (int a = 3; a < argc; a++) {
        std::string clip = argv[a], dir = std::string(argv[2]) + "/" + clip;
        std::vector<long> ms, es;
        auto mel_c = load_npy(dir + "/mel.npy", ms);      // [1,128,T]
        auto ref = load_npy(dir + "/enc.npy", es);        // [N,2048]
        const int T = (int)ms[2], F = (int)ms[1], N = (int)es[0];
        std::vector<float> mel((size_t)T * F);
        for (int f = 0; f < F; f++) for (int t = 0; t < T; t++) mel[(size_t)t * F + f] = mel_c[(size_t)f * T + t];
        if (const char* sd = std::getenv("STAGE_DIR")) {  // optional: per-stage activations through the public seams
            const auto& c = enc.config();
            const int Dm = c.d_model, CF = c.chunk_frames(), ntok = qwen3_asr_audio_tokens(T);
            std::vector<float> x((size_t)ntok * Dm);
            int off = 0;
            for (int ch = 0; ch * CF < T; ch++) {
                int valid = std::min(CF, T - ch * CF), nt = qwen3_asr_audio_tokens(valid);
                enc.conv_front(mel.data() + (size_t)ch * CF * c.num_mel_bins, valid, x.data() + (size_t)off * Dm);
                std::ofstream(std::string(sd) + "/" + clip + ".conv" + std::to_string(ch) + ".f32", std::ios::binary).write((const char*)(x.data() + (size_t)off * Dm), (size_t)nt * Dm * 4);
                enc.add_pos(x.data() + (size_t)off * Dm, nt);
                off += nt;
            }
            for (int l = -1; l < c.layers; l++) {
                if (l >= 0) enc.encoder_layer(l, x.data(), ntok);
                std::ofstream(std::string(sd) + "/" + clip + ".L" + std::to_string(l + 1) + ".f32", std::ios::binary).write((const char*)x.data(), x.size() * 4);
            }
        }
        enc.encode(mel.data(), T);  // warm up
        auto s0 = std::chrono::steady_clock::now();
        auto out = enc.encode(mel.data(), T);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        const int n = (int)(out.size() / D);
        if (n != N) { std::printf("%-22s TOKEN COUNT MISMATCH got %d ref %d\n", clip.c_str(), n, N); fails++; continue; }
        if (const char* dd = std::getenv("DUMP_DIR")) {   // optional: raw fp32 output for external comparison
            std::vector<float> o32(out.size());
            for (size_t i = 0; i < out.size(); i++) o32[i] = bf2f(out[i]);
            std::ofstream(std::string(dd) + "/" + clip + ".f32", std::ios::binary).write((const char*)o32.data(), o32.size() * 4);
        }
        double dot = 0, na = 0, nb = 0, nd = 0, maxd = 0, mincos = 1;
        std::vector<double> wnd_l2n, wnd_l2d;
        for (int i = 0; i < n; i++) {
            double d_ = 0, a2 = 0, b2 = 0;
            for (int j = 0; j < D; j++) {
                double x = bf2f(out[(size_t)i * D + j]), y = ref[(size_t)i * D + j], d = x - y;
                d_ += x * y; a2 += x * x; b2 += y * y; nd += d * d; maxd = std::max(maxd, std::fabs(d));
            }
            dot += d_; na += a2; nb += b2;
            mincos = std::min(mincos, d_ / std::sqrt(a2 * b2 + 1e-30));
            const int w = i / WT;
            if ((int)wnd_l2n.size() <= w) { wnd_l2n.push_back(0); wnd_l2d.push_back(0); }
            double dd = 0; for (int j = 0; j < D; j++) { double d = bf2f(out[(size_t)i * D + j]) - ref[(size_t)i * D + j]; dd += d * d; }
            wnd_l2n[w] += dd; wnd_l2d[w] += b2;
        }
        double cos = dot / std::sqrt(na * nb), rel = std::sqrt(nd / nb), worst = 0;
        for (size_t w = 0; w < wnd_l2n.size(); w++) worst = std::max(worst, std::sqrt(wnd_l2n[w] / wnd_l2d[w]));
        const double audio_s = T / 100.0;
        // min per-token cosine is informational (a loose 0.80 floor): one token per clip can flip at the massive-activation channels in
        // layers >= 17, see README; the gates are the global cosine, the relative L2 (BF16 noise floor is 1-7 %) and the token count
        bool ok = cos > 0.999 && rel < 0.05 && mincos > 0.80;
        if (!ok) fails++;
        std::printf("%-22s %6.1f %6d %9.6f %9.5f %9.4f %9.5f %10.5f %9.1f %s\n", clip.c_str(), audio_s, n, cos, mincos, maxd, rel, worst,
                    1000.0 * sec / audio_s, ok ? "" : "FAIL");
    }
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS (token counts equal, cos > 0.999, relL2 < 0.05, min per-token cos > 0.80)");
    return fails ? 1 : 0;
}
