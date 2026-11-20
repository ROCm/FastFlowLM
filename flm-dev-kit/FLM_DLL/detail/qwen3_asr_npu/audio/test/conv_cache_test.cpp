// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

/// conv_cache_test: encode a growing recording step by step the way the streaming engine does (closed 8 s windows dropped, the
/// conv front-end cache kept), with and without the cache; the outputs must be identical. Build with -fsanitize=address to catch
/// out-of-bounds use of the cache.
///   conv_cache_test <audio_weight.q4nx> <goldens>/<clip>/mel.npy <step frames>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../qwen3_asr_audio_encoder.hpp"

static std::vector<float> read_mel(const std::string& path, int& T) {   // npy fp32 [1][128][T] -> [T][128]
    std::ifstream f(path, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t hl = (unsigned char)s[8] | ((unsigned char)s[9] << 8);
    const std::string hdr = s.substr(10, hl);
    const size_t a = hdr.find("(1, 128, ") + 9;
    T = std::stoi(hdr.substr(a));
    const float* d = (const float*)(s.data() + 10 + hl);
    std::vector<float> m((size_t)T * 128);
    for (int t = 0; t < T; t++)
        for (int b = 0; b < 128; b++) m[(size_t)t * 128 + b] = d[(size_t)b * T + t];
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    qwen3_asr_audio_encoder enc(argv[1]);
    int T = 0;
    const std::vector<float> mel = read_mel(argv[2], T);
    const int step = std::stoi(argv[3]);
    constexpr int WINDOW_FRAMES = 800, EDGE = 3;
    qwen3_asr_audio_encoder::conv_cache cache;
    int windows = 0, bad = 0, calls = 0;
    for (int frames = step; ; frames = std::min(frames + step, T)) {
        const int start = windows * WINDOW_FRAMES;
        const auto a = enc.encode(mel.data() + (size_t)start * 128, frames - start, &cache);
        const auto b = enc.encode(mel.data() + (size_t)start * 128, frames - start, nullptr);
        calls++;
        if (a != b) { bad++; std::printf("frames %d (from %d): cached != uncached\n", frames, start); }
        const int closed = std::max(0, (frames - EDGE) / WINDOW_FRAMES);
        if (closed > windows) {
            cache.drop_front((closed - windows) * (WINDOW_FRAMES / 100), 13 * 1024, 100 * 128);
            windows = closed;
        }
        if (frames == T) break;
    }
    std::printf("%d calls, %d differ, cache %d chunks\n", calls, bad, cache.chunks);
    return bad ? 1 : 0;
}
