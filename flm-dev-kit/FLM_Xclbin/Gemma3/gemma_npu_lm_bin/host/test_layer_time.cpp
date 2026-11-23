// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// Timing only: one decode layer run of layer_lm.xclbin (global and sliding sequences) at several context lengths,
// random inputs (Q4_1 weight blocks, norms, rope tables, x, a random K/V cache and its bfp16 copy). Use with the
// normal and the STUB=1 (no math) build to split a layer's time (README "Layer time split").

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/gemma/gemma_npu_sequence.hpp"
#include "npu_utils/npu_utils.hpp"
#include "npu_utils/runtime_wait.hpp"
#include "gemma_bfp16_pack.hpp"

namespace {

constexpr int BLOCK_BYTES = 32 * 256 * 5 / 8;
constexpr uint32_t MAX_L = 8192;
constexpr uint32_t WINDOW = 1024;
constexpr int CHUNK = 16;

uint16_t bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

void fill_normal(std::vector<uint16_t>& v, std::mt19937& gen, float sigma) {
    std::normal_distribution<float> dist(0.0f, sigma);
    for (auto& x : v) x = bf16_bits(dist(gen));
}

// Q4_1 blocks: 32x8 bf16 scales, 32x8 bf16 mins, 32x256 4-bit codes; w = q * scale + min stays near zero
void fill_weights(std::vector<uint8_t>& w, std::mt19937& gen) {
    std::uniform_real_distribution<float> scale_dist(0.002f, 0.02f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (size_t b = 0; b + BLOCK_BYTES <= w.size(); b += BLOCK_BYTES) {
        uint16_t* scales = reinterpret_cast<uint16_t*>(w.data() + b);
        uint16_t* mins = scales + 256;
        for (int i = 0; i < 256; i++) {
            const float s = scale_dist(gen);
            scales[i] = bf16_bits(s);
            mins[i] = bf16_bits(-7.5f * s);
        }
        for (size_t i = b + 1024; i < b + BLOCK_BYTES; i++) w[i] = (uint8_t)byte_dist(gen);
    }
}

struct Layer {
    npu_app app;
    buffer<bf16> x, rms, rope, kv;
    buffer<u8> w;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir, xclbin;
    int iters = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if (k == "--model") model_dir = v;
        else if (k == "--xclbin") xclbin = v;
        else if (k == "--iters") iters = std::stoi(v);
        else { std::cerr << "unknown option " << k << std::endl; return 2; }
    }
    if (model_dir.empty() || xclbin.empty()) {
        std::cerr << "usage: test_layer_time --model <dir with config.json> --xclbin <layer_lm.xclbin> [--iters N]" << std::endl;
        return 2;
    }
    LM_Config config;
    {
        std::ifstream cf(model_dir + "/config.json");
        config._json_config = nlohmann::json::parse(cf);
    }
    const size_t D = config.get("hidden_size");
    const size_t DH = config.get("head_dim");
    const size_t DQ = DH * config.get("num_attention_heads");
    const size_t DKV = DH * config.get("num_key_value_heads");
    const size_t FF = config.get("intermediate_size");
    gemma_npu_sequence sequence(config, MAX_L);

    flm_rt::device device(0u);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* mgr = npu.register_xclbin(xclbin);

    std::mt19937 gen(20261123);
    std::cout << "layer run, median of " << iters << " (ms); weights "
              << (DQ * D + 2 * DKV * D + DQ * D + 3 * FF * D) * 5 / 8 / 1e6 << " MB" << std::endl;
    for (int t = 0; t < 2; t++) {
        const bool sliding = t == 1;
        const size_t cap = sliding ? WINDOW : MAX_L;
        Layer l;
        l.app = mgr->create_app();
        std::vector<uint16_t> x(D), rms(4 * D), rope(3 * DH);
        std::vector<uint8_t> w((DQ * D + 2 * DKV * D + DQ * D + 3 * FF * D) * 5 / 8);
        fill_normal(x, gen, 1.0f);
        fill_normal(rms, gen, 0.1f);
        fill_weights(w, gen);
        std::uniform_real_distribution<float> angle(-3.14159f, 3.14159f);
        for (size_t j = 0; j < DH / 2; j++) {
            const float a = angle(gen);
            rope[j] = bf16_bits(std::cos(a));
            rope[DH / 2 + j] = bf16_bits(std::sin(a));
        }
        std::vector<uint16_t> norms(2 * DH);
        fill_normal(norms, gen, 0.1f);
        std::copy(norms.begin(), norms.end(), rope.begin() + DH);
        std::vector<uint16_t> kv(sequence.get_kv_bfp16_offset(sliding, 4));
        std::vector<uint16_t> rows(cap * 2 * DKV);
        fill_normal(rows, gen, 1.0f);
        std::copy(rows.begin(), rows.end(), kv.begin());
        for (int s = 0; s < 4; s++) {
            const bf16* r = reinterpret_cast<const bf16*>(kv.data()) + s * cap * 512;
            uint8_t* units = reinterpret_cast<uint8_t*>(kv.data() + sequence.get_kv_bfp16_offset(sliding, s));
            for (size_t c = 0; c < cap / CHUNK; c++) {
                gemma_npu_bfp16::pack_stream_chunk(units + c * gemma_npu_bfp16::unit_bytes, r + c * CHUNK * 512, s >= 2);
            }
        }
        l.x = l.app.create_bo_buffer<bf16>(x.size());
        l.w = l.app.create_bo_buffer<u8>(w.size());
        l.rms = l.app.create_bo_buffer<bf16>(rms.size());
        l.rope = l.app.create_bo_buffer<bf16>(rope.size());
        l.kv = l.app.create_bo_buffer<bf16>(kv.size());
        memcpy(l.x.data(), x.data(), x.size() * 2);
        memcpy(l.w.data(), w.data(), w.size());
        memcpy(l.rms.data(), rms.data(), rms.size() * 2);
        memcpy(l.rope.data(), rope.data(), rope.size() * 2);
        memcpy(l.kv.data(), kv.data(), kv.size() * 2);
        l.x.sync_to_device();
        l.w.sync_to_device();
        l.rms.sync_to_device();
        l.rope.sync_to_device();
        l.kv.sync_to_device();

        const std::vector<uint32_t> lengths = {16, 1088, 4160};
        for (uint32_t L : lengths) {
            sequence.gen_layer_seq(l.app.seq(), L, sliding);
            auto run = l.app.create_run(l.x, l.w, l.rms, l.rope, l.kv);
            std::vector<double> ms;
            for (int i = 0; i < iters + 2; i++) {
                auto t0 = std::chrono::steady_clock::now();
                run.start();
                flm::npu::wait_for_completion(run, "layer");
                if (i >= 2) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::cout << "  " << (sliding ? "sliding" : "global ") << " L " << L << ": " << median(ms) << " (min "
                      << *std::min_element(ms.begin(), ms.end()) << ")" << std::endl;
        }
    }
    return 0;
}
