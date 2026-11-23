// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

// One decode layer run on the default build of layer_lm.xclbin (bf16 KV) and on the KV_BFP16 build (finished chunks
// from the bfp16 copy, gemma_bfp16_pack.hpp), with the same random inputs: the layer output x and the K/V row the
// layer writes must be the same bytes. Global and sliding layers, many context lengths. README "Test (test_layer_kv)".

#include <algorithm>
#include <chrono>
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

struct Options {
    std::string model_dir;
    std::string bf16_xclbin;
    std::string kvb_xclbin;
    std::string kvb_stub_xclbin;
    int iters = 5;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if (k == "--model") o.model_dir = v;
        else if (k == "--bf16") o.bf16_xclbin = v;
        else if (k == "--kvb") o.kvb_xclbin = v;
        else if (k == "--kvb-stub") o.kvb_stub_xclbin = v;
        else if (k == "--iters") o.iters = std::stoi(v);
        else { std::cerr << "unknown option " << k << std::endl; std::exit(2); }
    }
    if (o.model_dir.empty() || o.bf16_xclbin.empty() || o.kvb_xclbin.empty()) {
        std::cerr << "usage: test_layer_kv --model <dir with config.json> --bf16 <layer_lm.xclbin> --kvb <KV_BFP16 build>"
                     " [--kvb-stub <KV_BFP16 STUB build>] [--iters N]" << std::endl;
        std::exit(2);
    }
    return o;
}

uint32_t fnv1a(const std::vector<uint16_t>& v) {
    uint32_t h = 2166136261u;
    for (uint16_t x : v) {
        h = (h ^ (x & 0xFF)) * 16777619u;
        h = (h ^ (x >> 8)) * 16777619u;
    }
    return h;
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

void fill_normal(std::vector<uint16_t>& v, std::mt19937& gen, float sigma, float mean = 0.0f) {
    std::normal_distribution<float> dist(mean, sigma);
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

struct LayerData {
    std::vector<uint16_t> x;
    std::vector<uint8_t> w;
    std::vector<uint16_t> rms;
    std::vector<uint16_t> rope;
    std::vector<uint16_t> kv;   // bf16 regions + bfp16 copy, as one layer's kv buffer
};

struct Device {
    npu_app app;
    buffer<bf16> x, rms, rope, kv;
    buffer<u8> w;
};

void upload(Device& d, const LayerData& in) {
    memcpy(d.x.data(), in.x.data(), in.x.size() * 2);
    memcpy(d.w.data(), in.w.data(), in.w.size());
    memcpy(d.rms.data(), in.rms.data(), in.rms.size() * 2);
    memcpy(d.rope.data(), in.rope.data(), in.rope.size() * 2);
    memcpy(d.kv.data(), in.kv.data(), in.kv.size() * 2);
    d.x.sync_to_device();
    d.w.sync_to_device();
    d.rms.sync_to_device();
    d.rope.sync_to_device();
    d.kv.sync_to_device();
}

Device make_device(npu_app_manager* mgr, const LayerData& in) {
    Device d;
    d.app = mgr->create_app();
    d.x = d.app.create_bo_buffer<bf16>(in.x.size());
    d.w = d.app.create_bo_buffer<u8>(in.w.size());
    d.rms = d.app.create_bo_buffer<bf16>(in.rms.size());
    d.rope = d.app.create_bo_buffer<bf16>(in.rope.size());
    d.kv = d.app.create_bo_buffer<bf16>(in.kv.size());
    return d;
}

struct Result {
    std::vector<uint16_t> x;
    std::vector<uint16_t> new_rows;   // the row the layer wrote, in K01, K23, V01, V23
    double ms = 0.0;
};

Result run_layer(Device& d, gemma_npu_sequence& sequence, const LayerData& in, uint32_t L, bool sliding, bool kvb,
                 const size_t region_elems[4]) {
    upload(d, in);
    sequence.gen_layer_seq(d.app.seq(), L, sliding, kvb);
    auto run = d.app.create_run(d.x, d.w, d.rms, d.rope, d.kv);
    auto t0 = std::chrono::steady_clock::now();
    run.start();
    flm::npu::wait_for_completion(run, "layer");
    Result r;
    r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    d.x.sync_from_device();
    d.kv.sync_from_device();
    r.x.assign(reinterpret_cast<uint16_t*>(d.x.data()), reinterpret_cast<uint16_t*>(d.x.data()) + in.x.size());
    const uint32_t row = sliding ? (L - 1) % WINDOW : L - 1;
    const uint16_t* kv = reinterpret_cast<const uint16_t*>(d.kv.data());
    for (int s = 0; s < 4; s++) {
        const uint16_t* p = kv + region_elems[s] + (size_t)row * 512;
        r.new_rows.insert(r.new_rows.end(), p, p + 512);
    }
    return r;
}

double time_layer(Device& d, gemma_npu_sequence& sequence, const LayerData& in, uint32_t L, bool sliding, bool kvb,
                  int iters) {
    upload(d, in);
    sequence.gen_layer_seq(d.app.seq(), L, sliding, kvb);
    auto run = d.app.create_run(d.x, d.w, d.rms, d.rope, d.kv);
    std::vector<double> ms;
    for (int i = 0; i < iters + 1; i++) {
        auto t0 = std::chrono::steady_clock::now();
        run.start();
        flm::npu::wait_for_completion(run, "layer");
        if (i > 0) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    LM_Config config;
    {
        std::ifstream cf(o.model_dir + "/config.json");
        config._json_config = nlohmann::json::parse(cf);
    }
    const size_t D = config.get("hidden_size");
    const size_t DH = config.get("head_dim");
    const size_t DQ = DH * config.get("num_attention_heads");
    const size_t DKV = DH * config.get("num_key_value_heads");
    const size_t FF = config.get("intermediate_size");
    gemma_npu_sequence sequence(config, MAX_L);

    std::mt19937 gen(20261123);
    LayerData data[2];   // 0: global layer, 1: sliding layer
    for (int t = 0; t < 2; t++) {
        const bool sliding = t == 1;
        LayerData& in = data[t];
        in.x.resize(D);
        fill_normal(in.x, gen, 1.0f);
        in.w.resize((DQ * D + 2 * DKV * D + DQ * D + 3 * FF * D) * 5 / 8);
        fill_weights(in.w, gen);
        in.rms.resize(4 * D);
        fill_normal(in.rms, gen, 0.1f);
        in.rope.resize(3 * DH);
        std::uniform_real_distribution<float> angle(-3.14159f, 3.14159f);
        for (size_t j = 0; j < DH / 2; j++) {
            const float a = angle(gen);
            in.rope[j] = bf16_bits(std::cos(a));
            in.rope[DH / 2 + j] = bf16_bits(std::sin(a));
        }
        std::vector<uint16_t> norms(2 * DH);
        fill_normal(norms, gen, 0.1f);
        std::copy(norms.begin(), norms.end(), in.rope.begin() + DH);
        in.kv.resize(sequence.get_kv_bfp16_offset(sliding, 4));
        const size_t cap = sliding ? WINDOW : MAX_L;
        std::vector<uint16_t> bf16_part(cap * 2 * DKV);
        fill_normal(bf16_part, gen, 1.0f);
        std::copy(bf16_part.begin(), bf16_part.end(), in.kv.begin());
        for (int s = 0; s < 4; s++) {
            const bf16* rows = reinterpret_cast<const bf16*>(in.kv.data()) + s * cap * 512;
            uint8_t* units = reinterpret_cast<uint8_t*>(in.kv.data() + sequence.get_kv_bfp16_offset(sliding, s));
            for (size_t c = 0; c < cap / CHUNK; c++) {
                gemma_npu_bfp16::pack_stream_chunk(units + c * gemma_npu_bfp16::unit_bytes, rows + c * CHUNK * 512, s >= 2);
            }
        }
    }

    flm_rt::device device(0u);
    npu_xclbin_manager npu(npu_device::device_npu2, &device, false);
    npu_app_manager* bf16_mgr = npu.register_xclbin(o.bf16_xclbin);
    npu_app_manager* kvb_mgr = npu.register_xclbin(o.kvb_xclbin);
    Device dev[2][2] = {{make_device(bf16_mgr, data[0]), make_device(bf16_mgr, data[1])},
                        {make_device(kvb_mgr, data[0]), make_device(kvb_mgr, data[1])}};

    const std::vector<uint32_t> global_L = {1, 2, 15, 16, 17, 31, 32, 33, 100, 1023, 1024, 1025, 1040, 1041,
                                            2000, 4096, 4097, 8000, 8191, 8192};
    const std::vector<uint32_t> sliding_L = {1, 15, 16, 17, 100, 1023, 1024, 1025, 1039, 1040, 1041, 1100,
                                             2047, 2048, 2049, 2063, 5000};
    bool all_same = true;
    for (int t = 0; t < 2; t++) {
        const bool sliding = t == 1;
        const size_t cap = sliding ? WINDOW : MAX_L;
        const size_t regions[4] = {0, cap * 512, 2 * cap * 512, 3 * cap * 512};
        for (uint32_t L : sliding ? sliding_L : global_L) {
            Result a = run_layer(dev[0][t], sequence, data[t], L, sliding, false, regions);
            Result b = run_layer(dev[1][t], sequence, data[t], L, sliding, true, regions);
            const bool x_same = a.x == b.x;
            const bool rows_same = a.new_rows == b.new_rows;
            all_same = all_same && x_same && rows_same;
            std::cout << (sliding ? "sliding" : "global ") << " L " << L << ": x " << (x_same ? "same" : "DIFFERENT")
                      << ", new K/V row " << (rows_same ? "same" : "DIFFERENT") << " (x hash " << std::hex
                      << fnv1a(b.x) << std::dec << ")" << std::endl;
        }
    }

    std::cout << "time ms per layer run (median of " << o.iters << "): bf16 build / KV_BFP16 build";
    Device stub[2];
    const bool with_stub = !o.kvb_stub_xclbin.empty();
    if (with_stub) {
        npu_app_manager* stub_mgr = npu.register_xclbin(o.kvb_stub_xclbin);
        stub[0] = make_device(stub_mgr, data[0]);
        stub[1] = make_device(stub_mgr, data[1]);
        std::cout << " / KV_BFP16 stub (no math)";
    }
    std::cout << std::endl;
    const std::vector<std::pair<bool, uint32_t>> timed = {{false, 1088}, {false, 4160}, {false, 8192}, {true, 1088}};
    for (auto [sliding, L] : timed) {
        const int t = sliding ? 1 : 0;
        std::cout << "  " << (sliding ? "sliding" : "global ") << " L " << L << ": "
                  << time_layer(dev[0][t], sequence, data[t], L, sliding, false, o.iters) << " / "
                  << time_layer(dev[1][t], sequence, data[t], L, sliding, true, o.iters);
        if (with_stub) std::cout << " / " << time_layer(stub[t], sequence, data[t], L, sliding, true, o.iters);
        std::cout << std::endl;
    }
    std::cout << (all_same ? "PASSED" : "FAILED") << ": KV_BFP16 build gives the same bytes as the bf16 build" << std::endl;
    return all_same ? 0 : 1;
}
