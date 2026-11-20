// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

#include <bits/stdc++.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdfloat>
#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"

namespace po = boost::program_options;

// Qwen3.5-2B full-attention layer: 8 q heads, 2 kv heads (GQA 4:1), head dim 256.
static constexpr int Heads = 8;
static constexpr int KV_heads = 2;
static constexpr int DH = 256;

static uint64_t fnv1a(const void* data, size_t bytes) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// 8 bf16 values (stride apart) -> one bfp16 block: largest exponent byte + 8 two's complement mantissas, truncated
// like the AIE2P conversion: q = signed 24-bit mantissa >> (17 + max_exp - exp), the shift capped at 31.
static void bfp16_block(uint8_t* out, const bf16* v, size_t stride) {
    uint32_t bits[8];
    uint32_t max_exp = 0;
    for (int i = 0; i < 8; i++) {
        uint16_t b;
        memcpy(&b, v + i * stride, 2);
        bits[i] = (uint32_t)b << 16;
        max_exp = std::max(max_exp, (bits[i] >> 23) & 0xFF);
    }
    out[0] = (uint8_t)max_exp;
    for (int i = 0; i < 8; i++) {
        const uint32_t exp = (bits[i] >> 23) & 0xFF;
        int32_t mantissa = (int32_t)((bits[i] & 0x7FFFFF) | (exp ? 0x800000 : 0));
        if (bits[i] >> 31) mantissa = -mantissa;
        const int shift = std::min<int>(17 + max_exp - exp, 31);
        out[1 + i] = (uint8_t)(mantissa >> shift);
    }
}

// The kernel's inputs (all tiles 8 x 8, 72 bytes, rows of 16 = 2 tiles stacked):
//   q    [head][L/16][dh/8][2][72]      a tile row = 8 dh of one query row
//   K    [kv head][MAX_L/16][dh/8][2][72]  a tile row = 8 dh of one key
//   V^T  [kv head][MAX_L/16][dh/8][2][72]  a tile row = 8 keys of one dh column
static void gen_mha_bfp16_layout(uint8_t* q_out, uint8_t* kv_out, const bf16* q, const bf16* kv, int L, int MAX_L, size_t kv_cache_size) {
    const int DQ = Heads * DH;
    const int DKV = KV_heads * DH;
    const size_t row_bytes = DH * 9 / 8;
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < Heads; h++)
        for (int r = 0; r < L; r++)
            for (int kb = 0; kb < DH / 8; kb++)
                bfp16_block(q_out + (size_t)h * L * row_bytes + (((size_t)(r / 16) * (DH / 8) + kb) * 2 + (r % 16) / 8) * 72 + (r % 8) * 9,
                            q + (size_t)r * DQ + h * DH + kb * 8, 1);
    const bf16* v = kv + kv_cache_size / 2;
    uint8_t* v_out = kv_out + (size_t)KV_heads * MAX_L * row_bytes;
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < KV_heads; h++)
        for (int t = 0; t < MAX_L; t++)
            for (int kb = 0; kb < DH / 8; kb++)
                bfp16_block(kv_out + (size_t)h * MAX_L * row_bytes + (((size_t)(t / 16) * (DH / 8) + kb) * 2 + (t % 16) / 8) * 72 + (t % 8) * 9,
                            kv + (size_t)t * DKV + h * DH + kb * 8, 1);
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < KV_heads; h++)
        for (int t = 0; t < MAX_L; t += 8)
            for (int d = 0; d < DH; d++)
                // chunk-major: per 128-key block, 8 chunks of 4 dh blocks, [pair][key tile][V00 V01 V10 V11] (see attn_kernel.h)
                bfp16_block(v_out + (size_t)h * MAX_L * row_bytes + ((size_t)(t / 128) * 512 + (size_t)((d / 8) / 4) * 64 + (size_t)(((d / 8) % 4) / 2) * 32 + (size_t)((t % 128) / 16) * 4 + (size_t)((d / 8) % 2) * 2 + (t % 16) / 8) * 72 + (d % 8) * 9,
                            v + (size_t)t * DKV + h * DH + d, DKV);
}

// fp32 causal attention for one query row of one head; keys 0 .. pos.
static void cpu_attention_row(float* out, const bf16* q_row, const bf16* k_cache, const bf16* v_cache, int pos, int kv_head) {
    const int kv_stride = KV_heads * DH;
    std::vector<float> score(pos + 1);
    float max_score = -INFINITY;
    for (int t = 0; t <= pos; t++) {
        const bf16* k_row = k_cache + (size_t)t * kv_stride + kv_head * DH;
        float dot = 0.0f;
        for (int d = 0; d < DH; d++) dot += (float)q_row[d] * (float)k_row[d];
        score[t] = dot / 16.0f;
        max_score = std::max(max_score, score[t]);
    }
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        score[t] = std::exp(score[t] - max_score);
        sum += score[t];
    }
    for (int d = 0; d < DH; d++) out[d] = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const bf16* v_row = v_cache + (size_t)t * kv_stride + kv_head * DH;
        const float p = score[t] / sum;
        for (int d = 0; d < DH; d++) out[d] += p * (float)v_row[d];
    }
}

int main(int argc, char **argv) {
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);
    desc.add_options()("len,L", po::value<int>()->default_value(10752), "query rows (prompt length, a multiple of 256)");
    desc.add_options()("iters,i", po::value<int>()->default_value(3), "timed calls (after 1 warmup)");
    desc.add_options()("check,c", po::value<int>()->default_value(48), "query rows checked against the CPU (all heads)");
    desc.add_options()("xclbin,x", po::value<std::string>()->default_value("build/final/xclbins/attn.xclbin"), "xclbin");
    desc.add_options()("begin,B", po::value<int>()->default_value(0), "rows already in the KV cache before the query rows (multi-turn)");
    arg_utils::parse_options(argc, argv, desc, vm);

    const int L = vm["len"].as<int>();
    const int iters = vm["iters"].as<int>();
    const int check_rows = vm["check"].as<int>();
    assert(L % 256 == 0);
    const int L_begin = vm["begin"].as<int>();
    const int MAX_L = (L_begin + L + 127) / 128 * 128;
    const int DQ = Heads * DH;
    const int DKV = KV_heads * DH;
    const size_t kv_cache_size = (size_t)2 * MAX_L * DKV;

    xrt::device npu_device = xrt::device(0);
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_mgr = xclbin_mgr.register_xclbin(vm["xclbin"].as<std::string>());
    npu_app app = attn_mgr->create_app();
    gen_mha_engine_seq(app.seq(), L_begin, L, MAX_L, Heads, KV_heads, DH);

    buffer<bf16> o = app.create_bo_buffer<bf16>((size_t)L * DQ);
    buffer<bf16> q((size_t)L * DQ);
    buffer<bf16> kv(kv_cache_size);
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < q.size(); i++) q[i] = bf16(dist(rng));
        for (size_t i = 0; i < kv.size(); i++) kv[i] = bf16(dist(rng));
    }
    o.memset(bf16(0.0f));
    o.sync_to_device();

    buffer<uint8_t> qb = app.create_bo_buffer<uint8_t>((size_t)L * Heads * DH * 9 / 8);
    buffer<uint8_t> kvb = app.create_bo_buffer<uint8_t>((size_t)2 * KV_heads * MAX_L * DH * 9 / 8);
    gen_mha_bfp16_layout(qb.data(), kvb.data(), q.data(), kv.data(), L, MAX_L, kv_cache_size);
    qb.sync_to_device();
    kvb.sync_to_device();
    auto call = [&]() { app(o, qb, kvb); };

    call();
    double total_us = 0.0;
    std::ostringstream call_times;
    for (int i = 0; i < iters; i++) {
        time_utils::time_point start = time_utils::now();
        call();
        time_utils::time_point stop = time_utils::now();
        double us = time_utils::duration_us(start, stop).first;
        total_us += us;
        call_times << " " << (int)us;
    }
    o.sync_from_device();
    header_print("attn", "call times (us):" << call_times.str());

    std::vector<int> rows = {0, 1, 15, 16, 127, 128, L - 1};
    std::mt19937 rng(2);
    std::uniform_int_distribution<int> pick(0, L - 1);
    while ((int)rows.size() < check_rows) rows.push_back(pick(rng));
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    const int R = rows.size();
    buffer<bf16> got((size_t)R * DQ);
    buffer<bf16> want((size_t)R * DQ);
    const bf16* k_cache = kv.data();
    const bf16* v_cache = kv.data() + kv_cache_size / 2;
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < R; r++) {
        std::vector<float> out(DH);
        for (int h = 0; h < Heads; h++) {
            cpu_attention_row(out.data(), q.data() + (size_t)rows[r] * DQ + h * DH, k_cache, v_cache, L_begin + rows[r], h / (Heads / KV_heads));
            for (int d = 0; d < DH; d++) {
                want[(size_t)r * DQ + h * DH + d] = bf16(out[d]);
                got[(size_t)r * DQ + h * DH + d] = o[(size_t)rows[r] * DQ + h * DH + d];
            }
        }
    }
    print_error_metrics(get_error_metrics(got, want), "vs fp32 CPU (" + std::to_string(R) + " rows x 8 heads)");

    const double causal_macs = 2.0 * Heads * DH * ((double)L * (L + 1) / 2.0);
    const double us = total_us / iters;
    std::cout << "SUMMARY L=" << L << " us/call=" << std::fixed << std::setprecision(0) << us
              << " useful TMAC/s=" << std::setprecision(2) << causal_macs / (us * 1e-6) / 1e12
              << " hash=" << std::hex << fnv1a(o.data(), o.size() * sizeof(bf16)) << std::dec << std::endl;
    return 0;
}
