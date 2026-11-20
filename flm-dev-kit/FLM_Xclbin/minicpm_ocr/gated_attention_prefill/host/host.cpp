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
#include "seq_gen_old.hpp"

namespace po = boost::program_options;

// Qwen3.5-2B / MiniCPM-OCR full-attention layer (same shape): 8 q heads, 2 kv heads (GQA 4:1), head dim 256.
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
                bfp16_block(v_out + (size_t)h * MAX_L * row_bytes + (((size_t)(t / 16) * (DH / 8) + d / 8) * 2 + (t % 16) / 8) * 72 + (d % 8) * 9,
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

// float32 2-D .npy (the engine's FLM_DUMP_DIR dumps) -> rows x cols
static std::vector<float> load_npy_2d(const std::string& path, size_t& rows, size_t& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    uint16_t header_len = 0;
    f.read((char*)&header_len, 2);
    std::string header(header_len, ' ');
    f.read(header.data(), header_len);
    if (header.find("'<f4'") == std::string::npos) throw std::runtime_error(path + ": not float32");
    const size_t open = header.find("'shape': (");
    if (sscanf(header.c_str() + open + 10, "%zu, %zu", &rows, &cols) != 2) throw std::runtime_error(path + ": not 2-D");
    std::vector<float> data(rows * cols);
    f.read((char*)data.data(), data.size() * sizeof(float));
    return data;
}

// a vs b over n bf16 values: how many are bit-identical, cos, rel L2 (vs b), max |a - b|
static void compare_outputs(const char* name, const bf16* a, const bf16* b, size_t n) {
    size_t same = 0;
    double ab = 0, aa = 0, bb = 0, dd = 0, max_abs = 0;
    for (size_t i = 0; i < n; i++) {
        uint16_t x, y;
        memcpy(&x, a + i, 2);
        memcpy(&y, b + i, 2);
        same += (x == y);
        const double fa = (float)a[i];
        const double fb = (float)b[i];
        ab += fa * fb;
        aa += fa * fa;
        bb += fb * fb;
        dd += (fa - fb) * (fa - fb);
        max_abs = std::max(max_abs, std::abs(fa - fb));
    }
    std::cout << "COMPARE " << name << ": identical " << same << " / " << n << (same == n ? " (bit-identical)" : "")
              << std::setprecision(8) << " cos=" << ab / std::sqrt(aa * bb) << " relL2=" << std::sqrt(dd / bb)
              << " max_abs=" << max_abs << std::endl;
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
    desc.add_options()("old", po::value<std::string>()->default_value(""), "also run this xclbin (the shipped bf16 attn.xclbin) on the same data and compare");
    desc.add_options()("npy", po::value<std::string>()->default_value(""), "real data: a dump folder with attn_q_rope / attn_k_rope / attn_v .npy (FLM_DUMP_DIR, layers/NN); L is cut to fit");
    arg_utils::parse_options(argc, argv, desc, vm);

    int L = vm["len"].as<int>();
    const int iters = vm["iters"].as<int>();
    const int check_rows = vm["check"].as<int>();
    const int L_begin = vm["begin"].as<int>();
    const std::string npy_dir = vm["npy"].as<std::string>();
    size_t npy_rows = 0;
    std::vector<float> npy_q, npy_k, npy_v;
    if (!npy_dir.empty()) {
        size_t cols = 0;
        npy_q = load_npy_2d(npy_dir + "/attn_q_rope.npy", npy_rows, cols);
        assert(cols == (size_t)Heads * DH);
        npy_k = load_npy_2d(npy_dir + "/attn_k_rope.npy", npy_rows, cols);
        assert(cols == (size_t)KV_heads * DH);
        npy_v = load_npy_2d(npy_dir + "/attn_v.npy", npy_rows, cols);
        assert(cols == (size_t)KV_heads * DH);
        L = std::min(L, ((int)npy_rows - L_begin) / 256 * 256);
        std::cout << "real data: " << npy_rows << " rows from " << npy_dir << ", L_begin " << L_begin << ", L " << L << std::endl;
    }
    assert(L > 0 && L % 256 == 0);
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
    if (!npy_dir.empty()) {
        // query rows L_begin .. L_begin + L of the dump; K / V rows 0 .. npy_rows (the rest stay random, as stale
        // cache rows would be)
        for (size_t i = 0; i < q.size(); i++) q[i] = bf16(npy_q[(size_t)L_begin * DQ + i]);
        for (size_t r = 0; r < std::min(npy_rows, (size_t)MAX_L); r++) {
            for (int c = 0; c < DKV; c++) {
                kv[r * DKV + c] = bf16(npy_k[r * DKV + c]);
                kv[kv_cache_size / 2 + r * DKV + c] = bf16(npy_v[r * DKV + c]);
            }
        }
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

    // the shipped kernel on the same q / KV, bf16 in, 16 rows per core
    const std::string old_xclbin = vm["old"].as<std::string>();
    buffer<bf16> o_old;
    if (!old_xclbin.empty()) {
        npu_app_manager* old_mgr = xclbin_mgr.register_xclbin(old_xclbin);
        npu_app old_app = old_mgr->create_app();
        gen_mha_engine_seq_old(old_app.seq(), L_begin, L, MAX_L, Heads, KV_heads, DH, kv_cache_size);
        o_old = old_app.create_bo_buffer<bf16>((size_t)L * DQ);
        buffer<bf16> q_old = old_app.create_bo_buffer<bf16>((size_t)L * DQ);
        buffer<bf16> kv_old = old_app.create_bo_buffer<bf16>(kv_cache_size);
        memcpy(q_old.data(), q.data(), q.size() * sizeof(bf16));
        memcpy(kv_old.data(), kv.data(), kv.size() * sizeof(bf16));
        q_old.sync_to_device();
        kv_old.sync_to_device();
        o_old.memset(bf16(0.0f));
        o_old.sync_to_device();
        old_app(o_old, q_old, kv_old);
        std::ostringstream old_times;
        for (int i = 0; i < iters; i++) {
            time_utils::time_point start = time_utils::now();
            old_app(o_old, q_old, kv_old);
            time_utils::time_point stop = time_utils::now();
            old_times << " " << (int)time_utils::duration_us(start, stop).first;
        }
        o_old.sync_from_device();
        header_print("attn", "old kernel call times (us):" << old_times.str());
        compare_outputs("new vs old (all rows)", o.data(), o_old.data(), o.size());
        std::cout << "SUMMARY_OLD hash=" << std::hex << fnv1a(o_old.data(), o_old.size() * sizeof(bf16)) << std::dec << std::endl;
    }

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
    compare_outputs("new vs fp32 CPU (checked rows)", got.data(), want.data(), got.size());
    if (!old_xclbin.empty()) {
        buffer<bf16> got_old((size_t)R * DQ);
        for (int r = 0; r < R; r++) {
            memcpy(got_old.data() + (size_t)r * DQ, o_old.data() + (size_t)rows[r] * DQ, DQ * sizeof(bf16));
        }
        compare_outputs("old vs fp32 CPU (checked rows)", got_old.data(), want.data(), got_old.size());
    }

    const double causal_macs = 2.0 * Heads * DH * ((double)L * (L + 1) / 2.0);
    const double us = total_us / iters;
    std::cout << "SUMMARY L=" << L << " us/call=" << std::fixed << std::setprecision(0) << us
              << " useful TMAC/s=" << std::setprecision(2) << causal_macs / (us * 1e-6) / 1e12
              << " hash=" << std::hex << fnv1a(o.data(), o.size() * sizeof(bf16)) << std::dec << std::endl;
    return 0;
}
