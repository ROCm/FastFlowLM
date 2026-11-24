// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

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
#include "attn_ref.hpp"
#include "tensor_2d.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify against the CPU reference");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(256), "Number of query tokens");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "Global position of the first query token, multiple of 128");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(100), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    // Must match `lq` in kernel/common.h and `lq` in iron/attn_aie.py.
    const int KERNEL_LQ = 16;

    const int d = 128;
    const int HEADS = 32;
    const int KV_HEADS = 8;
    const int DQ = d * HEADS;
    const int DK = d * KV_HEADS;
    const int DV = DK;
    // The KV cache is stored as 2 chunks of 4 kv heads each; seq_gen indexes it
    // with `Num_of_d_in_KV_cache` = KV_HEADS_PER_CHUNK.
    const int KV_HEADS_PER_CHUNK = 4;
    const int KV_CHUNKS = KV_HEADS / KV_HEADS_PER_CHUNK;

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();
    // 8 cores per CU x lq query rows each is the round granularity.
    const int TOKENS_PER_ROUND = 8 * KERNEL_LQ;
    if (L_begin % TOKENS_PER_ROUND != 0) {
        header_print("error", "L_begin must be a multiple of " << TOKENS_PER_ROUND << ", got " << L_begin);
        return 1;
    }
    const int L_q_rows = ((L_wanted + TOKENS_PER_ROUND - 1) / TOKENS_PER_ROUND) * TOKENS_PER_ROUND;
    const int L_end = L_begin + L_q_rows;
    // K/V are addressed from global token 0, so the cache spans [0, L_end).
    const int MAX_L = L_end;
    header_print("info", "L_begin: " << L_begin << " L_wanted: " << L_wanted
                        << " L_q_rows: " << L_q_rows << " L_end: " << L_end);

    // Must match what is handed to gen_seq_1_4 below.
    const int WINDOW_SIZE = 1024;
    const bool IS_SLIDING = false;
    const int using_window_size = IS_SLIDING ? WINDOW_SIZE : L_end;

    const int Q_VOLUME = L_q_rows * DQ;
    const int K_VOLUME = MAX_L * DK;
    const int V_VOLUME = MAX_L * DV;
    const int O_VOLUME = L_q_rows * DQ;
    const int KV_CACHE_SIZE = MAX_L * (DK + DV);

    // Host side staging buffers in the plain [rows][heads * dh] layout the CPU
    // reference expects: q_host row r is global token (L_begin + r), k/v_host
    // row r is global token r.
    buffer<bf16> q_host(Q_VOLUME);
    buffer<bf16> k_host(K_VOLUME);
    buffer<bf16> v_host(V_VOLUME);

    header_print("info", "Generating random q/k/v, seed " << SEED);
    std::mt19937 rng(SEED);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < Q_VOLUME; i++) q_host[i] = (bf16)dist(rng);
    for (int i = 0; i < K_VOLUME; i++) k_host[i] = (bf16)dist(rng);
    for (int i = 0; i < V_VOLUME; i++) v_host[i] = (bf16)dist(rng);

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = attn_app_mgr->create_app();

    header_print("info", "Matrix size " << L_q_rows << "x" << DQ << "x" << MAX_L);

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME);

    // q keeps the host layout as is.
    q_0.copy_from(q_host.data(), Q_VOLUME);
    q_0.sync_to_device();

    // kv is chunked: K for kv heads 0-3, K for kv heads 4-7, then the same two
    // chunks for V. Each chunk is [MAX_L][KV_HEADS_PER_CHUNK * d].
    const size_t chunk_stride = (size_t)MAX_L * KV_HEADS_PER_CHUNK * d;
    const size_t v_base = (size_t)KV_CACHE_SIZE / 2;
    for (int c = 0; c < KV_CHUNKS; c++) {
        for (int l = 0; l < MAX_L; l++) {
            const size_t src = (size_t)l * DK + (size_t)c * KV_HEADS_PER_CHUNK * d;
            const size_t dst = (size_t)c * chunk_stride + (size_t)l * KV_HEADS_PER_CHUNK * d;
            memcpy(kv_0.data() + dst, k_host.data() + src, KV_HEADS_PER_CHUNK * d * sizeof(bf16));
            memcpy(kv_0.data() + v_base + dst, v_host.data() + src, KV_HEADS_PER_CHUNK * d * sizeof(bf16));
        }
    }
    kv_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_seq_1_4(app_0.seq(), L_begin, L_end, MAX_L, WINDOW_SIZE, KERNEL_LQ, d, HEADS,
                KV_HEADS_PER_CHUNK, KV_CACHE_SIZE, IS_SLIDING);

    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, kv_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();

    if (VERIFY) {
        buffer<bf16> o_npu(L_wanted * DQ);
        memcpy(o_npu.data(), o_0.data(), (size_t)L_wanted * DQ * sizeof(bf16));

        header_print("info", "Computing the CPU reference");
        attn_ref_config cfg;
        cfg.dh = d;
        cfg.n_q_heads = HEADS;
        cfg.n_kv_heads = KV_HEADS;
        cfg.window_size = using_window_size;
        cfg.scale = 1.0f / std::sqrt((float)d);
        buffer<bf16> o_ref(L_q_rows * DQ);
        time_utils::time_point cpu_start = time_utils::now();
        attn_cpu_ref(o_ref, q_host, k_host, v_host, L_q_rows, L_begin, cfg);
        time_utils::time_with_unit cpu_time = time_utils::duration_ms(cpu_start, time_utils::now());
        header_print("info", "CPU reference time: " << cpu_time.first << " ms");

        buffer<bf16> o_ref_wanted(L_wanted * DQ);
        memcpy(o_ref_wanted.data(), o_ref.data(), (size_t)L_wanted * DQ * sizeof(bf16));

        error_metrics metrics = get_error_metrics(o_npu, o_ref_wanted);
        print_error_metrics(metrics);
        pass = (metrics.CosineSimilarity > 0.99f) && (metrics.NormalizedMSE < 2e-2f);
    }

    header_print("info", "Finished running kernel");
    header_print("info", "NPU time: " << npu_time.first << " us");

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return pass ? 0 : 1;
}
