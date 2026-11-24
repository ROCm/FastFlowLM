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
#include "tensor_utils/q4_npu_eXpress.hpp"
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
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("Random,r", po::value<bool>()->default_value(true), "Use random q/k/v + CPU reference instead of the safetensors dump");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(80), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1000), "Iteration");
    const char* env_ref = std::getenv("FLM_REFERENCE_DIR");
    desc.add_options()("reference", po::value<std::string>()->default_value(
        env_ref ? std::string(env_ref) + "/hy_mt2_1.8b_reference.safetensors" : ""),
        "With -r false: the safetensors dump to read q/k/v and the expected output from "
        "(default: $FLM_REFERENCE_DIR/hy_mt2_1.8b_reference.safetensors; not shipped)");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const bool RANDOM = vm["Random"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const std::string reference_path = vm["reference"].as<std::string>();
    if (!RANDOM && reference_path.empty()) {
        header_print("error", "No reference: -r false needs --reference or FLM_REFERENCE_DIR (references are not shipped with the dev kit)");
        return 1;
    }

    const int d = 128;
    const int HEADS = 16;
    const int KV_HEADS = 4;
    const int DQ = d * HEADS;
    const int DK = d * KV_HEADS;
    const int DV = DK;

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_wanted_chunked = ((L_wanted + 127) / 128) * 128;
    int L_end = L_begin + L_wanted_chunked;
    int L_end_chunked = ((L_end + 127) / 128) * 128;
    int L_padded = L_end_chunked;
    int MAX_L = L_padded;
    std::cout << "L_begin: " << L_begin << " L_wanted: " << L_wanted << " L_end: " << L_end << " L_wanted_chunked: " << L_wanted_chunked << " L_padded: " << L_padded << std::endl;

    // Must match what is handed to gen_seq_1_4 below.
    const int WINDOW_SIZE = 512;
    const bool IS_SLIDING = false;
    const int using_window_size = IS_SLIDING ? WINDOW_SIZE : MAX_L;

    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = L_padded * DK;
    const int V_VOLUME = L_padded * DV;
    const int O_VOLUME = L_padded * DQ;

    // Host side staging buffers, already in the layout the NPU expects:
    // q_host row r holds the global token (L_begin + r), k/v_host row r holds the global token r.
    buffer<bf16> q_host(Q_VOLUME);
    buffer<bf16> k_host(K_VOLUME);
    buffer<bf16> v_host(V_VOLUME);
    // Golden output, either the dumped one or the one computed on the CPU.
    buffer<bf16> aux_out(L_wanted * DQ);

    if (RANDOM) {
        header_print("info", "Generating random q/k/v, seed " << SEED);
        std::mt19937 rng(SEED);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < Q_VOLUME; i++) q_host[i] = (bf16)dist(rng);
        for (int i = 0; i < K_VOLUME; i++) k_host[i] = (bf16)dist(rng);
        for (int i = 0; i < V_VOLUME; i++) v_host[i] = (bf16)dist(rng);
    } else {
        buffer<bf16> q_cache;
        buffer<bf16> k_cache;
        buffer<bf16> v_cache;
        buffer<bf16> aux_out_whole;
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
        references->load_weights(q_cache, "layer_0.q_norm");
        references->load_weights(k_cache, "layer_0.k_norm");
        references->load_weights(v_cache, "layer_0.v_proj");
        references->load_weights(aux_out_whole, "layer_0.attention_out");
        header_print("info", "Finished loading data");
        header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
        header_print("info", "K shape: " << k_cache.size() / DK << "x" << DK);
        header_print("info", "V shape: " << v_cache.size() / DV << "x" << DV);
        header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);

        q_host.memset((bf16)0.0f);
        k_host.memset((bf16)0.0f);
        v_host.memset((bf16)0.0f);
        q_host.copy_from(q_cache.data() + (size_t)L_begin * DQ, std::min<size_t>(Q_VOLUME, q_cache.size() - (size_t)L_begin * DQ));
        k_host.copy_from(k_cache.data(), std::min<size_t>(K_VOLUME, k_cache.size()));
        v_host.copy_from(v_cache.data(), std::min<size_t>(V_VOLUME, v_cache.size()));
        aux_out.copy_from(aux_out_whole.data() + (size_t)L_begin * DQ, L_wanted * DQ);
    }

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME);

    // move q_cache
    q_0.copy_from(q_host.data(), Q_VOLUME);
    q_0.sync_to_device();

    // move kv_cache, k first then v, each L_padded rows of DK columns
    kv_0.copy_from(k_host.data(), K_VOLUME);
    memcpy(kv_0.data() + K_VOLUME, v_host.data(), (size_t)V_VOLUME * sizeof(bf16));
    kv_0.sync_to_device();

    const int KV_CACHE_SIZE = L_padded * (DK + DV);
    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_seq_1_4(app_0.seq(), L_begin, L_wanted_chunked, L_padded, WINDOW_SIZE, 16, d, HEADS, KV_HEADS, KV_CACHE_SIZE, IS_SLIDING);

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
    buffer<bf16> o_wanted(L_wanted * DQ);
    o_wanted.copy_from(o_0.data(), L_wanted * DQ);

    if (VERIFY) {
        if (RANDOM) {
            header_print("info", "Computing the CPU reference");
            attn_ref_config cfg;
            cfg.dh = d;
            cfg.n_q_heads = HEADS;
            cfg.n_kv_heads = KV_HEADS;
            cfg.window_size = using_window_size;
            cfg.scale = 1.0f / std::sqrt((float)d);
            time_utils::time_point cpu_start = time_utils::now();
            attn_cpu_ref(aux_out, q_host, k_host, v_host, L_wanted, L_begin, cfg);
            time_utils::time_with_unit cpu_time = time_utils::duration_ms(cpu_start, time_utils::now());
            header_print("info", "CPU reference time: " << cpu_time.first << " ms");
        }

        header_print("info", "Reference out");
        utils::print_matrix(aux_out, DQ);
        header_print("info", "NPU out");
        utils::print_matrix(o_wanted, DQ);

        error_metrics metrics = get_error_metrics(o_wanted, aux_out);
        print_error_metrics(metrics);
        pass = (metrics.CosineSimilarity > 0.97f) && (metrics.NormalizedMSE < 6e-2f);
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
