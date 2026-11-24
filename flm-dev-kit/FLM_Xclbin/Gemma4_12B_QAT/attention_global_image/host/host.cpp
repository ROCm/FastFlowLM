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
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "tensor_2d.hpp"
#include "attn_ref.hpp"
#include <random>

namespace po = boost::program_options;

int main(int argc, char **argv) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(256), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(4738), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(100), "Iteration");
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Random,r", po::value<bool>()->default_value(true), "Use random q/k/v + CPU reference instead of the safetensors dump");
    const char* env_ref = std::getenv("FLM_REFERENCE_DIR");
    desc.add_options()("reference", po::value<std::string>()->default_value(
        env_ref ? std::string(env_ref) + "/gemma4_12b_qat_reference.safetensors" : ""),
        "With -r false: the safetensors dump to read q/k/v and the expected output from "
        "(default: $FLM_REFERENCE_DIR/gemma4_12b_qat_reference.safetensors; not shipped)");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();

    int d = 512;
    int DQ = d * 16;
    int DK = d * 1;
    int DV = DK;

    const bool RANDOM = vm["Random"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const std::string reference_path = vm["reference"].as<std::string>();
    if (!RANDOM && reference_path.empty()) {
        header_print("error", "No reference: -r false needs --reference or FLM_REFERENCE_DIR (references are not shipped with the dev kit)");
        return 1;
    }

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_wanted_chunked = ((L_wanted + 127) / 128) * 128;
    int L_end = L_begin + L_wanted_chunked;
    int L_end_chunked = ((L_end + 127) / 128) * 128;
    int L_padded = L_end_chunked;
    
    std::cout << "L_begin: " << L_begin << " L_wanted: " << L_wanted << " L_end: " << L_end
              << " L_wanted_chunked: " << L_wanted_chunked << " L_padded: " << L_padded << std::endl;

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;
    if (RANDOM) {
        header_print("info", "Generating random q/k/v, seed " << SEED);
        std::mt19937 rng(SEED);
        auto fill_random = [&rng](buffer<bf16> &b, size_t n, float stddev) {
            std::normal_distribution<float> dist(0.0f, stddev);
            b.resize(n);
            for (size_t i = 0; i < n; i++) b[i] = (bf16)dist(rng);
        };
        fill_random(q_cache, (size_t)(L_begin + L_wanted) * DQ, 0.3f);
        fill_random(k_cache, (size_t)L_end * DK, 0.3f);
        fill_random(v_cache, (size_t)L_end * DV, 1.0f);
    } else {
        std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
        references->load_weights(q_cache, "layer_5.q_embed");
        references->load_weights(k_cache, "layer_5.k_embed");
        references->load_weights(v_cache, "layer_5.v_norm");
        references->load_weights(aux_out_whole, "layer_5.attention_out");
        header_print("info", "Finished loading data");
        header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
        header_print("info", "K shape: " << k_cache.size() / DK << "x" << DK);
        header_print("info", "V shape: " << v_cache.size() / DV << "x" << DV);
        header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);
    }

    const int Q_VOLUME = L_wanted_chunked * DQ;
    const int K_VOLUME = L_padded * DK;
    const int V_VOLUME = L_padded * DV;
    const int O_VOLUME = L_wanted_chunked * DQ;

    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn_global_image.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME);
    kv_0.memset(bf16(0.0f));

    // move q_cache
    q_0.memset((bf16)(0.0f));
    q_0.copy_from(q_cache.data() + L_begin * DQ, (size_t)L_wanted * DQ);
    q_0.sync_to_device();

    const size_t kv_v_offset = (size_t)L_padded * DK;
    kv_0.memset((bf16)(0.0f));
    memcpy(kv_0.data(), k_cache.data(), (size_t)L_end * DK * sizeof(bf16));
    memcpy(kv_0.data() + kv_v_offset, v_cache.data(), (size_t)L_end * DV * sizeof(bf16));
    kv_0.sync_to_device();
    const int KV_CACHE_SIZE = L_padded * (DK + DV);
    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    int lc = 8;
    int Heads = 16;

    gen_seq_1_4(app_0.seq(), L_begin, L_wanted_chunked, L_padded, lc, d, Heads, KV_CACHE_SIZE);

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
    buffer<bf16> o_wanted((size_t)L_wanted * DQ);
    o_wanted.copy_from(o_0.data(), (size_t)L_wanted * DQ);
    buffer<bf16> aux_out((size_t)L_wanted * DQ);
    if (RANDOM) {
        header_print("info", "CPU reference over " << L_end << " keys");
        attn_ref_config cfg{Heads, 1, d};
        attn_cpu_ref(q_0.data(), L_wanted, kv_0.data(), kv_0.data() + kv_v_offset, L_end, cfg, aux_out.data());
    } else {
        aux_out.copy_from(aux_out_whole.data() + L_begin * DQ, (size_t)L_wanted * DQ);
    }

    header_print("info", "Aux out");
    utils::print_matrix(aux_out, DQ);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, DQ);

    // With -r false the verdict is the reference dump itself: layer_5.attention_out was
    // produced by transformers with the full (block) mask on, so a causal-only overlay
    // cannot reach it. The residual that remains is the bf16/bfp16 mmul floor of the array.
    error_metrics metrics = get_error_metrics(o_wanted, aux_out);
    print_error_metrics(metrics, RANDOM ? "NPU vs CPU reference" : "NPU vs reference dump");

    bool pass = RANDOM ? (metrics.CosineSimilarity > 0.97f && metrics.NormalizedMSE < 6e-2f)
                       : metrics.CosineSimilarity > 0.99f;

    header_print("info", "Finished running kernel");
    header_print("info", "NPU time: " << npu_time.first << " us");

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return pass ? 0 : 1;
}