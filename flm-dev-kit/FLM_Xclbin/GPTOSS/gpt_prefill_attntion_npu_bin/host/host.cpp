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

#include "tensor_utils/q4_npu_eXpress.hpp"
#include "sequence_gen.hpp"

namespace po = boost::program_options;

float get_normalized_l2_error(buffer<bf16>& y, buffer<bf16>& y_ref){
    float sum_error = 0;
    float sum_ref = 0;
    for (int i = 0; i < y.size(); i++){
        sum_error += (y[i] - y_ref[i]) * (y[i] - y_ref[i]);
        sum_ref += y_ref[i] * y_ref[i];
    }
    return sqrt(sum_error / sum_ref);
}

void load_data(buffer<bf16>& y, std::string filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()){
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }
    // get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    y.resize(file_size / sizeof(float));
    buffer<float> y_float(y.size());
    file.read(reinterpret_cast<char*>(y_float.data()), file_size);
    file.close();
    y.resize(y_float.size());
    for (int i = 0; i < y.size(); i++){
        y[i] = (bf16)y_float[i];
    }
}

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
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("L_end,e", po::value<int>()->default_value(1024), "L_end");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(8), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();
    const int L_end = vm["L_end"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_begin_chunked = (L_begin / 32) * 32;
    int L_end_chunked = ((L_end + 31) / 32) * 32;
    int L_padded = L_end_chunked - L_begin_chunked;
    int L_offset = L_begin - L_begin_chunked;

    header_print("info", "Starting calculating the attention from " << L_begin << " (" << L_begin_chunked << ")" << " to " << L_end << " (" << L_end_chunked << ")");
    int d = 64;
    const int DQ = 4096;
    const int DK = 512;
    const int DV = 512;
    const int attn_heads = 64;
    const int kv_heads = 8;

    buffer<bf16> q_cache_ref;
    buffer<bf16> k_cache_ref;
    buffer<bf16> v_cache_ref;
    buffer<bf16> sinks_cache_ref;
    buffer<bf16> aux_out_whole_ref;
    load_data(q_cache_ref, "q_cache_causal.bin");
    load_data(k_cache_ref, "k_cache_causal.bin");
    load_data(v_cache_ref, "v_cache_causal.bin");
    load_data(sinks_cache_ref, "sinks_causal.bin");
    load_data(aux_out_whole_ref, "aux_out_causal.bin");
    std::cout << "Q:" << "(" << q_cache_ref.size() / DQ << "x" << DQ << ")" << std::endl;
    utils::print_matrix(q_cache_ref, DQ, 4);
    std::cout << "K:" << "(" << k_cache_ref.size() / DK << "x" << DK << ")" << std::endl;
    utils::print_matrix(k_cache_ref, DK, 4);
    std::cout << "V:" << "(" << v_cache_ref.size() / DV << "x" << DV << ")" << std::endl;
    utils::print_matrix(v_cache_ref, DV, 4);
    std::cout << "Sinks:" << sinks_cache_ref.size() << std::endl;
    utils::print_matrix(sinks_cache_ref, attn_heads);
    std::cout << "O:" << "(" << aux_out_whole_ref.size() / DQ << "x" << DQ << ")" << std::endl;
    utils::print_matrix(aux_out_whole_ref, DQ, 4);

    int AVAILABLE_L = q_cache_ref.size() / DQ;
    if (L_end > AVAILABLE_L){
        header_print("info", "L_end is greater than AVAILABLE_L, set L_end to AVAILABLE_L");
        return 1;
    }
    int MAX_L;
    float log2_AVAILABLE_L = std::log2(AVAILABLE_L);
    if (log2_AVAILABLE_L != std::floor(log2_AVAILABLE_L)){
        MAX_L = 1 << ((int)std::floor(log2_AVAILABLE_L) + 1);
    } else {
        MAX_L = AVAILABLE_L;
    }
    header_print("info", "AVAILABLE_L/MAX_L: " << AVAILABLE_L << "/" << MAX_L);

    // initialize npu

    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = attn_app_mgr->create_app();

    // define KV cache buffers
    // orderred in Lx(DK / 2)
    // K03, K47, V03, V47
    const int KV_CACHE_SIZE = MAX_L * (DK + DV);
    const int PARTIAL_SIZE = KV_CACHE_SIZE / 4;
    const int K03_OFFSET = 0;
    const int K47_OFFSET = PARTIAL_SIZE;
    const int V03_OFFSET = PARTIAL_SIZE * 2;
    const int V47_OFFSET = PARTIAL_SIZE * 3;
    buffer<bf16> kv_cache = app_0.create_bo_buffer<bf16>(KV_CACHE_SIZE, 5);
    buffer<bf16> k03_cache = buffer<bf16>(kv_cache.data(), PARTIAL_SIZE);
    buffer<bf16> k47_cache = buffer<bf16>(kv_cache.data() + K47_OFFSET, PARTIAL_SIZE);
    buffer<bf16> v03_cache = buffer<bf16>(kv_cache.data() + V03_OFFSET, PARTIAL_SIZE);
    buffer<bf16> v47_cache = buffer<bf16>(kv_cache.data() + V47_OFFSET, PARTIAL_SIZE);

    header_print("info", "Loading data to KV cache");
    
    for (int l = 0; l < L_end; l++){
        memcpy(k03_cache.data() + l * DK / 2, k_cache_ref.data() + l * DK         , DK * sizeof(bf16) / 2);
        memcpy(k47_cache.data() + l * DK / 2, k_cache_ref.data() + l * DK + DK / 2, DK * sizeof(bf16) / 2);
        memcpy(v03_cache.data() + l * DV / 2, v_cache_ref.data() + l * DV         , DV * sizeof(bf16) / 2);
        memcpy(v47_cache.data() + l * DV / 2, v_cache_ref.data() + l * DV + DV / 2, DV * sizeof(bf16) / 2);
    }

    kv_cache.sync_to_device();
    
    buffer<bf16> q_cache = app_0.create_bo_buffer<bf16>(L_padded * DQ, 4);
    memcpy(q_cache.data(), q_cache_ref.data() + L_begin_chunked * DQ, L_padded * DQ * sizeof(bf16));
    q_cache.sync_to_device();

    buffer<bf16> aux_out = app_0.create_bo_buffer<bf16>(L_padded * DQ, 3);
    aux_out.memset((bf16)0.0f);
    aux_out.sync_to_device();
    

    time_utils::time_with_unit npu_time = {0.0, "us"};
    size_t data_size = 0;

    gen_mha_engine_seq(app_0.seq(), L_begin_chunked, L_end_chunked, MAX_L, sinks_cache_ref.data(), false);


    // for (int i = L_begin_chunked / 16; i < L_end_chunked / 16; i++){
    //     data_size += 2 * 16 * (i + 1) * d * 8 * 2;
    // }
    // size_t data_size = 2 * 2 * 16 * d * 8 * (1 + L / 16) * L / 16 / 2;

    header_print("info", "Warm up.");
    app_0(aux_out, q_cache, kv_cache);
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(aux_out, q_cache, kv_cache);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    aux_out.sync_from_device();
    int L_valid = L_end - L_begin;
    buffer<bf16> aux_out_wanted(aux_out.data() + L_offset * DQ, L_valid * DQ);
    header_print("info", "Aux out");
    utils::print_matrix(aux_out_wanted, DQ, 16);

    buffer<bf16> aux_out_wanted_ref(aux_out_whole_ref.data() + L_begin * DQ, L_valid * DQ);
    header_print("info", "Aux out ref");
    utils::print_matrix(aux_out_wanted_ref, DQ, 16);

    buffer<float> cosine_similarity_table = buffer<float>(L_valid * DQ / d);
    for (int l = 0; l < L_valid; l++){
        for (int i = 0; i < DQ / d; i++){
            buffer<bf16> aux_out_wanted_block(aux_out_wanted.data() + l * DQ + i * d, d);
            buffer<bf16> aux_out_wanted_ref_block(aux_out_wanted_ref.data() + l * DQ + i * d, d);
            auto error_metrics = get_error_metrics(aux_out_wanted_block, aux_out_wanted_ref_block);
            cosine_similarity_table[l * DQ / d + i] = error_metrics.CosineSimilarity;
            // cosine_similarity_table[l * DQ / d + i] = error_metrics.RelativeL2;
            // cosine_similarity_table[l * DQ / d + i] = float(aux_out_wanted_block[0]);
        }
    }
    header_print("info", "Cosine similarity table, L x (DQ / DH)");
    utils::print_matrix(cosine_similarity_table, DQ / d, 16);

    error_metrics metrics = get_error_metrics(aux_out_wanted, aux_out_wanted_ref);
    print_error_metrics(metrics);

    // header_print("info", "Finished running kernel");
    // header_print("info", "Running kernel with bare call (run based).");
    // header_print("info", "Bandwidth: " << data_size / npu_time.first * 1e6 / 1e9 << " GB/s");

    // if (pass){
    //     header_print("info", "PASSED ");
    // } else {
    //     header_print("info", "FAILED!");
    // }
    return 0;
}
