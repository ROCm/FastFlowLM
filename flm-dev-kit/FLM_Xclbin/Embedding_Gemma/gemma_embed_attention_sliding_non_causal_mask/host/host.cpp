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
#include <math.h>
#include "metrices.hpp"
#include "seq_gen.hpp"
#include "tensor_2d.hpp"

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
    buffer<float> y_float(file_size / sizeof(float));
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
    desc.add_options()("L,l", po::value<int>()->default_value(1024), "L");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1), "Iteration");
    desc.add_options()("Window_type,t", po::value<std::string>()->default_value("local"), "Window type");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const std::string window_type = vm["Window_type"].as<std::string>();
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const int L = vm["L"].as<int>();
    const int SEED = vm["Seed"].as<unsigned int>();

    const int d = 256;
    const int DQ = d * 3;
    const int DK = d;
    const int DV = d;
    const int MAX_L = 2048;
    const int W = 257;
    const int KV_CACHE_SIZE = MAX_L * (DK + DV);
    const uint32_t K_OFFSET = 0;
    const uint32_t V_OFFSET = KV_CACHE_SIZE / 2;
    const int CHUNK_SIZE = 128;

    if (L > MAX_L){
        std::cout << "L is greater than MAX_L" << std::endl;
        return 1;
    }
    
    int L_padded = (L + CHUNK_SIZE - 1) / CHUNK_SIZE * CHUNK_SIZE;
    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);
    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = MAX_L * DK;
    const int V_VOLUME = MAX_L * DK;
    const int O_VOLUME = L_padded * DQ;

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    load_data(q_cache, "data/q_cache_" + window_type + ".bin");
    load_data(k_cache, "data/k_cache_" + window_type + ".bin");
    load_data(v_cache, "data/v_cache_" + window_type + ".bin");
    load_data(aux_out_whole, "data/attention_output_" + window_type + ".bin");

    header_print("info", "Finished loading data");
    header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
    header_print("info", "K shape: " << k_cache.size() / (DK) << "x" << DK);
    header_print("info", "V shape: " << v_cache.size() / (DK) << "x" << DK);
    header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);

    header_print("info", "q_cache.size()" << q_cache.size());
    utils::print_matrix(q_cache, DQ);
    header_print("info", "k_cache.size()" << k_cache.size());
    utils::print_matrix(k_cache, DK);
    header_print("info", "v_cache.size()" << v_cache.size());
    utils::print_matrix(v_cache, DK);
    header_print("info", "aux_out_whole.size()" << aux_out_whole.size());
    utils::print_matrix(aux_out_whole, DQ);
  
   

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/GEMMA_1B/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);


    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME, 3);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME, 4);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(KV_CACHE_SIZE, 5);
    // move q_cache
    memcpy(q_0.data(), q_cache.data(), L * DQ * sizeof(bf16));

    // move kv_cache
    size_t offset = MAX_L * DK;

    void * k_src_ptr = static_cast<void*>(k_cache.data());
    void * v_src_ptr = static_cast<void*>(v_cache.data());
    void * k_ptr = static_cast<void*>(kv_0.data());
    void * v_ptr = static_cast<void*>(kv_0.data() + offset);

    size_t data_block_size_in_bytes = DK * sizeof(bf16);
    memcpy(k_ptr, k_src_ptr, L * data_block_size_in_bytes);
    memcpy(v_ptr, v_src_ptr, L * data_block_size_in_bytes);
   
    kv_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    size_t data_size = 0;

    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_sliding_mha_engine_seq(app_0.seq(), L);
    for (int i = 0; i < NT; i++) {
        q_0.sync_to_device();
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, kv_0);
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);
    if (VERIFY){
        o_0.sync_from_device();
        buffer<bf16> o_wanted(L * DQ);
        o_wanted.copy_from(o_0.data(), L * DQ);

        buffer<bf16> o_ref(L * DQ);
        memcpy(o_ref.data(), aux_out_whole.data(), L * DQ * sizeof(bf16));

        header_print("info", "Aux out");
        utils::print_matrix(o_ref, DQ, 16);
        header_print("info", "O out");
        utils::print_matrix(o_wanted, DQ, 16);
        
        tensor_2d<bf16> o_wanted_tensor(o_wanted, DQ, 0);
        tensor_2d<bf16> o_ref_tensor(o_ref, DQ, 0);
        for (int i = 0; i < L; i++){
            error_metrics metrics = get_error_metrics(o_wanted_tensor[i], o_ref_tensor[i]);
            float cosine_similarity = metrics.CosineSimilarity;
            std::cout << "Cosine similarity [" << i << "]: " << cosine_similarity << std::endl;
        }

        error_metrics metrics = get_error_metrics(o_wanted, o_ref);
        print_error_metrics(metrics);
    }
    
    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");
    header_print("info", "Bandwidth: " << data_size / npu_time.first * 1e6 / 1e9 << " GB/s");

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}
