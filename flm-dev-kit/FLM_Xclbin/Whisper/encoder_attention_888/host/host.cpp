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
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->default_value("../../models"), "Model file");
    desc.add_options()("Seed,s", po::value<unsigned int>()->default_value(32), "Random seed");
    desc.add_options()("Model_type,t", po::value<std::string>()->default_value("GEMMA_1B"), "Model type");
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1536), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(300), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    std::string model_path = vm["model"].as<std::string>();
    const std::string model_type = vm["Model_type"].as<std::string>();
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int DH = 64;
    const int num_of_heads = 20;
    const int DQ = DH * num_of_heads;
    const int DK = DH * num_of_heads;
    const int DV = DH * num_of_heads;

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    load_data(q_cache, "q_proj.bin");
    load_data(k_cache, "k_proj.bin");
    load_data(v_cache, "v_proj.bin");
    load_data(aux_out_whole, "aux_out.bin");

    header_print("info", "Finished loading data");
    header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
    header_print("info", "K shape: " << k_cache.size() / (DK) << "x" << DK);
    header_print("info", "V shape: " << v_cache.size() / (DK) << "x" << DK);
    header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);

    int MAX_L = q_cache.size() / DQ; 
    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_end = L_begin + L_wanted;

    int L_begin_chunked = (L_begin / 512) * 512;
    int L_end_chunked = ((L_end + 511) / 512) * 512;
    int L_padded = L_end_chunked - L_begin_chunked;


    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = MAX_L * DK;
    const int V_VOLUME = MAX_L * DV;
    const int O_VOLUME = L_padded * DQ;

    bool pass = true;
    // NPU instance
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0

    std::string xclbin_name = "build/xclbins/attn.xclbin";
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* attn_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);


    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

    npu_app app_0 = attn_app_mgr->create_app();

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME, 3);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME, 4);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME, 5);

    // move q_cache
    q_0.copy_from(q_cache.data(), (size_t)L_wanted * DQ);
    q_0.sync_to_device();
    
    // move k_cache and v_cache
    size_t offset = (size_t)MAX_L * DK;
    buffer<bf16> ones((size_t)MAX_L * DK);
    ones.memset((bf16)1.00f);
    void * k_src_ptr = static_cast<void*>(k_cache.data());
    void * v_src_ptr = static_cast<void*>(v_cache.data());
    size_t kv_chunk_size = (size_t)MAX_L * DK;
    void * k_ptr = static_cast<void*>(kv_0.data());
    void * v_ptr = static_cast<void*>(kv_0.data() + kv_chunk_size);

    size_t data_block_size_in_bytes = DK * sizeof(bf16);
    for (int l = 0; l < MAX_L; l++){
        // k-0-1
        memcpy(k_ptr, k_src_ptr, data_block_size_in_bytes);
        k_src_ptr += data_block_size_in_bytes;
        k_ptr += data_block_size_in_bytes;

        // v-0-1
        memcpy(v_ptr, v_src_ptr, data_block_size_in_bytes);
        v_src_ptr += data_block_size_in_bytes;
        v_ptr += data_block_size_in_bytes;
    }
   
    kv_0.sync_to_device();

    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};

    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    gen_whisper_seq(app_0.seq(), L_padded, MAX_L, 64, DH, num_of_heads);
    
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
    buffer<bf16> aux_out((size_t)L_padded * DQ);
    aux_out.copy_from(aux_out_whole.data(), (size_t)L_wanted * DQ);

    header_print("info", "Aux out");
    utils::print_matrix(aux_out, DQ);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, DQ);

    tensor_2d<bf16> o_wanted_tensor(o_wanted, DQ, 0);
    tensor_2d<bf16> aux_out_tensor(aux_out, DQ, 0);
    buffer<bf16> c_sin(L_wanted);
    for (int i = 0; i < L_wanted; i++){
        error_metrics metrics = get_error_metrics(o_wanted_tensor[i], aux_out_tensor[i]);
        c_sin[i] = metrics.CosineSimilarity;
    }
    header_print("info", "Cosine similarity");
    utils::print_matrix(c_sin, 512, 3);

    error_metrics metrics = get_error_metrics(o_wanted, aux_out);
    print_error_metrics(metrics);
    
    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    header_print("info", "NPU time: " << npu_time.first << " us");

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }
    return 0;
}