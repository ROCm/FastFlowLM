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
#include "gemma_text/gemma_text_npu_sequence.hpp"
#include "lm_config.hpp"
#include <math.h>
#include "metrices.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

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
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(2048), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    std::string model_path = vm["model"].as<std::string>();
    const std::string model_type = vm["Model_type"].as<std::string>();
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int d = 0;
    if (model_type == "GEMMA_1B"){
        d = 256;
        model_path = model_path + "/gemma-3-1b-it";
    } else if (model_type == "GEMMA_4B"){
        d = 256;
        model_path = model_path + "/gemma-3-4b-it";
    } 
    else{
        std::cout << "Invalid model type: " << model_type << std::endl;
        return 1;
    }

    LM_Config config;
    config.from_pretrained(model_path);
    const int DQ = d * config.num_attention_heads;
    const int DK = d * config.num_key_value_heads;
    std::string window_type = "global";

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    load_data(q_cache, model_type + "/q_cache_" + window_type + ".bin");
    load_data(k_cache, model_type + "/k_cache_" + window_type + ".bin");
    load_data(v_cache, model_type + "/v_cache_" + window_type + ".bin");
    load_data(aux_out_whole, model_type + "/attention_output_" + window_type + ".bin");

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

    int MAX_L = q_cache.size() / DQ; 
    gemma_text_npu_sequence sequence(config, MAX_L);
    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_end = L_begin + L_wanted;

    int L_begin_chunked = (L_begin / 16) * 16;
    int L_end_chunked = ((L_end + 15) / 16) * 16;
    int L_padded = L_end_chunked - L_begin_chunked;

    buffer<bf16> aux_out((size_t)L_wanted * DQ);
    aux_out.copy_from(aux_out_whole.data() + L_begin * DQ, (size_t)L_wanted * DQ);

    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = MAX_L * DK;
    const int V_VOLUME = MAX_L * DK;
    const int O_VOLUME = L_padded * DQ;

    bool pass = true;
    // NPU instance
    npu_xclbin_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/" + model_type + "/xclbins/attn.xclbin";
    npu_app_desc accel_desc_0;
    accel_desc_0.xclbin_name = xclbin_name;
    accel_desc_0.app_name = "attn_compiled";


    header_print("info", "Matrix size " << L_padded << "x" << DQ << "x" << L_padded);

    npu_app app_0 = npu_instance.create_app(accel_desc_0);

    buffer<bf16> o_0 = app_0.create_bo_buffer<bf16>(O_VOLUME, 3);
    buffer<bf16> q_0 = app_0.create_bo_buffer<bf16>(Q_VOLUME, 4);
    buffer<bf16> kv_0 = app_0.create_bo_buffer<bf16>(K_VOLUME + V_VOLUME, 5);
    // move q_cache

    // move kv_cache
    size_t offset = (size_t)MAX_L * DK;

    void * k_src_ptr = static_cast<void*>(k_cache.data());
    void * v_src_ptr = static_cast<void*>(v_cache.data());
    void * k_ptr = static_cast<void*>(kv_0.data());
    void * v_ptr = static_cast<void*>(kv_0.data() + offset);

    size_t data_block_size_in_bytes = DK * sizeof(bf16);
    for (int l = 0; l < MAX_L; l++){
        // k-0-3
        memcpy(k_ptr, k_src_ptr, data_block_size_in_bytes);
        k_src_ptr += data_block_size_in_bytes;
        k_ptr += data_block_size_in_bytes;
        // v-0-3
        memcpy(v_ptr, v_src_ptr, data_block_size_in_bytes);
        v_src_ptr += data_block_size_in_bytes;
        v_ptr += data_block_size_in_bytes;
    }
   
    kv_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    size_t data_size = 0;

    for (int i = L_begin_chunked / 16 / 8; i < L_end_chunked / 16 / 8; i++){
        data_size += (size_t)2 * 16 * 8 * (i + 1) * d * 2;
    }
    std::cout << "L_begin_chunked " << L_begin_chunked << " L_end_chunked " << L_end_chunked << std::endl;
    std::cout << "L_end_chunked " << L_end_chunked << std::endl;
    std::cout << "data_size " << data_size << std::endl;
    // size_t data_size = 2 * 2 * 16 * d * 8 * (1 + (L_end_chunked - L_begin_chunked) / 16 / 8) * (L_end_chunked - L_begin_chunked) / 16 / 8 / 2;

    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    for (int i = 0; i < NT; i++) {
        q_0.copy_from(q_cache.data() + L_begin_chunked * DQ, (size_t)L_padded * DQ);
        q_0.sync_to_device();
        sequence.gen_mha_engine_seq(app_0.instr_seq, L_begin_chunked, L_end_chunked, false, 16);
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

    o_wanted.copy_from(o_0.data() + (L_begin - L_begin_chunked) * DQ, (size_t)L_wanted * DQ);

    header_print("info", "Aux out");
    utils::print_matrix(aux_out, DQ * 64, 16);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, DQ * 64, 16);

    error_metrics metrics = get_error_metrics(o_wanted, aux_out);
    print_error_metrics(metrics);
    // for (int i = 0; i < L_wanted; i += 128){
    //     buffer<bf16> o_wanted_block(o_wanted.data() + i * DQ, 128 * DQ);
    //     buffer<bf16> aux_out_block(aux_out.data() + i * DQ, 128 * DQ);
    //     error_metrics metrics = get_error_metrics(o_wanted_block, aux_out_block);
    //     std::cout << "Error till " << i + 128 << ": " << metrics.RelativeL2 << std::endl;
    // }
    int l_location = 2048 - 128;
    for (int chunk_idx = 0; chunk_idx < 8; chunk_idx++){
        buffer<bf16> o_wanted_block(o_wanted.data() + l_location * DQ + chunk_idx * 1024 * 16, 1024 * 16);
        buffer<bf16> aux_out_block(aux_out.data() + l_location * DQ + chunk_idx * 1024 * 16, 1024 * 16);
        error_metrics metrics = get_error_metrics(o_wanted_block, aux_out_block);
        std::cout << "Error on chunk " << chunk_idx + 1 << ": " << metrics.RelativeL2 << std::endl;
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
