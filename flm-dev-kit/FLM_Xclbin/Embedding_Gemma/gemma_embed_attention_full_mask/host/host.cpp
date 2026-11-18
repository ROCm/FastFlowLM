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
    desc.add_options()("Verify,v", po::value<bool>()->default_value(true), "Verify the result");
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(2048), "L_wanted");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(100), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int d = 256;

    const int DQ = d * 3;
    const int DK = d;
    std::string window_type = "global";

    buffer<bf16> q_cache;
    buffer<bf16> k_cache;
    buffer<bf16> v_cache;
    buffer<bf16> aux_out_whole;

    load_data(q_cache, "../data/q_cache_" + window_type + ".bin");
    load_data(k_cache, "../data/k_cache_" + window_type + ".bin");
    load_data(v_cache, "../data/v_cache_" + window_type + ".bin");
    load_data(aux_out_whole, "../data/attention_output_" + window_type + ".bin");

    header_print("info", "Finished loading data");
    header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
    header_print("info", "K shape: " << k_cache.size() / (DK) << "x" << DK);
    header_print("info", "V shape: " << v_cache.size() / (DK) << "x" << DK);
    header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);

    header_print("info", "q_cache.size()" << q_cache.size());
    // utils::print_matrix(q_cache, DQ);
    header_print("info", "k_cache.size()" << k_cache.size());
    // utils::print_matrix(k_cache, DK);
    header_print("info", "v_cache.size()" << v_cache.size());
    // utils::print_matrix(v_cache, DK);
    header_print("info", "aux_out_whole.size()" << aux_out_whole.size());
    // utils::print_matrix(aux_out_whole, DQ);

    int MAX_L = 2048; 
    const int L_wanted = vm["L_wanted"].as<int>();

    int L_end = L_wanted;

    int LK_end_chunked = ((L_end + 15) / 16) * 16;
    int LQ_end_chunked = ((L_end + 127) / 128) * 128;
    int L_padded = LQ_end_chunked;

    buffer<bf16> aux_out(44 * DQ);
    aux_out.copy_from(aux_out_whole.data(), 44 * DQ);

    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = MAX_L * DK;
    const int V_VOLUME = MAX_L * DK;
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
    q_0.copy_from(q_cache.data(), 44 * DQ);
    utils::print_matrix(q_0, DQ);
    q_0.sync_to_device();

    // move kv_cache
    size_t offset = MAX_L * DK;
    buffer<bf16> ones(MAX_L * DK);
    for (int i = 0; i < MAX_L * DK; i++){
        ones[i] = 1.0f;
    }

    void * k_src_ptr = static_cast<void*>(k_cache.data());
    void * v_src_ptr = static_cast<void*>(v_cache.data());
    void * k_ptr = static_cast<void*>(kv_0.data());
    void * v_ptr = static_cast<void*>(kv_0.data() + offset);

    size_t data_block_size_in_bytes = DK * sizeof(bf16);
    for (int l = 0; l < 44; l++){
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
    o_0.memset((bf16)(-1.0f));
    o_0.sync_to_device();

    time_utils::time_with_unit npu_time = {0.0, "us"};
    size_t data_size = 0;

    // for (int i = 0; i < LQ_end_chunked / 16 / 8; i++){
    //     data_size += 2 * 16 * 8 * (i + 1) * d * 2;
    // }
    std::cout << "LQ_end_chunked " << LQ_end_chunked << std::endl;
    // std::cout << "data_size " << data_size << std::endl;
    data_size = (L_wanted * DK * 2 * L_wanted / 128 + L_wanted * DQ) * 2;

    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    for (int i = 0; i < NT; i++) {
        gen_mha_engine_seq(app_0.seq(), LK_end_chunked, LQ_end_chunked, L_wanted);
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
    buffer<bf16> o_wanted(44 * DQ);

    o_wanted.copy_from(o_0.data(), 44 * DQ);

    header_print("info", "Aux out");
    utils::print_matrix(aux_out, DQ);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, DQ);

    error_metrics metrics = get_error_metrics(o_wanted, aux_out);
    print_error_metrics(metrics);
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
