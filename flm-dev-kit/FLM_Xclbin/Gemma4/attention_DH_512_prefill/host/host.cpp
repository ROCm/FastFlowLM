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
    desc.add_options()("L_wanted,l", po::value<int>()->default_value(1017), "L_wanted");
    desc.add_options()("L_begin,b", po::value<int>()->default_value(0), "L_begin");
    desc.add_options()("Iteration,i", po::value<int>()->default_value(1), "Iteration");
    arg_utils::parse_options(argc, argv, desc, vm);

    // User logic
    const int NT = vm["Iteration"].as<int>();
    const bool VERIFY = vm["Verify"].as<bool>();
    const unsigned int SEED = vm["Seed"].as<unsigned int>();

    int d = 512;
    int DQ = d * 8;
    int DK = d * 2;
    int DV = DK;

    // load reference data
    std::string reference_path = "../layer_5.safetensors";
    std::unique_ptr<SafeTensors> references = std::make_unique<SafeTensors>(reference_path);
    buffer<bf16> q_cache;
    references->load_weights(q_cache, "q_embed");
    buffer<bf16> k_cache;
    references->load_weights(k_cache, "k_embed");
    buffer<bf16> v_cache;
    references->load_weights(v_cache, "v_norm");
    buffer<bf16> aux_out_whole;
    references->load_weights(aux_out_whole, "attention_output");

    header_print("info", "Finished loading data");
    header_print("info", "Q shape: " << q_cache.size() / DQ << "x" << DQ);
    header_print("info", "K shape: " << k_cache.size() / DK << "x" << DK);
    header_print("info", "V shape: " << v_cache.size() / DV << "x" << DV);
    header_print("info", "O shape: " << aux_out_whole.size() / DQ << "x" << DQ);

    const int L_wanted = vm["L_wanted"].as<int>();
    const int L_begin = vm["L_begin"].as<int>();

    int L_end = L_begin + L_wanted;

    int L_begin_chunked = (L_begin / 128) * 128;
    int L_end_chunked = ((L_end + 127) / 128) * 128;
    int L_padded = L_end_chunked - L_begin_chunked;

    const int Q_VOLUME = L_padded * DQ;
    const int K_VOLUME = L_padded * DK;
    const int V_VOLUME = L_padded * DV;
    const int O_VOLUME = L_padded * DQ;

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
    kv_0.memset(bf16(0.0f));

    // move q_cache
    q_0.copy_from(q_cache.data() + L_begin_chunked * DQ, (size_t)L_padded * DQ);
    // for (int i = 0; i < L_padded * DQ; i++){
    //     q_0[i] = bf16(1.0f);
    // }
    q_0.sync_to_device();

    buffer<bf16> ones((size_t)L_padded * DK);
    buffer<bf16> k((size_t)L_padded * DK);
    for (int i = 0; i < L_padded * DK; i++){
        ones[i] = 1.0f;
        k[i] = bf16(i * 0.00012f);
    }

    size_t offset = L_padded * d * 2;
    void * k_src_ptr = static_cast<void*>(k_cache.data());
    void * v_src_ptr = static_cast<void*>(v_cache.data());
    void * k_01_ptr = static_cast<void*>(kv_0.data());
    void * v_01_ptr = static_cast<void*>(kv_0.data() + offset);

    size_t data_block_size_in_bytes = d * 2 * sizeof(bf16);
    // move kv_cache
    for (int l = 0; l < L_wanted; l++){
        memcpy(k_01_ptr, k_src_ptr, data_block_size_in_bytes);
        k_src_ptr += data_block_size_in_bytes;
        k_01_ptr += data_block_size_in_bytes;
        memcpy(v_01_ptr, v_src_ptr, data_block_size_in_bytes);
        v_src_ptr += data_block_size_in_bytes;
        v_01_ptr += data_block_size_in_bytes;
    }
    kv_0.sync_to_device();
    const int KV_CACHE_SIZE = L_padded * (DK + DV);
    time_utils::time_with_unit npu_time = {0.0, "us"};
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    int window_size = 512;
    int lc = 8;
    int Heads = 8;
    int GQA = 4;
    int Num_of_d_in_KV_cache = 2;
    gen_seq_1_4(app_0.seq(), L_begin_chunked, L_end_chunked, L_wanted, window_size, lc, d, Heads, GQA, Num_of_d_in_KV_cache, KV_CACHE_SIZE, false);

    o_0.memset(bf16(-1.0f));
    o_0.sync_to_device();
    for (int i = 0; i < NT; i++) {
        time_utils::time_point start = time_utils::now();
        app_0(o_0, q_0, kv_0);
        // ert_cmd_state ret_state = run.wait2();
        // if (ret_state != ERT_CMD_STATE_COMPLETED) {
        //     header_print("error", "Kernel execution failed at iteration " + std::to_string(i) + " with state: " + app_0.cmd_state_map.at(ret_state));
        //     pass = false;
        //     break;
        // }
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    if (!pass) {
        header_print("error", "Kernel execution failed. Skipping verification.");
        return 1;
    }
    npu_time.first /= NT;
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);

    o_0.sync_from_device();
    // tensor_2d<bf16> o_0_tensor(o_0, DQ, 0);
    // tensor_2d<bf16> aux_out_tensor(aux_out_whole, DQ, 0);
    // buffer<bf16> c_sin(128);
    // for (int i = 0; i < 128; i++){
    //     error_metrics metrics = get_error_metrics(o_0_tensor[i], aux_out_tensor[i]);
    //     c_sin[i] = metrics.CosineSimilarity;
    // }
    // utils::print_matrix(c_sin, 8, 16);

    buffer<bf16> o_wanted((size_t)L_wanted * DQ);
    o_wanted.copy_from(o_0.data() + (L_begin - L_begin_chunked) * DQ, (size_t)L_wanted * DQ);
    buffer<bf16> aux_out((size_t)L_wanted * DQ);
    aux_out.copy_from(aux_out_whole.data() + (L_begin - L_begin_chunked) * DQ, (size_t)L_wanted * DQ);

    header_print("info", "Aux out");
    utils::print_matrix(aux_out, DQ);
    header_print("info", "O out");
    utils::print_matrix(o_wanted, DQ);

    // for (int i = 0; i < L_wanted * DQ; i++){
    //     if(o_wanted[i] == 0.0f){
    //         std::cout << "Warning: o_wanted[" << i << "] is zero, which may cause cosine similarity to be NaN." << std::endl;
    //     }
    // }

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