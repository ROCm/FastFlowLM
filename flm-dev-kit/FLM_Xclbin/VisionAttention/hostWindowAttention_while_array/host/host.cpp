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

#include <cassert>
#include <stdfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <bit>
#include <cstdint>
#include <vector>
#include <thread>
#include <cinttypes>


#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_queue.h"

namespace po = boost::program_options;
#include "error_measure.hpp"
#include <immintrin.h>
#include <algorithm>
#include "WindowAttentionRuntimeSequence.hpp"

// Helper function to compute ceiling division
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}





void window_attention_reference(
    std::bfloat16_t* q_buffer,
    std::bfloat16_t* k_buffer,
    std::bfloat16_t* v_buffer,
    std::bfloat16_t* output_buffer,
    int L_SEQ,
    int S_SEQ,
    int D_H,
    int NUM_HEAD,
    const int BLOCK_L_Q,
    const int BLOCK_L_V,

    std::vector<float> & S,
    std::vector<float> & P,
    bool causal_mask = false
){

    float scale = 1.0f / std::sqrt((float)D_H);
    int stride_row = D_H * NUM_HEAD;
    
    // Loop over windows
    // Assumes 1-to-1 mapping of Q window to K/V window
    // i_start and j_start progress in lockstep
    for (int i_start = 0; i_start < L_SEQ; i_start += BLOCK_L_Q) {
        // Calculate corresponding K/V block start
        // Assuming linear correspondence for window attention where each Q block attends to corresponding K/V block
        int j_start = (i_start / BLOCK_L_Q) * BLOCK_L_V;
        
        if (j_start >= S_SEQ) break; // Should not happen if lengths aligned

        int i_end = std::min(i_start + BLOCK_L_Q, L_SEQ);
        int current_block_q_len = i_end - i_start;
        
        int j_end = std::min(j_start + BLOCK_L_V, S_SEQ);
        int current_block_v_len = j_end - j_start;

        S.resize(current_block_q_len * current_block_v_len);
        P.resize(current_block_q_len * current_block_v_len);

        for(int h = 0; h < NUM_HEAD; h++){

            // 1. Compute Scores for this window
            // std::vector<float> S(current_block_q_len * current_block_v_len);
            
            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                for (int j = 0; j < current_block_v_len; ++j) {
                    int global_j = j_start + j;
                    float dot = 0.0f;
                    for (int d = 0; d < D_H; ++d) {
                        dot += (float)q_buffer[global_i * stride_row + h*D_H + d] * (float)k_buffer[global_j * stride_row + h*D_H + d];
                    }
                    S[i * current_block_v_len + j] = dot * scale;
                }
            }

            // 2. Softmax row-wise (with optional causal mask)
            //std::vector<float> P(current_block_q_len * current_block_v_len);

            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < current_block_v_len; ++j) {
                    int global_j = j_start + j;
                    // Causal mask: query at global_i can only attend to keys at global_j <= global_i
                    if (causal_mask && global_j > global_i) continue;
                    if (S[i * current_block_v_len + j] > max_val) max_val = S[i * current_block_v_len + j];
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < current_block_v_len; ++j) {
                    int global_j = j_start + j;
                    if (causal_mask && global_j > global_i) {
                        P[i * current_block_v_len + j] = 0.0f;
                        continue;
                    }
                    float val = std::exp(S[i * current_block_v_len + j] - max_val);
                    P[i * current_block_v_len + j] = val;
                    sum_exp += val;
                }

                // Normalize
                for (int j = 0; j < current_block_v_len; ++j) {
                    if (causal_mask && (j_start + j) > global_i) continue;
                    P[i * current_block_v_len + j] /= sum_exp;
                }
            }

            // 3. Compute Output = P * V
            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                for (int d = 0; d < D_H; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < current_block_v_len; ++j) {
                        int global_j = j_start + j;
                        sum += P[i * current_block_v_len + j] * (float)v_buffer[global_j * stride_row + h*D_H + d];
                    }
                    output_buffer[global_i * stride_row + h*D_H + d] = (std::bfloat16_t)sum;
                }
            }
        }
    }
}


int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    desc.add_options()
        ("trace_size", po::value<int>()->default_value(2048*1024))
        ("NUM_OF_COLUMNS", po::value<int>()->default_value(1))
        ("NUM_OF_CT_PER_COLUMN", po::value<int>()->default_value(2))
        ("r", po::value<int>()->default_value(8))
        ("s", po::value<int>()->default_value(8))
        ("t", po::value<int>()->default_value(8))
        ("DH", po::value<int>()->default_value(80))
        ("LQ", po::value<int>()->default_value(64))
        ("LK", po::value<int>()->default_value(64))
        ("LOW_PRECISION", po::value<int>()->default_value(0))
        ("USE_CASUAL_MASK", po::value<int>()->default_value(0))
        ("NUM_HEAD", po::value<int>()->default_value(1))
        ("SHIM_L_SEQ", po::value<int>()->default_value(64))
        ("sHIM_S_SEQ", po::value<int>()->default_value(64));

    po::positional_options_description p;
    p.add("device", 1);
    p.add("trace_size", 1);
    p.add("NUM_OF_COLUMNS", 1);
    p.add("NUM_OF_CT_PER_COLUMN", 1);
    p.add("r", 1);
    p.add("s", 1);
    p.add("t", 1);
    p.add("DH", 1);
    p.add("LQ", 1);
    p.add("LK", 1);
    p.add("LOW_PRECISION", 1);
    p.add("USE_CASUAL_MASK", 1);
    p.add("NUM_HEAD", 1);
    p.add("SHIM_L_SEQ", 1);
    p.add("sHIM_S_SEQ", 1);

    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);


    int Iterations = 200;
    
    // Note: for now, do a single head attentiom implemetnation first
    //NOTE: sync with the python version 
    int D_H = vm["DH"].as<int>();
    int LQ = vm["LQ"].as<int>();
    int LK = vm["LK"].as<int>();

    int SHIM_L_SEQ = vm["SHIM_L_SEQ"].as<int>();
    int sHIM_S_SEQ = vm["sHIM_S_SEQ"].as<int>();
    int NUM_HEAD = vm["NUM_HEAD"].as<int>();
    int NUM_OF_COLUMNS = vm["NUM_OF_COLUMNS"].as<int>();
    int NUM_OF_CT_PER_COLUMN = vm["NUM_OF_CT_PER_COLUMN"].as<int>();
    int USE_CASUAL_MASK = vm["USE_CASUAL_MASK"].as<int>();

    //NOTE: for now
    int L_SEQ_padded = ceil_div(SHIM_L_SEQ, LQ)* LQ; // query length
    int S_SEQ_padded = ceil_div(sHIM_S_SEQ, LK)* LK; // key/value length   
    
    assert(L_SEQ_padded % LQ ==0);
    assert(S_SEQ_padded % LK ==0);
    std::cout << "L_SEQ_padded is: " << L_SEQ_padded << std::endl;
    std::cout << "S_SEQ_padded is: " << S_SEQ_padded <<std::endl;



    // let us now allocate the buffer and randomly assign value to it
    std::bfloat16_t* q_buffer = (std::bfloat16_t*) malloc( L_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));
    std::bfloat16_t* k_buffer = (std::bfloat16_t*) malloc( S_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));
    std::bfloat16_t* v_buffer = (std::bfloat16_t*) malloc( S_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));
    std::bfloat16_t* ref_flash_attention_output_buffer = (std::bfloat16_t*) malloc( L_SEQ_padded * D_H*NUM_HEAD  * sizeof(std::bfloat16_t));


    {
        // assign random value  of -1 to 1
        float min_val = -5.0f;
        float max_val = 5.0f;
        for ( int i = 0; i < SHIM_L_SEQ * D_H * NUM_HEAD; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            q_buffer[i] = std::bfloat16_t(val);
        }
        for ( int i = 0; i < sHIM_S_SEQ * D_H * NUM_HEAD; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            k_buffer[i] = std::bfloat16_t(val);
        }
        for ( int i = 0; i < sHIM_S_SEQ * D_H * NUM_HEAD; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            v_buffer[i] = std::bfloat16_t(val);
        }
    }
    std::vector<float> S;
    std::vector<float> P;

    if(USE_CASUAL_MASK == 0){
        // now, let us call the window attention implementation 
        window_attention_reference(
            q_buffer,
            k_buffer,
            v_buffer,
            ref_flash_attention_output_buffer,
            SHIM_L_SEQ,
            sHIM_S_SEQ,
            D_H,
            NUM_HEAD,
            LQ,
            LK,
            S,
            P
        );

    }else{
        std::cout << "use causal mask" << std::endl;
        // window attention with causal mask applied on
        window_attention_reference(
            q_buffer,
            k_buffer,
            v_buffer,
            ref_flash_attention_output_buffer,
            SHIM_L_SEQ,
            sHIM_S_SEQ,
            D_H,
            NUM_HEAD,
            LQ,
            LK,
            S,
            P,
            /*causal_mask=*/true
        );
    }

    std::cout << "************************"<< std::endl;

    //load the npu instance

// NPU instance
    // npu_device device_type = npu_device::device_npu2;
    // if (vm["device"].as<std::string>() == "npu1") {
    //     device_type = npu_device::device_npu1;
    // }
    // npu_manager npu_instance(device_type);
    // if (VERBOSE >= 1){
    //     npu_instance.get_npu_power(true);
    //     npu_instance.print_npu_info();
    // }

    std::string xclbin_name = "build/xclbins/attention_bf16.xclbin";
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = mm_app_mgr->create_app();

    // npu_app_desc accel_desc_0, accel_desc_1;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "attention_bf16";


    // npu_app app_0 = npu_instance.create_app(accel_desc_0);

    // app_0.instr_seq->from_file("build/insts/attention_bf16.txt");
    // buffer<float> seq_0 = app_0.instr_seq->dump().cast_to<float>();

    window_attention_runtimeSequence<std::bfloat16_t, std::bfloat16_t>(
        *app_0.seq(),
        1, // for now, single batch
        LQ, LK, D_H,
        NUM_HEAD, NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN,
        
        {SHIM_L_SEQ}, // L_SEQ_per_batch
        {sHIM_S_SEQ}, // S_SEQ_per_batch,
        0,0,0,0,
        0, // batch_Q_offset
        1, // batch_K_offset
        2, // batch_V_offset
        3 // batch_O_offset
    );

    buffer<std::bfloat16_t> q_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(L_SEQ_padded * D_H*NUM_HEAD);
    buffer<std::bfloat16_t> k_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(S_SEQ_padded * D_H*NUM_HEAD);
    buffer<std::bfloat16_t> v_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(S_SEQ_padded * D_H*NUM_HEAD);

    buffer<std::bfloat16_t> output_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(L_SEQ_padded * D_H*NUM_HEAD);    

    {
        memcpy(q_buffer_npu.data(), q_buffer, L_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));
        memcpy(k_buffer_npu.data(), k_buffer, S_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));
    
        memcpy(v_buffer_npu.data(), v_buffer, S_SEQ_padded * D_H*NUM_HEAD * sizeof(std::bfloat16_t));

    }


    q_buffer_npu.sync_to_device();
    k_buffer_npu.sync_to_device();
    v_buffer_npu.sync_to_device();

    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    time_utils::time_with_unit npu_time = {0.0, "us"};
    time_utils::time_point start = time_utils::now();
    app_0(q_buffer_npu, k_buffer_npu, v_buffer_npu, output_buffer_npu);
    time_utils::time_point stop = time_utils::now();

    npu_time.first += time_utils::duration_us(start, stop).first;

    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(60);
    output_buffer_npu.sync_from_device();   






    std::bfloat16_t* npu_output_ptr = output_buffer_npu.data();




    print_error_metrics<std::bfloat16_t, std::bfloat16_t>(
        npu_output_ptr,
        ref_flash_attention_output_buffer,
        1,
        SHIM_L_SEQ,
        D_H*NUM_HEAD,
        SHIM_L_SEQ,
        D_H*NUM_HEAD

    );



    // now, run multiple times
    npu_time.first = 0;

    for(int i = 0; i < Iterations; i++){
        auto start = std::chrono::high_resolution_clock::now();
        app_0(q_buffer_npu, k_buffer_npu, v_buffer_npu, output_buffer_npu);
        auto end = std::chrono::high_resolution_clock::now();
        npu_time.first += time_utils::duration_us(start, end).first;

    }
    auto average_time_us = npu_time.first / Iterations;
    MSG_BOX_LINE(60, "Average NPU time over " << Iterations << " iterations: " << average_time_us << " us");
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "Average NPU time over " << Iterations << " iterations: " << average_time_us << " us");
    MSG_BONDLINE(60);


    // calculate the memory usage
    auto total_memory_bytes = sizeof(std::bfloat16_t) * (q_buffer_npu.size() + k_buffer_npu.size() + v_buffer_npu.size());
    auto total_memory_gb = (float)total_memory_bytes / (1024 * 1024 * 1024);
    MSG_BOX_LINE(60, "Total memory usage: " << total_memory_gb << " GB");
    MSG_BONDLINE(60);

    



    return 0;
}

