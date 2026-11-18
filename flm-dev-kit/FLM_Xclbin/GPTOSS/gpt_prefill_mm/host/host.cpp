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
#include <chrono>
#include <iostream>

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
#include <immintrin.h> // For AVX-512 intrinsics
// #include "debug_utils.hpp"
#include <stdfloat>
#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"
#include "mmRuntimeSequence.hpp"
#include "MM_CPU.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_queue.h"
#include "gemm_detail.hpp"

namespace po = boost::program_options;


template<typename T>
void print_errors(buffer<T> & y_0, buffer<T> & y_ref_0) {
    float rmse = get_rmse(y_0, y_ref_0);
    float relativeL1 = get_relativeL1(y_0, y_ref_0);
    float relativeL2 = get_relativeL2(y_0, y_ref_0);
    float cosine_similarity = get_cosine_similarity(y_0, y_ref_0);

    header_print("info", "Relative L1: " << relativeL1);
    header_print("info", "Relative L2: " << relativeL2);
    header_print("info", "Cosine similarity: " << cosine_similarity);
    header_print("info", "RMSE: " << rmse);
    header_print("info", "Finished running kernel");
    header_print("info", "Running kernel with bare call (run based).");
    
}

int main(int argc,  char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // // Program arguments parsing
    MatmulArgs args;
    if (!parse_matmul_args(argc, argv, args))
        return 1;  // parsing failed; message already printed

    /* ---- Parsed values are now in 'args' ---- */
    std::cout << "arch: " << args.arch << '\n'
              << "trace_size: " << args.trace_size << '\n'
              << "M_size: " << args.M_size << ", K_size: " << args.K_size
              << ", N_size: " << args.N_size << '\n'
              << "m: " << args.m << ", k: " << args.k << ", n: " << args.n << '\n'
              << "IS_B_ROW_MAJOR: " << args.IS_B_ROW_MAJOR << '\n'
              << "total_row: " << args.total_row << ", total_col: " << args.total_col << '\n';

    bool B_row_major = args.IS_B_ROW_MAJOR==1;
    bool ENABLE_AXI4 = args.ENABLE_AXI4==1;
    bool IS_B_COL_MAJOR_S_T_ROW_MAJOR = args.IS_B_COL_MAJOR_S_T_ROW_MAJOR==1;
    bool IS_BFLOAT16_OUT_TY = args.IS_BFLOAT16_OUT_TY==1;
    int ADD_BIAS = args.ADD_BIAS;
    int OUTPUT_MODE = args.OUTPUT_MODE;
    int OUTPUT_CLAMP = args.OUTPUT_CLAMP;
    


    if(IS_BFLOAT16_OUT_TY){
        assert(sizeof(dtype_out) == 2);
    }else{
        assert(sizeof(dtype_out) == 4);
    }

    std::cout << "IS B_row_major" <<B_row_major << "!!"<< std::endl;
    std::cout << "ENABLE_AXI4" << ENABLE_AXI4 << "!!"<< std::endl;
    std::cout << "ADD_BIAS: " << ADD_BIAS<< std::endl;
    std::cout << "OUTPUT_MODE: " << OUTPUT_MODE<< std::endl;
    std::cout << "OUTPUT_CLAMP: " << OUTPUT_CLAMP<< std::endl;


    int M = args.M_size;
    int K = args.K_size;
    int N = args.N_size;

    int W_VOLUME = M*K;
    int X_VOLUME = K*N;
    if(ADD_BIAS == 1){
        X_VOLUME +=N;
    }
    int Y_VOLUME = M*N;
    int Iterations = 100;
    int warmup_iterations = 20;
    bool pass = true;
    // NPU instance
    // npu_manager npu_instance(npu_device::device_npu2);

    std::string xclbin_name = "build/xclbins/mm_bf16_accf32.xclbin";
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = mm_app_mgr->create_app();
    npu_app app_1 = mm_app_mgr->create_app();

    // std::string xclbin_name = "build/xclbins/mm_bf16_accf32.xclbin";
    // npu_app_desc accel_desc_0, accel_desc_1;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "mm_bf16_accf32";
    // accel_desc_1.xclbin_name =  xclbin_name;
    // accel_desc_1.app_name  = "mm_bf16_accf32_generated"; 



    // npu_app app_0 = npu_instance.create_app(accel_desc_0);
    // npu_app app_1 = npu_instance.create_app(accel_desc_1);
    uint32_t rtp_address = 4096; // offset right after stack size
    uint32_t rtp_sync_lock_id = 10; // the rtp sync lock
    // app_0.instr_seq->from_file("build/insts/mm_bf16_accf32.txt");
    //app_0.instr_seq->interpret();
    // buffer<int> seq_python  = app_0.instr_seq->dump().cast_to<int>();

    float output_clamp_min = static_cast<std::bfloat16_t>(utils::getRand(-2, -1));
    float output_clamp_max =  static_cast<std::bfloat16_t>(utils::getRand(1, 2));
    generate_mm_sequence<dtype_in, dtype_out>(*app_0.seq(), M,K,N, 
        args.m, args.k, args.n, 
        args.r, args.s, args.t,
        rtp_address, rtp_sync_lock_id, 
        args.total_row, args.total_col, 
        0, (ADD_BIAS == 1) ? N : 0, 0, 0,
        B_row_major, ENABLE_AXI4, false, // set to false for all time
        (ADD_BIAS == 1)? true: false, OUTPUT_MODE,
        OUTPUT_CLAMP, output_clamp_min, output_clamp_max, // set clamp range to [-10, 10], this is a common setting for activation clamp
        "origin_dimension.txt"
    );


    // std::cout << "C++ generation" << std::endl;
    // app_0.instr_seq->interpret();



    buffer<dtype_in> w_0 = app_0.create_bo_buffer<dtype_in>(W_VOLUME); // A matrix
    buffer<dtype_in> x_0 = app_0.create_bo_buffer<dtype_in>(X_VOLUME); // B matrix
    buffer<dtype_out> y_0 = app_0.create_bo_buffer<dtype_out>(Y_VOLUME); // A*B matrix result

    for (int i = 0; i < W_VOLUME; i++){
        w_0[i] =  static_cast<std::bfloat16_t>(utils::getRand(-1, 1));

    }
    for (int i = 0; i < X_VOLUME; i++){
        x_0[i] =  static_cast<std::bfloat16_t>(utils::getRand(-1, 1));
    }

    if(IS_B_COL_MAJOR_S_T_ROW_MAJOR){
        // server assumption is required for this mode
        // 1. first, it treat the data in B matrix as col major
        
        // 2. It first reorder the col-major B matrix data into kxn block col-major, and the blocks are still in col major, and within the blocks, the data is also in col major
        // stride: [N_size/n,K_size/k  ,n,     k]
        // offset: [K_size*n,k         ,K_SIZE, 1]
        buffer<dtype_in> x_0_temp(X_VOLUME);
        
        size_t dst_index = 0;
        for(int i = 0; i < args.N_size/args.n; i++){
            for(int j= 0; j <args.K_size/args.k; j++){
                for(int l = 0; l <args.n; l++){
                    for(int m = 0; m < args.k; m++){
                        int src_index = m*1 + l*args.K_size + j*(args.k) + i * args.K_size * args.n;
                        x_0_temp[dst_index] = x_0[src_index];
                        dst_index++;
                    }
                }
            }
        }

        // now, need to reorder the k*n each data block into small sxt blocks, but sxt blocks are change from col-major to row major
        





        //  todo, some further data manipulation is required
        // std::cout << "Not implement for IS_B_COL_MAJOR_S_T_ROW_MAJOR yet!" << std::endl;
        // return -1; 
    }
    header_print("info", "Calculate reference for " << M << "x" << K << "x" << N);
    buffer<dtype_out> y_ref_0(Y_VOLUME);    
    // if(B_row_major){
    //     linear(y_ref_0, w_0, x_0, M, K, N);
    // }else{
    //     linear_colMajorW(y_ref_0, w_0, x_0, M, K, N);
    // }
    if(ADD_BIAS == 1){
        if(B_row_major){
            linear_row_major_w_with_bias_template<dtype_out>(y_ref_0.data(), w_0.data(), x_0.data(), M, K, N);
        }else{
            linear_col_major_w_with_bias_template<dtype_out>(y_ref_0.data(), w_0.data(), x_0.data(), M, K, N);
        }
    }else{
        if(B_row_major){
            linear_row_major_w_template<dtype_out>(y_ref_0.data(), w_0.data(), x_0.data(), M, K, N);
        }else{
            linear_col_major_w_template<dtype_out>(y_ref_0.data(), w_0.data(), x_0.data(), M, K, N);
        }


    }   

    if(OUTPUT_MODE  == 1){
        // do a gelu on all y_ref_0
        gelu_bfloat16_ref_scalar(y_ref_0.data(), y_ref_0.data(), M*N);
    }else if(OUTPUT_MODE == 2){
        silu_bfloat16_ref_scalar(y_ref_0.data(), y_ref_0.data(), M*N);
    }else if(OUTPUT_MODE == 3){
        // do a sigmoid
        sigmoid_bfloat16_ref_scalar(y_ref_0.data(), y_ref_0.data(), M*N);
    }


    if(OUTPUT_CLAMP == 1){
        // clamp the output to the range of [output_clamp_min, output_clamp_max]
        for(int i = 0; i < M*N; i++){
            float val = static_cast<float>(y_ref_0[i]);
            val = std::max(val, output_clamp_min);
            val = std::min(val, output_clamp_max);
            y_ref_0[i] = static_cast<dtype_out>(val);
        }
    }

    
    // // pure debug
    // for(int i = 0; i < M; i++){
    //     // override with the bias
    //     memcpy(y_ref_0.data() + i*N, x_0.data(), N*sizeof(dtype_in));
    // }

    int m = args.m;
    int n = args.n;
    int k = args.k;
    unsigned long data_size = M * K * N / n / 4 + K * N * M / m / 8;
	
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call (direct based).");
    time_utils::time_with_unit npu_time = {0.0, "us"};


    w_0.sync_to_device();
    x_0.sync_to_device();
    time_utils::time_point start = time_utils::now();
    app_0(w_0, x_0, y_0);
    time_utils::time_point stop = time_utils::now();    
    y_0.sync_from_device();    
    MSG_BOX_LINE(60, "NPU time raw: " << stop-start << " us");
    // header_print("INFO", "y res");
    // utils::print_matrix(y_0, 64);
    // header_print("INFO", "y ref");
    // utils::print_matrix(y_ref_0, 64);    
    print_errors(y_0, y_ref_0);

    // runlist to check performance 
    npu_time.first = 0;
    MSG_BOX_LINE(60, "Running kernel with bare call  (run based).");

    for(int i = 0; i < Iterations; i++){

         auto start = std::chrono::high_resolution_clock::now();
        app_0(w_0, x_0, y_0);
        auto end = std::chrono::high_resolution_clock::now();
        npu_time.first +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();


    }
    // npu_time.first = 0;
    // xrt::runlist runlist = npu_instance.create_runlist(app_0);
    // y_0.memset(0);
    // for(int i = 0; i<Iterations; i++){
    //     xrt::run run_0 = app_0.create_run(w_0, x_0, y_0);
    //     runlist.add(run_0);
    // }

    // MSG_BOX_LINE(60, "Running kernel with bare call  (run based).");
    // npu_time.first = 0;
    // {
    //     time_utils::time_point start = time_utils::now();
    //     runlist.execute();
    //     runlist.wait();
    //     time_utils::time_point stop = time_utils::now();
    //     npu_time.first += time_utils::duration_us(start, stop).first;
    // }
    y_0.sync_from_device();
    // assert y_0 valie is between min_clamp and max_clamp if output clamp is enabled
    if(OUTPUT_CLAMP == 1){
        for(int i = 0; i < M*N; i++){
            float val = static_cast<float>(y_0[i]);
            if(val < output_clamp_min || val > output_clamp_max){
                std::cout << "Value out of clamp range at index " << i << ": " << val << " not in [" << output_clamp_min << ", " << output_clamp_max << "]" << std::endl;
                pass = false;
                break;
            }
        }
    }

    MSG_BOX_LINE(60, "NPU time with runlist: " << npu_time.first << " us");
    header_print("info", "Finished running kernel");
    print_errors(y_0, y_ref_0);

    // some stats
    auto A_repeat = (N/(8*n));
    auto B_repeat = (M/(4*m));
    auto read_data_size = (M*K*A_repeat  + K*N*B_repeat)*2.0;
    auto write_data_size = M*N*2;

    utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N), Iterations);
    header_print("info", "Running runtime test with runtime sequence generation.");    
    auto average_time_us = npu_time.first/Iterations;
    auto average_time_s = average_time_us/(1e6);
    std::cout <<"average time in second " << average_time_s << std::endl;
    double actual_read_utilization_GBps = (read_data_size/(1e9) ) /  average_time_s;
    double actual_write_utilization_GBps = (write_data_size/(1e9) /average_time_s);

    MSG_BOX_LINE(60, "Actual read memory utilization " << actual_read_utilization_GBps);
    MSG_BOX_LINE(60, "Actual write memory utilization " << actual_write_utilization_GBps);

    // std::cout << "**Running runtime sequence generation**" << std::endl;
    // // Testing of change dimension at runtime


    // // // another different matrix size
    // // int M_2 = 2048; 
    // // int K_2 = K/2 ;
    // // int N_2 = 1024;
    // // assert(M_2 <=M);
    // // assert(K_2 <= K);
    // // assert(N_2 <= N);
    // // generate_mm_sequence<dtype_in, dtype_out>(*app_1.instr_seq, M_2,K_2,N_2, 
    // //     args.m, args.k, args.n, 
    // //     args.r, args.s, args.t,
    // //     rtp_address, rtp_sync_lock_id, 
    // //     args.total_row, args.total_col, 
    // //     0,0,0,
    // //     B_row_major,ENABLE_AXI4,
    // //     "modified_dimension.txt"
    // // );

    // int A_size = M*K;
    // int B_size = K*N;
    // int C_Size = M*N;
    // buffer<dtype_in> A_0= app_1.create_bo_buffer<dtype_in>(A_size, 3);
    // buffer<dtype_in> B_0 = app_1.create_bo_buffer<dtype_in>(B_size, 4);
    // buffer<dtype_out> C_0 = app_1.create_bo_buffer<dtype_out>(C_Size, 5);

    // for(auto i = 0; i < A_size; i++){
    //     A_0[i] = static_cast<std::bfloat16_t>(utils::getRand(-1, 1));
    // }
    // for(auto i = 0; i < B_size; i++){
    //     B_0[i] = static_cast<std::bfloat16_t>(utils::getRand(-1, 1));
    // }    
    // header_print("info", "calculating the reference value");
    // if(B_row_major){
    //     linear_row_major_w_template<dtype_out>(y_ref_0.data(), A_0.data(), B_0.data(), M, K, N);
    // }else{
    //     linear_col_major_w_template<dtype_out>(y_ref_0.data(), A_0.data(), B_0.data(), M, K, N);
    // }



    // bool B_COL_in_K_N_block_col_major_order = false;
    // if(B_COL_in_K_N_block_col_major_order && B_row_major==false){
        
    //     // 1. first, it treat the data in B matrix as col major, which reflect in the if else statement
        
    //     // 2. It first reorder the col-major B matrix data into kxn block col-major, and the blocks are still in col major, and within the blocks, the data is also in col major
    //     // stride: [N_size/n,K_size/k  ,n,     k]
    //     // offset: [K_size*n,k         ,K_SIZE, 1]
    //     buffer<dtype_in> B_temp(X_VOLUME);
        
    //     size_t dst_index = 0;
    //     for(int i = 0; i < args.N_size/args.n; i++){
    //         for(int j= 0; j <args.K_size/args.k; j++){
    //             for(int l = 0; l <args.n; l++){
    //                 for(int m = 0; m < args.k; m++){
    //                     int src_index = m*1 + l*args.K_size + j*(args.k) + i * args.K_size * args.n;
    //                     B_temp[dst_index] = B_0[src_index];
    //                     dst_index++;
    //                 }
    //             }
    //         }
    //     }
    //     // store back
    //     memcpy(B_0.data(), B_temp.data(), B_size * sizeof(dtype_in));
    //     // now, need to reorder the k*n each data block into small sxt blocks, but sxt blocks are change from col-major to row major
        
    //     generate_mm_sequence<dtype_in, dtype_out>(*app_1.instr_seq, M,K,N, 
    //         args.m, args.k, args.n, 
    //         args.r, args.s, args.t,
    //         rtp_address, rtp_sync_lock_id, 
    //         args.total_row, args.total_col, 
    //         0,0,0,        
    //         B_row_major, ENABLE_AXI4, true, // since this mode we change the blocks 
    //         "origin_dimension.txt"
    //     );
    // }else{
    //     generate_mm_sequence<dtype_in, dtype_out>(*app_1.instr_seq, M,K,N, 
    //         args.m, args.k, args.n, 
    //         args.r, args.s, args.t,
    //         rtp_address, rtp_sync_lock_id, 
    //         args.total_row, args.total_col, 
    //         0,0,0,        
    //         B_row_major, ENABLE_AXI4, false, // set to false for all time
    //         "origin_dimension.txt"
    //     );
    // }



    // buffer<int> seq_0 = app_0.instr_seq->dump().cast_to<int>();
    // buffer<int> seq_1 = app_1.instr_seq->dump().cast_to<int>();

    // MSG_BONDLINE(60);
    // // app_0.instr_seq->interpret();
    // MSG_BONDLINE(60);
    // // app_1.instr_seq->interpret();
    // MSG_BONDLINE(60);    
    // // compare the two sequences
    // if (utils::compare_vectors_integer(seq_0, seq_1) > 0){
    //     std::cout << "Sequences are not the same!" << std::endl;
    //     return -1;
    // } else {
    //     std::cout << "Sequences are the same!" << std::endl;
    // }


    
    // // first run the original dimension
    // A_0.sync_to_device();
    // B_0.sync_to_device();
    // app_1(A_0, B_0, C_0);
    // C_0.sync_from_device();

    // print_errors(C_0, y_ref_0);


    // // now, same bitstream but different instruction file
    // app_1.instr_seq->from_file("modified_dimension.txt"); // the file that was generated earlier
    // int A_2_size = M*K;
    // int B_2_size = K*N;
    // int C_2_size = M * N;
    // // Manually call the destructor for an object:
    // A_0.~buffer<dtype_in>(); // manually call the destructor NOTE: There is an issue with memory leak if don't do so
    // B_0.~buffer<dtype_in>();
    // C_0.~buffer<dtype_out>();
    // A_0= app_1.create_bo_buffer<dtype_in>(A_2_size, 3);
    // B_0 = app_1.create_bo_buffer<dtype_in>(B_2_size, 4);
    // C_0 = app_1.create_bo_buffer<dtype_out>(C_2_size, 5);
    // // A_0.resize(A_2_size);
    // // B_0.resize(B_2_size);
    // // C_0.resize(C_2_size);
    // for(auto i = 0; i < A_2_size; i++){
    //     A_0[i] =static_cast<std::bfloat16_t>(utils::getRand(-1, 1));
    // }
    // for(auto i = 0; i < B_2_size; i++){
    //     B_0[i] = static_cast<std::bfloat16_t>(utils::getRand(-1, 1));
    // }
    // header_print("info", "calculating the reference value for second bitstream");
    // buffer<dtype_out> y_ref_2(C_2_size);    
    // if(B_row_major){
    //     linear_row_major_w_template<dtype_out>(y_ref_2.data(), A_0.data(), B_0.data(), M_2, K_2, N_2);
    // }else{
    //     linear_col_major_w_template<dtype_out>(y_ref_2.data(), A_0.data(), B_0.data(), M_2, K_2, N_2);
    // }
    // header_print("info", "sync data to npu for second bitstream");
    // A_0.sync_to_device();
    // B_0.sync_to_device();
    // app_1(A_0, B_0, C_0);
    // C_0.sync_from_device();

    // print_errors(C_0, y_ref_2);


    std::cout << "EXISTING" <<std::endl;


    return 0;
}



