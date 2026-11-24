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
#include <omp.h>

#include "utils/debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils/npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_queue.h"
#include "error_measure.hpp"
#include <immintrin.h>
#include <algorithm>
#include "attention_avx512.hpp"
namespace po = boost::program_options;

#include "AttentionRumtimeSequence.hpp"
void attention_reference(
    std::bfloat16_t* q_buffer,
    std::bfloat16_t* k_buffer,
    std::bfloat16_t* v_buffer,
    std::bfloat16_t* output_buffer,
    int L_SEQ,
    int S_SEQ,
    int D_H,
    int num_head

){
    float scale = 1.0f / std::sqrt((float)D_H);
    int total_dim = D_H * num_head;

    // Process each head independently
    for (int h = 0; h < num_head; ++h) {
        int head_offset = h * D_H;

        // 1. Compute Scores S = Q * K^T / sqrt(D_H) for this head
        std::vector<float> S(L_SEQ * S_SEQ);

        for (int i = 0; i < L_SEQ; ++i) {
            for (int j = 0; j < S_SEQ; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < D_H; ++d) {
                    dot += (float)q_buffer[i * total_dim + head_offset + d] * (float)k_buffer[j * total_dim + head_offset + d];
                }
                S[i * S_SEQ + j] = dot * scale;
            }
        }

        // 2. Softmax row-wise
        std::vector<float> P(L_SEQ * S_SEQ);
        for (int i = 0; i < L_SEQ; ++i) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < S_SEQ; ++j) {
                if (S[i * S_SEQ + j] > max_val) max_val = S[i * S_SEQ + j];
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < S_SEQ; ++j) {
                float val = std::exp(S[i * S_SEQ + j] - max_val);
                P[i * S_SEQ + j] = val;
                sum_exp += val;
            }

            for (int j = 0; j < S_SEQ; ++j) {
                P[i * S_SEQ + j] /= sum_exp;
            }
        }

        // 3. Output O = P * V for this head
        for (int i = 0; i < L_SEQ; ++i) {
            for (int d = 0; d < D_H; ++d) {
                float val = 0.0f;
                for (int j = 0; j < S_SEQ; ++j) {
                    val += P[i * S_SEQ + j] * (float)v_buffer[j * total_dim + head_offset + d];
                }
                output_buffer[i * total_dim + head_offset + d] = (std::bfloat16_t)val;
            }
        }
    }
}









void flash_attention_progressive(
    std::bfloat16_t* q_buffer,
    std::bfloat16_t* k_buffer,
    std::bfloat16_t* v_buffer,
    std::bfloat16_t* output_buffer,
    int L_SEQ,
    int S_SEQ,
    int D_H,
    const int BLOCK_L_Q,
    const int BLOCK_L_V,
    int num_head,

    // debug parameter: Now holds the full block (Q_len * V_len)
    std::vector<float> & S_ij_row,
    std::vector<float> & m_i,
    std::vector<float> & m_new_vec,
    std::vector<float> &exp_old_vec,
    std::vector<float> & l_i,
    std::vector<float> &P_ij_unnorm_flat,
    std::vector<float> &O_accum
){

    float scale = 1.0f / std::sqrt((float)D_H);
    int total_dim = D_H * num_head;
    //NOTE: for debug, set scale to 1
    //float scale = 1.0f;
    
    // Process each head independently
    for (int h = 0; h < num_head; ++h) {
        int head_offset = h * D_H;
        
    // Loop over Q blocks
    for (int i_start = 0; i_start < L_SEQ; i_start += BLOCK_L_Q) {
        int i_end = std::min(i_start + BLOCK_L_Q, L_SEQ);
        int current_block_q_len = i_end - i_start;

        // Initialize Output Accumulator to 0
        O_accum.resize(current_block_q_len * D_H);
        for(int i =0; i< current_block_q_len * D_H; ++i){
            O_accum[i] = 0.0f;
        }
        
        // m_i (running max) and l_i (running sum) for the current block
        m_i.resize(current_block_q_len);
        l_i.resize(current_block_q_len);        
        for(int i = 0; i<current_block_q_len; ++i){
            m_i[i] = -std::numeric_limits<float>::infinity();
            l_i[i] = 0.0f;
        }


        // //FIXME:
        // // TODO: note, debug p urpose, sync with hose
        // for(int i = 0; i < current_block_q_len * D_H; ++i){
        //    O_accum[i] =0.5f;
        // }
        // for(int i = 0; i < current_block_q_len; ++i){
        //     m_i[i]  =(i*3)%20;
        // }

        // Loop over K/V blocks
        for (int j_start = 0; j_start < S_SEQ; j_start += BLOCK_L_V) {
            int j_end = std::min(j_start + BLOCK_L_V, S_SEQ);
            int current_block_v_len = j_end - j_start;

            // ---------------------------------------------------------
            // STEP 1: Resize and Compute the FULL Block of Scores
            // ---------------------------------------------------------
            
            // Resize buffer to hold the Q_block * V_block matrix
            S_ij_row.resize(current_block_q_len * current_block_v_len);

            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                for (int j = 0; j < current_block_v_len; ++j) {
                    int global_j = j_start + j;
                    
                    float dot = 0.0f;
                    for (int d = 0; d < D_H; ++d) {
                        dot += (float)q_buffer[global_i * total_dim + head_offset + d] * (float)k_buffer[global_j * total_dim + head_offset + d];
                    }
                    
                    // Store in the flat vector (row-major: i * width + j)
                    S_ij_row[i * current_block_v_len + j] = dot * scale;
                }
            }

            // ---------------------------------------------------------
            // STEP 2: Apply FlashAttention Tiling Trick (Block-wise)
            // ---------------------------------------------------------
            
            // 2.1 Find max of S_ij for each row in the block
            std::vector<float> m_ij_block_vec(current_block_q_len);
            for (int i = 0; i < current_block_q_len; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < current_block_v_len; ++j) {
                    float val = S_ij_row[i * current_block_v_len + j];
                    if (val > max_val) max_val = val;
                }
                m_ij_block_vec[i] = max_val;
            }

            // 2.2 Compute m_new and exp_old for each row
            m_new_vec.resize(current_block_q_len);
            exp_old_vec.resize(current_block_q_len);
            for (int i = 0; i < current_block_q_len; ++i) {
                // int global_i = i_start + i;
                float m_old = m_i[i];
                float m_new = std::max(m_old, m_ij_block_vec[i]);
                m_new_vec[i] = m_new;

                // do a clamp of x to -80 and 80 to avoid overflow

                exp_old_vec[i] = std::exp(m_old - m_new);
         
            }

            // 2.3 Compute P_ij_unnormalized for the whole block
            P_ij_unnorm_flat.resize(current_block_q_len * current_block_v_len);
            for (int i = 0; i < current_block_q_len; ++i) {
                float m_new = m_new_vec[i];
                for (int j = 0; j < current_block_v_len; ++j) {
                    float s = S_ij_row[i * current_block_v_len + j];
                    P_ij_unnorm_flat[i * current_block_v_len + j] = std::exp(s - m_new);
                }
            }

            // 2.4 Update L_i for all rows
            for (int i = 0; i < current_block_q_len; ++i) {
                // int global_i = i_start + i;
                float row_sum = 0.0f;
                for (int j = 0; j < current_block_v_len; ++j) {
                    row_sum += P_ij_unnorm_flat[i * current_block_v_len + j];
                }
                l_i[i] = exp_old_vec[i] * l_i[i] + row_sum;
            }

            // 2.5 Update O_accum for all rows
            for (int i = 0; i < current_block_q_len; ++i) {
                int global_i = i_start + i;
                float exp_old = exp_old_vec[i];
                
                // Scale old O
                for (int d = 0; d < D_H; ++d) {
                    O_accum[i * D_H + d] *= exp_old;
                }
  
                // Add P * V contribution
                for (int d = 0; d < D_H; ++d) {
                    float pv_sum = 0.0f;
                    for (int j = 0; j < current_block_v_len; ++j) {
                        int global_j = j_start + j;
                        float p_val = P_ij_unnorm_flat[i * current_block_v_len + j];
                        float v_val = (float)v_buffer[global_j * total_dim + head_offset + d];
                        pv_sum += p_val * v_val;
                    }
                    O_accum[i * D_H + d] += pv_sum;
                }
            }

            // 2.6 Update m_i state
            for (int i = 0; i < current_block_q_len; ++i) {
                m_i[i] = m_new_vec[i];
            }
        }

        // 4. Final normalization for this block
        for (int i = 0; i < current_block_q_len; ++i) {
            int global_i = i_start + i;
            for (int d = 0; d < D_H; ++d) {
                output_buffer[global_i * total_dim + head_offset + d] = (std::bfloat16_t)(O_accum[i * D_H + d] / l_i[i]);
            }
        }
    }
    } // end of head loop
}

// Helper function to compute ceiling division
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    int random_seed = 0;
    srand(random_seed);
    std::string output_data_dir = "./data";

    // create the output directory if not exists
    std::string mkdir_cmd = "mkdir -p " + output_data_dir;
    system(mkdir_cmd.c_str());



    // Configure OpenMP threads (default: 16, override with OMP_NUM_THREADS env var)
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        setenv("OMP_NUM_THREADS", "16", 0);
        omp_set_num_threads(16);
    }
    
    int num_threads = std::thread::hardware_concurrency();
    std::cout << "==================================================" << std::endl;
    std::cout << "OpenMP Configuration:" << std::endl;
    std::cout << "  Max threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "  Hardware concurrency: " << num_threads << std::endl;
    std::cout << "  OMP_NUM_THREADS env: " << (getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "not set") << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // // Program arguments parsing
    AttentionArgs args;
    if (!parse_attention_args(argc, argv, args)) {
        return 1;
    }
    
    // Note: multi-head attention implementation
    //NOTE: sync with the python version 
    int D_H = args.DH;      // dimension per head
    int num_head = args.NUM_HEAD;  // number of attention heads
    int LQ_per_CT = args.LQ_per_CT;
    int LK_per_CT = args.LK_per_CT;
    int LQ = LQ_per_CT;//TODO: should be LQ_per_CT* number of CT
    int LK = LK_per_CT;
    int NUM_OF_COLUMNS = args.NUM_OF_COLUMNS;
    int NUM_OF_CT_PER_COLUMN = args.NUM_OF_CT_PER_COLUMN;

    //NOTE: for now
    
    int L_SEQ = args.SHIM_L_SEQ; // query length
    int S_SEQ = args.SHIM_S_SEQ; // key/value length


    int S_SEQ_PADDED = ceil_div(S_SEQ, LK_per_CT)* LK_per_CT; // padded key/value length
    // print out all command arguments
    std::cout << "================ Command Line Arguments ================" << std::endl;
    
    std::cout << "Device: " << args.device << std::endl;
    std::cout << "r: " << args.r << std::endl;
    std::cout << "s: " << args.s << std::endl;
    std::cout << "t: " << args.t << std::endl;
    std::cout << "D_H: " << D_H << std::endl;
    std::cout << "LQ_per_CT: " << LQ_per_CT << std::endl;
    std::cout << "LK_per_CT: " << LK_per_CT << std::endl;
    std::cout << "lq_internal: " << args.lq_internal << std::endl;
    std::cout << "lk_lv_internal: " << args.lk_lv_internal << std::endl;
    std::cout << "NUM_HEAD: " << num_head << std::endl;
    std::cout << "SHIM_L_SEQ: " << L_SEQ << std::endl;
    std::cout << "SHIM_S_SEQ: " << S_SEQ << std::endl;
    std::cout << "NUM_OF_COLUMNS: " << NUM_OF_COLUMNS << std::endl;
    std::cout << "NUM_OF_CT_PER_COLUMN: " << NUM_OF_CT_PER_COLUMN << std::endl;
    std::cout << "CU_MODE: " << args.CU_MODE << std::endl;
    std::cout << "LOW_PRECISION: " << args.LOW_PRECISION << std::endl;
    std::cout << "========================================================" << std::endl;


    std::cout << "S_SEQ_PADDED is: " << S_SEQ_PADDED << std::endl;
    int total_dim = D_H * num_head; // total dimension


    // let us now allocate the buffer and randomly assign value to it
    std::bfloat16_t* q_buffer = (std::bfloat16_t*) malloc( L_SEQ * total_dim * sizeof(std::bfloat16_t));
    std::bfloat16_t* k_buffer = (std::bfloat16_t*) malloc( S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));
    std::bfloat16_t* v_buffer = (std::bfloat16_t*) malloc( S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));
    std::bfloat16_t* ref_attention_output_buffer = (std::bfloat16_t*) malloc( L_SEQ * total_dim * sizeof(std::bfloat16_t));
    std::bfloat16_t* ref_flash_attention_output_buffer = (std::bfloat16_t*) malloc( L_SEQ * total_dim * sizeof(std::bfloat16_t));


    {
        // assign random value  of -1 to 1
        float min_val = -5.0f;
        float max_val = 5.0f;
        for ( int i = 0; i < L_SEQ * total_dim; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            q_buffer[i] = std::bfloat16_t(val);
        }
        for ( int i = 0; i < S_SEQ * total_dim; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            k_buffer[i] = std::bfloat16_t(val);
        }
        for ( int i = 0; i < S_SEQ * total_dim; i++ ) {
            float val = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
            v_buffer[i] = std::bfloat16_t(val);
        }
    }

    // now, let us call the both implementation 
    std::vector<float> S_ij_row;
    std::vector<float> m_i;
    std::vector<float> m_i_new;
    std::vector<float> exp_old_vec;
    std::vector<float> l_i;
    std::vector<float> P_ij_unnorm_flat; 
    std::vector<float> O_accum;
    

    
    // ===== Benchmark AVX-512 Implementations =====
    std::cout << "===== AVX-512 OPTIMIZED IMPLEMENTATIONS =====" << std::endl;
    
    // Warmup and verify OpenMP is working
    std::cout << "Testing OpenMP parallel region..." << std::endl;
    int thread_count = 0;
    #pragma omp parallel
    {
        #pragma omp atomic
        thread_count++;
        
        #pragma omp barrier
        
        #pragma omp single
        {
            std::cout << "OpenMP parallel region has " << omp_get_num_threads() << " threads (atomic count: " << thread_count << ")" << std::endl;
        }
    }
    
    if (thread_count == 1) {
        std::cout << "WARNING: OpenMP is NOT working correctly! Only 1 thread detected." << std::endl;
        std::cout << "Try: export OMP_NUM_THREADS=8 before running" << std::endl;
    }
    
    std::bfloat16_t* avx512_flash_attention_output_buffer = (std::bfloat16_t*) malloc( L_SEQ * total_dim * sizeof(std::bfloat16_t));
    std::bfloat16_t* avx512_attention_output_buffer = (std::bfloat16_t*) malloc( L_SEQ * total_dim * sizeof(std::bfloat16_t));
    




    // to avoid doing redundant computation, we try to save the result into a file
    
    // filename consist of  random_seed_DH_num_head_SHIM_L_SEQ_SHIM_S_SEQ
    std::string avx512_flash_attention_output_file = "avx512_flash_attention_output_" +
     std::to_string(random_seed) + "_" + std::to_string(D_H) + "_"
      + std::to_string(num_head) + "_" + std::to_string(L_SEQ) + "_" + std::to_string(S_SEQ) + ".bin";
    // check if the file exists
    std::ifstream infile(output_data_dir + "/" + avx512_flash_attention_output_file, std::ios::binary);
    if (infile.good()) {
        std::cout << "AVX-512 Flash Attention output file found: " << avx512_flash_attention_output_file << std::endl;
        std::cout << "Skipping AVX-512 Flash Attention computation." << std::endl;
        // file exists, load the data
        std::cout << "Loading AVX-512 Flash Attention output from file: " << avx512_flash_attention_output_file << std::endl;
        infile.read( (char*)avx512_flash_attention_output_buffer, L_SEQ * total_dim * sizeof(std::bfloat16_t));
        infile.close();
    } else {
            // file does not exist, run the computation and save the result
        flash_attention_progressive_avx512(
            q_buffer,
            k_buffer,
            v_buffer,
            avx512_flash_attention_output_buffer,
            L_SEQ,
            S_SEQ,
            D_H,
            LQ,
            LK,
            num_head,
            S_ij_row,
            m_i,
            m_i_new,
            exp_old_vec,
            l_i,
            P_ij_unnorm_flat,
            O_accum
        );
        // save the result to file
        std::ofstream outfile(output_data_dir  +  "/" + avx512_flash_attention_output_file, std::ios::binary);
        outfile.write( (char*)avx512_flash_attention_output_buffer, L_SEQ * total_dim * sizeof(std::bfloat16_t));
        outfile.close();
        std::cout << "AVX-512 Flash Attention output saved to file: " << avx512_flash_attention_output_file << std::endl;
    }

    std::cout << "************************"<< std::endl;

    //load the npu instance

// NPU instance
    // npu_manager npu_instance(npu_device::device_npu2);
    // if (VERBOSE >= 1){
    //     npu_instance.get_npu_power(true);
    //     npu_instance.print_npu_info();
    // }

    std::string xclbin_name = "build/xclbins/attention_bf16.xclbin";
    // npu_app_desc accel_desc_0, accel_desc_1;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "attention_bf16";


    // npu_app app_0 = npu_instance.create_app(accel_desc_0);
    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* vision_attention_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = vision_attention_app_mgr->create_app();

    bool RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION = args.RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION;


    // because q, k, v buffer all in on buffer after MM
    constexpr int ARG_Q = 0;
    constexpr int ARG_K = 1;
    constexpr int ARG_V = 2;
    constexpr int ARG_O = 3;
    int batch_size = 1;

    std::vector<int> SHIM_L_SEQ_list = { args.SHIM_L_SEQ};
    std::vector<int> SHIM_S_SEQ_list = { args.SHIM_S_SEQ};
    std::vector<int> SHIM_S_SEQ_PADDED_list = { S_SEQ_PADDED};

    std::vector<int> Q_batch_offset_list = {0};
    std::vector<int> K_batch_offset_list = {0};
    std::vector<int> V_batch_offset_list = {0};
    std::vector<int> O_batch_offset_list = {0};

    bool REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH =false;
    AttentionConfig config = {
        (uint32_t)D_H, (uint32_t)num_head,
        LQ_per_CT, LK_per_CT,
        args.lq_internal, args.lk_lv_internal,
        NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN,
        args.CU_MODE,
        batch_size,

        ARG_Q, ARG_K, ARG_V, ARG_O,

        (uint32_t)(D_H * num_head), (uint32_t)(D_H * num_head), (uint32_t)(D_H * num_head),
        (uint32_t)(D_H * num_head),

        REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,
         args.SHIM_L_SEQ, S_SEQ_PADDED, S_SEQ_PADDED
    };

    BatchMetadata batch_data = {
        SHIM_L_SEQ_list, SHIM_S_SEQ_list, SHIM_S_SEQ_PADDED_list,
        Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
    };
    // app_0.instr_seq->from_file("build/insts/attention_bf16.txt");
    setup_SHM_configuration<std::bfloat16_t, std::bfloat16_t>(
        *app_0.seq(),
        config,
        batch_data,
        1.0f / sqrtf((float)D_H)
    );    







    buffer<std::bfloat16_t> q_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(L_SEQ * total_dim);
    buffer<std::bfloat16_t> k_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(S_SEQ_PADDED * total_dim);
    buffer<std::bfloat16_t> v_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(S_SEQ_PADDED * total_dim);
    buffer<std::bfloat16_t> output_buffer_npu = app_0.create_bo_buffer<std::bfloat16_t>(L_SEQ * total_dim);    
    

    {
        memcpy(q_buffer_npu.data(), q_buffer, L_SEQ * total_dim * sizeof(std::bfloat16_t));
        memcpy(k_buffer_npu.data(), k_buffer, S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));

        memcpy(v_buffer_npu.data(), v_buffer, S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));

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




    std::bfloat16_t* npu_output_ptr = output_buffer_npu.data() ;
    // print the first 10 value
    // for(int i = 0; i < 10; i++){
    //     std::cout << "NPU output[" << i << "] = " << (npu_output_ptr[i]) << ", " << " Reference output[" << i << "] = " << static_cast<float>(ref_flash_attention_output_buffer[i]) << "\n";

    // }
    print_error_metrics<std::bfloat16_t, std::bfloat16_t>(
        npu_output_ptr,
        avx512_flash_attention_output_buffer,
        1,
        L_SEQ,
        total_dim,
        L_SEQ,
        total_dim

    );


    std::cout << "************************"<< std::endl;
    q_buffer_npu.sync_to_device();
    k_buffer_npu.sync_to_device();
    v_buffer_npu.sync_to_device();
    int Iterations = 200;
    // runlist to check performance 
    npu_time.first = 0;

    auto start_overall = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < Iterations; i++){
         auto start = std::chrono::high_resolution_clock::now();        
        app_0(q_buffer_npu, k_buffer_npu, v_buffer_npu, output_buffer_npu);
        auto end = std::chrono::high_resolution_clock::now();
        npu_time.first +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();    
    }
    auto end_overall = std::chrono::high_resolution_clock::now();
    auto overall_time = std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall);
    std::cout << "Overall time for " << Iterations << " iterations: " << overall_time.count() << " us" <<std::endl;
    std::cout << "Average time per iteration (including overhead): " << overall_time.count() / Iterations << " us" <<std::endl;
    // now, run it
    output_buffer_npu.sync_from_device();   

    auto average_time_us = npu_time.first / Iterations;
    MSG_BOX_LINE(60, "Average NPU time over " << Iterations << " iterations: " << average_time_us << " us");
    MSG_BONDLINE(60);
    print_error_metrics<std::bfloat16_t, std::bfloat16_t>(
        output_buffer_npu.data(),
        avx512_flash_attention_output_buffer,
        1,
        L_SEQ,
        total_dim,
        L_SEQ,
        total_dim

    );

    // the csv file should be named of DH_NUM_HDEAD.csv
    std::string result_csv_file = output_data_dir + "/DH_" + std::to_string(D_H) + "_NUM_HEAD_" + std::to_string(num_head) + ".csv";
    // create the output_data_dir+ result_csv_file if not exists
    std::filesystem::path result_csv_path = result_csv_file;
    if (!std::filesystem::exists(result_csv_path)) {
        std::ofstream result_file_header;
        result_file_header.open(result_csv_file, std::ios::app);
        result_file_header << "Device,"
                           << "r,"
                           << "s,"
                           << "t,"
                           << "D_H,"
                           << "LQ_per_CT,"
                           << "LK_per_CT,"
                           << "lq_internal,"
                           << "lk_lv_internal,"
                           << "num_head,"
                           << "L_SEQ,"
                           << "S_SEQ,"
                           << "NUM_OF_COLUMNS,"
                           << "NUM_OF_CT_PER_COLUMN,"
                           << "CU_MODE,"
                           << "LOW_PRECISION,"
                           << "REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,"
                           << "RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION,"
                           << "Average_Time_us\n";
        result_file_header.close();
    }
    
    std::ofstream result_file;
    result_file.open(result_csv_path, std::ios::app);
    result_file << args.device << ","
                << args.r << ","
                << args.s << ","
                << args.t << ","
                << D_H << ","
                << LQ_per_CT << ","
                << LK_per_CT << ","
                << args.lq_internal << ","
                << args.lk_lv_internal << ","
                << num_head << ","
                << L_SEQ << ","
                << S_SEQ << ","
                << NUM_OF_COLUMNS << ","
                << NUM_OF_CT_PER_COLUMN << ","
                << args.CU_MODE << ","
                << args.LOW_PRECISION << ","
                << REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH << ","
                << RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION << ","
                << average_time_us << "\n";
    result_file.close();




    // // now, let us try the reorder version(TODO: for now, we don't check the error )
    // REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH =true;
    // std::cerr << "Running with REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH = true" << std::endl;
    // AttentionConfig config2 = {
    //     (uint32_t)D_H, (uint32_t)num_head,
    //     LQ_per_CT, LK_per_CT,
    //     args.lq_internal, args.lk_lv_internal,
    //     NUM_OF_COLUMNS, NUM_OF_CT_PER_COLUMN,
    //     args.CU_MODE,
    //     batch_size,

    //     ARG_Q, ARG_K, ARG_V, ARG_O,

    //     (uint32_t)(D_H * num_head), (uint32_t)(D_H * num_head), (uint32_t)(D_H * num_head),
    //     (uint32_t)(D_H * num_head),

    //     REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH,
    //      args.SHIM_L_SEQ, S_SEQ_PADDED, S_SEQ_PADDED
    // };

    // BatchMetadata batch_data2 = {
    //     SHIM_L_SEQ_list, SHIM_S_SEQ_list, SHIM_S_SEQ_PADDED_list,
    //     Q_batch_offset_list, K_batch_offset_list, V_batch_offset_list, O_batch_offset_list
    // };

    // setup_SHM_configuration<std::bfloat16_t, std::bfloat16_t>(
    //     *app_0.instr_seq,
    //     config2,
    //     batch_data2,
    //     RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION
    // );    


    // std::cout << "************************"<< std::endl;
    
    // // Given q, k, v buffer is L X DH*num_head, reorder to [num_head, L, DH]
    // {
    //     std::vector<std::bfloat16_t> q_reordered(L_SEQ * total_dim);
    //     for (int h = 0; h < num_head; ++h) {
    //         for (int l = 0; l < L_SEQ; ++l) {
    //             std::memcpy(&q_reordered[(h * L_SEQ + l) * D_H], 
    //                         &q_buffer[l * total_dim + h * D_H], 
    //                         D_H * sizeof(std::bfloat16_t));
    //         }
    //     }
    //     std::memcpy(q_buffer_npu.data(), q_reordered.data(), L_SEQ * total_dim * sizeof(std::bfloat16_t));

    //     std::vector<std::bfloat16_t> k_reordered(S_SEQ_PADDED * total_dim);
    //     for (int h = 0; h < num_head; ++h) {
    //         for (int s = 0; s < S_SEQ_PADDED; ++s) {
    //             std::memcpy(&k_reordered[(h * S_SEQ_PADDED + s) * D_H], 
    //                         &k_buffer[s * total_dim + h * D_H], 
    //                         D_H * sizeof(std::bfloat16_t));
    //         }
    //     }
    //     std::memcpy(k_buffer_npu.data(), k_reordered.data(), S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));

    //     std::vector<std::bfloat16_t> v_reordered(S_SEQ_PADDED * total_dim);
    //     for (int h = 0; h < num_head; ++h) {
    //         for (int s = 0; s < S_SEQ_PADDED; ++s) {
    //             std::memcpy(&v_reordered[(h * S_SEQ_PADDED + s) * D_H], 
    //                         &v_buffer[s * total_dim + h * D_H], 
    //                         D_H * sizeof(std::bfloat16_t));
    //         }
    //     }
    //     std::memcpy(v_buffer_npu.data(), v_reordered.data(), S_SEQ_PADDED * total_dim * sizeof(std::bfloat16_t));
    // }
    
    // q_buffer_npu.sync_to_device();
    // k_buffer_npu.sync_to_device();
    // v_buffer_npu.sync_to_device();

    // npu_time.first = 0;
  
    // for(int i = 0; i < Iterations; i++){
    //      auto start = std::chrono::high_resolution_clock::now();        
    //     app_0(q_buffer_npu, k_buffer_npu, v_buffer_npu, output_buffer_npu);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     npu_time.first +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();    
    // }
    // std::cout << "Overall time for " << Iterations << " iterations: " << overall_time.count() << " us" <<std::endl;
    // // now, run it
    // output_buffer_npu.sync_from_device();   

    // average_time_us = npu_time.first / Iterations;
    // MSG_BOX_LINE(60, "Average NPU time over " << Iterations << " iterations: " << average_time_us << " us");



    // result_file.open(result_csv_path, std::ios::app);
    // result_file << args.device << ","
    //             << args.r << ","
    //             << args.s << ","
    //             << args.t << ","
    //             << D_H << ","
    //             << LQ_per_CT << ","
    //             << LK_per_CT << ","
    //             << args.lq_internal << ","
    //             << args.lk_lv_internal << ","
    //             << num_head << ","
    //             << L_SEQ << ","
    //             << S_SEQ << ","
    //             << NUM_OF_COLUMNS << ","
    //             << NUM_OF_CT_PER_COLUMN << ","
    //             << args.CU_MODE << ","
    //             << args.LOW_PRECISION << ","
    //             << REORDER_OUTPUT_FROM_M_N_TO_NUM_DH_M_DH << ","
    //             << RUNTIME_SEQUNENCE_START_FROM_L_DIMENSION << ","
    //             << average_time_us << "\n";
    // result_file.close();



    return 0;
}

