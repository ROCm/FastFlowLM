// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
// Portions of this file consist of AI-generated content.

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

// #include "debug_utils.hpp"

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
#include "GateDeltaNetPrefillRuntimeSequence.hpp"
size_t get_file_size(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 0;
    }
    return file.tellg();
}


// S is stored row-major as DK x DV (i.e. S^T in the equation, since eq uses S as DV x DK)
// u = S_{t-1} * k  =>  shape: DV  (MVM: DV x DK  *  DK)
// S_t = alpha*(S_{t-1} - beta * u * k^T) + beta * v * k^T
//      = alpha*S_{t-1} + beta*(v - alpha*u)*k^T   <-- your current form, correct


void gate_delta_net_ref(
    float alpha, float beta,
    const std::bfloat16_t* _q,  // Query:  size DQ (asserted equal to DK)
    const std::bfloat16_t* k,   // Key:    size DK
    const std::bfloat16_t* v,   // Value:  size DV
    std::bfloat16_t* out,       // Output: size DV
    float* S,                   // State:  row-major DV x DK
    int DV, int DK
) {
    // Basic safety checks for null pointers
    assert(_q != nullptr);
    assert(k != nullptr);
    assert(v != nullptr);
    assert(out != nullptr);
    assert(S != nullptr);

    // Use DK for query since DQ == DK is enforced
    std::vector<std::bfloat16_t> q_scaled(DK);
    
    // first, do the Q * 1/sqrt(DK)
    for (int j = 0; j < DK; j++) {
        q_scaled[j] = std::bfloat16_t(static_cast<float>(_q[j]) * (1.0f / std::sqrt((float)DK)));
    }

    // Step 1: u = S_{t-1} * k,  shape: DV
    // Corresponds to S_{t-1} k_t in the equation
    std::vector<float> u(DV, 0.0f);
    for (int i = 0; i < DV; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < DK; ++j) {
            sum += S[i * DK + j] * static_cast<float>(k[j]);
        }
        u[i] = sum;
    }

    // Step 2: S_t = alpha * S_{t-1} + beta * (v - alpha*u) * k^T
    // Directly from expanding: S_{t-1} * (alpha*(I - beta*k*k^T)) + beta*v*k^T
    for (int i = 0; i < DV; ++i) {
        // w_i = beta * (v_i - alpha * u_i)  -- scalar per row
        float w = beta * (static_cast<float>(v[i]) - alpha * u[i]);
        for (int j = 0; j < DK; ++j) {
            float kj = static_cast<float>(k[j]);
            // alpha decays old state, w*kj adds rank-1 update
            S[i * DK + j] = alpha * S[i * DK + j] + w * kj;
        }
    }

    // Step 3: out = S_t * q,  shape: DV
    for (int i = 0; i < DV; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < DK; ++j) {
            sum += S[i * DK + j] * static_cast<float>(q_scaled[j]);
        }
        out[i] = static_cast<std::bfloat16_t>(sum);
    }
}


int roundup(int x, int m) {
    return ((x + m - 1) / m) * m;
}

// Test knobs read from the environment, so the positional IRON_ARGS stay unchanged.
// GDN_XCLBIN    : xclbin to load (to A/B a DMA_ONLY build against a full one)
// GDN_ITERS     : timed runs to average over
// GDN_ALPHA_MIN : smallest decay drawn; push towards 0 to stress the decay path
// GDN_ALPHA_MAX : largest decay drawn
// GDN_SAVE      : file to write the raw O and S outputs to
static std::string env_str(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::string(v) : fallback;
}
static float env_float(const char* name, float fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::stof(v) : fallback;
}

void initalize_alpha_beta_q_k_v_s(

    std::vector<float>& alpha_list,
    std::vector<float>& beta_list,
    std::vector<std::vector<std::bfloat16_t>>& q_list, // each head is size of lxDQ for num_qk_head
    std::vector<std::vector<std::bfloat16_t>>& k_list, // each head is size of lxDK for num_qk_head
    std::vector<std::vector<std::bfloat16_t>>& v_list, // each head is size of lxDV for num_head
    std::vector<std::vector<float>>& S_before_update_list, // each head is size of DVxDK for num_head
    int L_size,
    int num_head,
    int num_qk_head,
    int DQ, int DK, int DV

){



    for(int i = 0; i < num_head*L_size; i++){

        float min_val = env_float("GDN_ALPHA_MIN", 0.1f);
        float max_val = env_float("GDN_ALPHA_MAX", 0.99f);
        alpha_list[i] = ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
        beta_list[i] = ((float) rand() / RAND_MAX) * (0.99f - 0.1f) + 0.1f;
    }

    float min_val = -10.0f;
    float max_val = 10.0f;
    {
        // initalize q, k, v
        for(int i = 0; i < num_head; i++){
            if(i < num_qk_head){
                for(int j = 0; j < DK*L_size; j++){
                    float val_k =  ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
                    k_list[i][j] = std::bfloat16_t(val_k);
                }

                for(int j = 0; j < DQ*L_size; j++){
                    float val_q =  ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
                    q_list[i][j] = std::bfloat16_t(val_q);
                }
                
                // apply L2 norm on q, and k for each DK
                for(int l_idx = 0; l_idx < L_size; l_idx++){
                    std::bfloat16_t* q_ptr = q_list[i].data() + l_idx*DQ;
                    std::bfloat16_t* k_ptr = k_list[i].data() + l_idx*DK;

                    float q_energy = 0.0f;
                    for(int j = 0; j < DQ; j++){
                        float q = (float)q_ptr[j];
                        q_energy += q*q;
                    }
                    float q_energy_sqrt = std::sqrt(q_energy);
                    for(int j = 0; j < DQ; j++){
                        float q = (float)q_ptr[j];
                        q_ptr[j] = std::bfloat16_t(q / q_energy_sqrt);
                    }

                    // similary, do it on k norm
                    float k_energy = 0.0f;
                    for(int j = 0; j < DK; j++){
                        float k = (float)k_ptr[j];
                        k_energy += k*k;
                    }
                    float k_energy_sqrt = std::sqrt(k_energy);
                    for(int j = 0; j < DK; j++){
                        float k = (float)k_ptr[j];
                        k_ptr[j] = std::bfloat16_t(k / k_energy_sqrt);
                    }   


                }            
            }

            for(int j = 0; j < DV*L_size; j++){
                float val_v =  ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
                v_list[i][j] = std::bfloat16_t(val_v);
            }
        }
        // now random initalize S, note S and S_before_update_list will be used as reference
        min_val = -5.0f;
        max_val = 5.0f;        
        for(int i = 0; i < num_head; i++){
            for(int j = 0; j < DV * DK; j++){
                float val_S =  ((float) rand() / RAND_MAX) * (max_val - min_val) + min_val;
                S_before_update_list[i][j] = val_S;
            }
        }

    }


}


int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // // Program arguments parsing
    DeltaNetPrefillArgs args;
    if (!parse_matmul_args(argc, (char**)argv, args)){
        return 1;
    }

    // print out all the arguments in here 
    std::cout << "Parsed arguments:" << std::endl;
    std::cout << "Architecture: " << args.arch << std::endl;
    std::cout << "Trace size: " << args.trace_size << std::endl;
    std::cout << "Sequence length (L): " << args.L << std::endl;
    std::cout << "Dimension of Query/Key (DQ): " << args.DQ << std::endl;
    std::cout << "Dimension of Key/Value (DK): " << args.DK << std::endl;
    std::cout << "Dimension of Value (DV): " << args.DV << std::endl;
    std::cout << "Number of row tiles: " << args.NUM_OF_ROW << std::endl;
    std::cout << "Number of column tiles: " << args.NUM_OF_COLUMN << std::endl;
    std::cout << "L chunk size for MT " << args.MT_L_chunk_size << std::endl;
    std::cout << "L chunk size for each CT: " << args.CT_L_chunk_size << std::endl;
    std::cout << "Number of attention heads: " << args.NUM_HEAD << std::endl;
    std::cout << "Number of head groups that share the same Q/K: " << args.QK_HEAD_GROUP << std::endl;
    std::cout << "Whether the qkv buffer is in Lx (  DQ*NUM_Q_K_HEAD + DV*NUM_Q_K_HEAD+ DV*NUM_HEAD) row major (1) or in  LxDQ*NUM_Q_K_HEAD + L*DK*NUM_Q_K_HEAD + L*DV*NUM_HEAD row major (0, default): " << args.QKV_buffer_L_head_row_major << std::endl;





    // npu_manager npu_instance(npu_device::device_npu2);
    // if (VERBOSE >= 1){
    //     npu_instance.get_npu_power(true);
    //     npu_instance.print_npu_info();
    // }
    std::string xclbin_name = env_str("GDN_XCLBIN", "build/xclbins/GateDeltaNet_prefill.xclbin");
    std::cout << "xclbin: " << xclbin_name << std::endl;
    // npu_app_desc accel_desc_0, accel_desc_1;
    // accel_desc_0.xclbin_name = xclbin_name;
    // accel_desc_0.app_name = "MVM_q80";


    xrt::device npu_device = xrt::device(0); // NPU device should always be 0
    
    npu_xclbin_manager xclbin_mgr(npu_device::device_npu2, &npu_device, false);
    npu_app_manager* mm_app_mgr = xclbin_mgr.register_xclbin(xclbin_name);

    npu_app app_0 = mm_app_mgr->create_app();



    const int num_head = args.NUM_HEAD;
    const int L_size = args.L;
    const int DQ = args.DQ;
    const int DK = args.DK;
    const int DV = args.DV;
    assert(num_head % args.QK_HEAD_GROUP == 0);
    assert(DQ == DK && "DQ and DK must be equal for DeltaNet query-state projection");
    const int num_qk_head = num_head/ args.QK_HEAD_GROUP;




    std::vector<float> alpha_list((size_t)L_size*num_head); // lx num_head, in row major order
    std::vector<float> beta_list( (size_t)L_size*num_head); // lx num_head, in row major order


    std::vector<std::vector<float>> S_after_update_list(num_head, std::vector<float>((size_t)DV * DK));   // num_head x (DVxDK) in row major order
    std::vector<std::vector<float>> S_before_update_list(num_head, std::vector<float>((size_t)DV * DK)); // num_head x (DVxDK) in row major order


    std::vector<std::vector<std::bfloat16_t>> q_list(num_qk_head, std::vector<std::bfloat16_t>((size_t)L_size*DQ)); // num_head/num_qk_head x (lxdim), in row-major order               
    std::vector<std::vector<std::bfloat16_t>> k_list(num_qk_head, std::vector<std::bfloat16_t>((size_t)L_size*DK)); // num_head/num_qk_head x (lxdim), in row-major order            
    std::vector<std::vector<std::bfloat16_t>> v_list(num_head, std::vector<std::bfloat16_t>((size_t)L_size*DV));    // num_head/num_qk_head x (lxdim), in row-major order         
    std::vector<std::vector<std::bfloat16_t>> out_list(num_head, std::vector<std::bfloat16_t>((size_t)L_size*DV));  // num_head/num_qk_head x (lxdim), in row-major order           



    initalize_alpha_beta_q_k_v_s(
        alpha_list, beta_list,
        q_list, k_list, v_list,
        S_before_update_list,
        L_size,
        num_head,num_qk_head,  DQ, DK, DV
    );

    // memcpy S_before_update_list to S_after_update_list as the initial state for S, since S will be updated in place on npu, and we also need the S_before_update_list for reference
    for(int h = 0; h < num_head; h++){
        memcpy(S_after_update_list[h].data(), S_before_update_list[h].data(), (size_t)DV * DK * sizeof(float));
    }


    // every v head owns its own state, so the heads are independent: one thread per head
    #pragma omp parallel for schedule(dynamic)
    for(int v_head_idx = 0; v_head_idx < num_head; v_head_idx++){
        {
            int q_head_idx = v_head_idx/args.QK_HEAD_GROUP;
            int k_head_idx = v_head_idx/args.QK_HEAD_GROUP;

            for(int l_idx = 0; l_idx < L_size; l_idx++){
                float* alpha_ptr = alpha_list.data() + v_head_idx + l_idx*(num_head);
                float* beta_ptr = beta_list.data() + v_head_idx + l_idx*(num_head);

                std::bfloat16_t * q_ptr = q_list[q_head_idx].data() + l_idx*DQ;
                std::bfloat16_t * k_ptr = k_list[k_head_idx].data() + l_idx*DK;
                std::bfloat16_t * v_ptr = v_list[v_head_idx].data() + l_idx*DV;
                std::bfloat16_t * out_ptr = out_list[v_head_idx].data() + l_idx*DV; 

                float* S_ptr = S_after_update_list[v_head_idx].data();
                // single step iteration update
                gate_delta_net_ref(
                    *alpha_ptr, *beta_ptr, 
                    q_ptr, k_ptr, v_ptr, out_ptr,
                    S_ptr, DV, DK
                );
            }


        }

    }

    const int L_padded = roundup(L_size, args.MT_L_chunk_size);
    std::cout <<"L_size: " << L_size << ", L_padded: " << L_padded << std::endl;




    // app_0.instr_seq->from_file("build/insts/GateDeltaNet_prefill.txt");
    // buffer<float> seq_0 = app_0.instr_seq->dump().cast_to<float>();
    bool IS_QKV_buffer_L_head_row_major = args.QKV_buffer_L_head_row_major==1? true : false;
    generate_gate_delta_net_prefill_sequence(
        *app_0.seq(),
        args.NUM_OF_COLUMN, args.NUM_OF_ROW,
        args.QK_HEAD_GROUP, args.NUM_HEAD,
        args.MT_L_chunk_size, L_size, L_padded,
        DQ, DK, DV,
        IS_QKV_buffer_L_head_row_major,
        0,0,0

    );




    buffer<float> S_buffer = app_0.create_bo_buffer<float>(
        (size_t)num_head * DV * DK
    ); 
    size_t qkv_buffer_row_stride = (num_qk_head* DQ + num_qk_head*DK + num_head*DV);
    buffer<std::bfloat16_t> qkv_buffer = app_0.create_bo_buffer<std::bfloat16_t>(
        L_padded * qkv_buffer_row_stride +  2* L_padded* num_head *(sizeof(float)/sizeof(std::bfloat16_t)) // plus 2 for scalar
                                            // 2*L_padded*num_head is for alpha and beta
    );
    buffer<float> alpha_beta_buffer = app_0.create_bo_buffer<float>(
        (size_t)2* L_padded* num_head// for alpha and beta
    );

    buffer<std::bfloat16_t> O_buffer = app_0.create_bo_buffer<std::bfloat16_t>(
        L_padded* num_head*DV
    );

    // do a cold start to load xclbin
    S_buffer.sync_to_device();
    qkv_buffer.sync_to_device();
    O_buffer.sync_to_device();    
    app_0(S_buffer, qkv_buffer, O_buffer);

    
    std::vector<std::vector<float>> S_before_update_col_major_list(num_head, std::vector<float>((size_t)DV * DK)); // num_head x (DVxDK) in column major order
    std::vector<std::vector<float>> S_after_update_col_major_list(num_head, std::vector<float>((size_t)DV *DK));
    for(int h = 0; h< num_head; h++){
        for(int i = 0; i < DV; i++){
            for(int j = 0; j < DK; j++){
                S_before_update_col_major_list[h][j * DV + i] = S_before_update_list[h][i * DK + j];
                S_after_update_col_major_list[h][j * DV + i] = S_after_update_list[h][i * DK + j];
            }
        }
    }


    // Now, for the bo buffer
    // S_buffer is num_head, DV*DK, in column major order


   
    // if qkv_buffer_in_row_majro is true, then
    // qkv_buffer is L x (num_qk_head* DQ + num_qk_head*DK + num_head*DV), in row major order, with padding in L dimension
    
    // else:
    // qkv buffer is (num_qk_head) *(LX Dq) + (num_qk_head) *(Lx DK) + (num_head) *(Lx DV), in row major order, with padding in L dimension
    
    // alpha_beta_buffer is 2x L x num_head, in row major order, with padding in L dimension
    // O_buffer is L x num_head*DV, in row major order, with padding in L dimension

    // now, copy S_before_update_col_major_list to S_buffer
    for(int h = 0; h < num_head; h++){
        size_t offset = (size_t)h * DV * DK;
        memcpy(S_buffer.data() + offset, S_before_update_col_major_list[h].data(), (size_t)DV * DK * sizeof(float));
    }


    if(IS_QKV_buffer_L_head_row_major){

        // now, copy q,k, v to qkv_buffer, note the order in qkv_buffer is [q_head0, q_head1,..., k_head0, k_head1,..., v_head0, v_head1,...], and each head's q/k/v is stored in row major order with padding in L dimension
        for(int q_h_idx = 0; q_h_idx< num_qk_head; q_h_idx++){
            for(int l_idx = 0; l_idx < L_size; l_idx++){
                size_t q_offset = (size_t)q_h_idx* DQ + l_idx*qkv_buffer_row_stride;
                size_t k_offset = num_qk_head* DQ + q_h_idx* DK + l_idx* qkv_buffer_row_stride;
                memcpy(qkv_buffer.data() + q_offset, q_list[q_h_idx].data() + l_idx*DQ, DQ * sizeof(std::bfloat16_t));
                memcpy(qkv_buffer.data() + k_offset, k_list[q_h_idx].data() + l_idx*DK, DK * sizeof(std::bfloat16_t));
            }
        }
        for(int v_h_idx = 0; v_h_idx < num_head; v_h_idx++){
            for(int l_idx = 0; l_idx < L_size; l_idx++){
                size_t v_offset = num_qk_head* DQ + num_qk_head* DK + v_h_idx* DV + l_idx* qkv_buffer_row_stride;
                memcpy(qkv_buffer.data() + v_offset, v_list[v_h_idx].data() + l_idx*DV, DV * sizeof(std::bfloat16_t));
            }
        }

    }else{

        for(int q_h_idx = 0; q_h_idx < num_qk_head; q_h_idx++){
            for(int l_idx =0; l_idx < L_size; l_idx++ ){
                size_t q_offset = q_h_idx * (L_padded *DQ) + l_idx* DQ;
                size_t k_offset = num_qk_head* (L_padded* DQ) + q_h_idx * (L_padded* DK) + l_idx* DK;
                

                memcpy( qkv_buffer.data() + q_offset, q_list[q_h_idx].data() + l_idx*DQ, DQ * sizeof(std::bfloat16_t));
                memcpy( qkv_buffer.data() + k_offset, k_list[q_h_idx].data() + l_idx*DK, DK * sizeof(std::bfloat16_t));
            }
        }
        for(int v_h_idx = 0; v_h_idx< num_head; v_h_idx++){
            for(int l_idx = 0; l_idx < L_size; l_idx++){
                size_t v_offset = num_qk_head* (L_padded* DQ + (L_padded* DK))  + v_h_idx * (L_padded* DV) + l_idx* DV;
                memcpy( qkv_buffer.data() + v_offset, v_list[v_h_idx].data() + l_idx*DV, DV * sizeof(std::bfloat16_t));
            }
        }


    }

    float *alpha_beta_buffer_ptr =(float*)( alpha_beta_buffer.data() );
    // now, copy alpha, beta 
    for(int h = 0; h < num_head; h++){
        for(int l_idx = 0; l_idx < L_size; l_idx++){
            size_t alpha_offset = h + l_idx* num_head;
            size_t beta_offset = alpha_offset + (size_t)L_padded* num_head;
            alpha_beta_buffer_ptr[alpha_offset] = alpha_list[h + l_idx* num_head];
            alpha_beta_buffer_ptr[beta_offset] = beta_list[h + l_idx* num_head];

        }
    }



    int Iterations = std::max(1, std::stoi(env_str("GDN_ITERS", "1")));
    time_utils::time_with_unit npu_time = {0.0, "us"};


    S_buffer.sync_to_device();
    qkv_buffer.sync_to_device();
    alpha_beta_buffer.sync_to_device();
    O_buffer.sync_to_device();

    // The kernel updates S in place, so the timed runs leave S advanced by Iterations
    // scans. Time first, then put the starting S back and run once more for the check.
    for(int it = 0; it < Iterations; it++){
        auto start = std::chrono::high_resolution_clock::now();
        app_0(S_buffer, qkv_buffer,alpha_beta_buffer, O_buffer);
        auto end = std::chrono::high_resolution_clock::now();
        npu_time.first +=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    if(Iterations > 1){
        for(int h = 0; h < num_head; h++){
            memcpy(S_buffer.data() + (size_t)h * DV * DK, S_before_update_col_major_list[h].data(), (size_t)DV * DK * sizeof(float));
        }
        S_buffer.sync_to_device();
        app_0(S_buffer, qkv_buffer,alpha_beta_buffer, O_buffer);
    }

    O_buffer.sync_from_device();
    S_buffer.sync_from_device();


    auto average_time_us = npu_time.first/Iterations;   


    // prin the time
    MSG_BONDLINE(60);
    MSG_BOX_LINE(60, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BOX_LINE(60, "Average NPU time per iteration: " << average_time_us << " us");
    MSG_BONDLINE(60);    


    
    int number_to_print = 32;
    int num_head_to_debug = 0;
    int l_idx_to_debug = 0;
    // std::cout << "First 32 elements of Q_buffer from O_buffer: ";
    // for(int i = 0; i < number_to_print; i++){
    //     std::cout << O_buffer.data()[i + 128*num_head   ] << " ";
    // }
    // std::cout << std::endl;

    // // print first 5 element from out_list[0]
    // std::cout << "First 32 elements Q_head_0: ";
    // for(int i = 0; i <number_to_print; i++){
    //     std::cout << q_list[0][i+128] << " ";

    // }
    // std::cout << std::endl;

    // std::cout << "First 32 elements Q_head_0: ";
    // for(int i = 0; i <number_to_print; i++){
    //     std::cout << qkv_buffer[ qkv_buffer_row_stride+i  ] << " ";

    // }
    // std::cout << std::endl;





    // std::cout << "First 32 elements of Q_buffer from O_buffer: ";
    // for(int i = 0; i < number_to_print; i++){
    //     std::cout << O_buffer.data()[i + 128*num_head   ] << " ";
    // }
    // std::cout << std::endl;

    // // print first 5 element from out_list[0]
    // std::cout << "First 32 elements Q_head_0: ";
    // for(int i = 0; i <number_to_print; i++){
    //     std::cout << k_list[0][i+128] << " ";

    // }
    // std::cout << std::endl;

    // std::cout << "First 32 elements Q_head_0: ";
    // for(int i = 0; i <number_to_print; i++){
    //     std::cout << qkv_buffer[ num_qk_head*DQ +   qkv_buffer_row_stride+i  ] << " ";

    // }
    // std::cout << std::endl;





    // print first 5 element from O_buffer for quick check
    std::cout << "First 5 elements of O_buffer: ";
    for(int i = 0; i <32; i++){
        std::cout << O_buffer.data()[i+128*2] << " ";
    }
    std::cout << std::endl;

    // print first 5 element from out_list[0]
    std::cout << "First 5 elements of out_list[0]: ";
    for(int i = 0; i < 32; i++){
        std::cout << out_list[0][i+128] << " ";
    }
    std::cout << std::endl;
    


    for(int head_idx = 0; head_idx < num_head; head_idx ++){

        std::cout << "printing head " << head_idx << " S matrix after update: " << std::endl;


        // for first head
        print_error_metrics<std::bfloat16_t, std::bfloat16_t>(
            O_buffer.data()+head_idx*DV, out_list[head_idx].data(),
            1, 
            L_size, DV,
            L_size, num_head*DV
        );


        // compare S
        print_error_metrics<float, float>(
            S_buffer.data()+ DV*DK*head_idx, S_after_update_col_major_list[head_idx].data(),
            1, 
            DV, DK,
            DV, DK
        );


    }

    // GDN_DUMP=1: first state values of head 0 (before / NPU / reference), to see
    // the shape of an error rather than just its size
    if(env_str("GDN_DUMP", "0") == "1"){
        for(int i = 0; i < 8; i++){
            std::cout << "S[" << i << "] before=" << S_before_update_col_major_list[0][i]
                      << " npu=" << S_buffer.data()[i]
                      << " ref=" << S_after_update_col_major_list[0][i] << std::endl;
        }
        std::cout << "alpha0=" << alpha_list[0] << " beta0=" << beta_list[0] << std::endl;
        if(env_str("GDN_DUMP_ROW", "0") == "1"){   // a debug kernel writes raw floats into output rows
            const float* dbg = reinterpret_cast<const float*>(O_buffer.data());
            std::cout << "row0 as floats:";
            for(int i = 0; i < 12; i++) std::cout << " " << dbg[i];
            std::cout << std::endl;
        }
        // a PROFILE=1 kernel writes cycle counts over the first 4 state words of the
        // head on core row 0: whole head, inside the passes, matvec + chunk scalars, tokens
        const uint32_t* prof = reinterpret_cast<const uint32_t*>(S_buffer.data());
        std::cout << "PROFILE head0 core row0: head_cycles=" << prof[0] << " pass_cycles=" << prof[1]
                  << " prep_cycles=" << prof[2] << " tokens=" << prof[3];
        if(prof[3] > 0){
            std::cout << "  per token: head=" << prof[0] / prof[3] << " pass=" << prof[1] / prof[3]
                      << " prep=" << prof[2] / prof[3]
                      << " rest=" << (prof[0] - prof[1] - prof[2]) / prof[3];
        }
        std::cout << std::endl;
    }

    // one line to grep: the worst head for each output
    float worst_o_cos = 1.0f, worst_s_cos = 1.0f, worst_o_l2 = 0.0f, worst_s_l2 = 0.0f;
    for(int head_idx = 0; head_idx < num_head; head_idx ++){
        std::bfloat16_t* o_got = O_buffer.data() + head_idx*DV;
        std::bfloat16_t* o_want = out_list[head_idx].data();
        float* s_got = S_buffer.data() + DV*DK*head_idx;
        float* s_want = S_after_update_col_major_list[head_idx].data();
        worst_o_cos = std::min(worst_o_cos, get_cosine_similarity(o_got, o_want, 1, L_size, DV, L_size, num_head*DV));
        worst_o_l2  = std::max(worst_o_l2,  get_relativeL2(o_got, o_want, 1, L_size, DV, L_size, num_head*DV));
        worst_s_cos = std::min(worst_s_cos, get_cosine_similarity(s_got, s_want, 1, DV, DK, DV, DK));
        worst_s_l2  = std::max(worst_s_l2,  get_relativeL2(s_got, s_want, 1, DV, DK, DV, DK));
    }
    std::cout << "SUMMARY L=" << L_size << " heads=" << num_head
              << " npu_us=" << average_time_us
              << " O_cos_min=" << worst_o_cos << " O_relL2_max=" << worst_o_l2
              << " S_cos_min=" << worst_s_cos << " S_relL2_max=" << worst_s_l2 << std::endl;

    // GDN_SAVE=<file>: raw O then S bytes, to compare two builds bit for bit
    const std::string save_path = env_str("GDN_SAVE", "");
    if (!save_path.empty()) {
        std::ofstream f(save_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(O_buffer.data()), O_buffer.size() * sizeof(std::bfloat16_t));
        f.write(reinterpret_cast<const char*>(S_buffer.data()), S_buffer.size() * sizeof(float));
    }

    return 0;
}

